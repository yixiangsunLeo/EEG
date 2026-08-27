#!/usr/bin/env python
"""Training diagnostics for the two SHU 2022 runs: accuracy, loss, weight histograms.

Accuracy and loss come from each cell's result.json ``history``.
Parameter distributions are read from ``best.pt`` (learnable tensors only;
BatchNorm running stats are skipped). Nothing here was logged during training
beyond the per-epoch metrics — weight histograms are a post-hoc snapshot of
the selected checkpoint.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT = Path("/share/home/yuan/SYX/eeg-mi-online")
OUT = PROJECT / "outputs/experiments/shu/training_viz_v1"

EXPS = [
    {
        "tag": "exp1_cross_session",
        "title": "Cross-session (3→2)",
        "run": PROJECT / "outputs/experiments/shu/cross_session_3to2_v1",
        "ckpt": PROJECT / "checkpoints/shu/cross_session_3to2_v1",
    },
    {
        "tag": "exp2_cross_subject",
        "title": "Cross-subject 5-fold",
        "run": PROJECT / "outputs/experiments/shu/cross_subject_5fold_v1",
        "ckpt": PROJECT / "checkpoints/shu/cross_subject_5fold_v1",
    },
]

MODELS = [
    "eegnet_official", "eegnex", "eeg_deformer", "atcnet",
    "deepconvnet", "fbcnet", "s4erp",
    "dualcd_s4_flatten", "dualcd_transformer",
]
LABEL = {
    "eegnet_official": "EEGNet",
    "eegnex": "EEGNeX",
    "eeg_deformer": "EEG-Deformer",
    "atcnet": "ATCNet",
    "deepconvnet": "DeepConvNet",
    "fbcnet": "FBCNet",
    "s4erp": "S4ERP",
    "dualcd_s4_flatten": "DualCD-S4-Flatten",
    "dualcd_transformer": "DualCD-Transformer",
}
CELL_TAG = re.compile(
    r"^(?P<model>.+?)__(?:split(?P<split>\d+)__)?"
    r"(?:fold(?P<fold>\d+)|(?P<subject>[^_]+(?:_[^_]+)*?))__seed(?P<seed>\d+)$"
)
SKIP_PARAM = ("running_mean", "running_var", "num_batches_tracked", "running_std")

BINS = np.linspace(-1.5, 1.5, 121)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "figure.dpi": 130,
        "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_model(cell_dir: str) -> Optional[str]:
    m = CELL_TAG.match(cell_dir)
    return m.group("model") if m else None


def load_histories(run: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in sorted((run / "cells").glob("*/result.json")):
        model = parse_model(p.parent.name)
        if model not in MODELS:
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[model].append({
            "cell": p.parent.name,
            "best_epoch": int((d.get("train_info") or {}).get("best_epoch") or 0),
            "history": d.get("history") or [],
        })
    return out


def mean_curve(histories: List[List[dict]], getter) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return xs, mean, lo, hi using cells that reached epoch t (need >=3)."""
    max_n = max((len(h) for h in histories), default=0)
    xs, means, lo, hi = [], [], [], []
    for t in range(max_n):
        vals = []
        for h in histories:
            if t >= len(h):
                continue
            v = getter(h[t])
            if v is not None and np.isfinite(v):
                vals.append(v)
        if len(vals) < 3:
            break
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        xs.append(t + 1)
        means.append(m)
        lo.append(m - s)
        hi.append(m + s)
    return np.array(xs), np.array(means), np.array(lo), np.array(hi)


def _acc(row, split: str) -> Optional[float]:
    v = (row.get(split) or {}).get("accuracy")
    return float(v) if v is not None else None


def _nll(row, split: str) -> Optional[float]:
    block = row.get(split) or {}
    v = block.get("nll")
    if v is None:
        v = block.get("val_loss")
    return float(v) if v is not None else None


def plot_acc_loss(exp_title: str, model: str, cells: List[dict], dest: Path) -> None:
    hists = [c["history"] for c in cells if c["history"]]
    if not hists:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    for key, label, color in [
        ("train_eval", "train (eval mode)", "tab:blue"),
        ("val", "validation (selection)", "tab:orange"),
        ("test", "test (monitor only)", "tab:green"),
    ]:
        xs, m, lo, hi = mean_curve(hists, lambda r, k=key: _acc(r, k))
        if xs.size == 0:
            continue
        ax.plot(xs, m, color=color, lw=1.7, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.16)
    ax.axhline(0.5, color="#C44E52", ls="--", lw=1.0, label="chance 50%")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.set_title(f"{LABEL[model]} — accuracy")
    ax.legend(fontsize=8)

    ax = axes[1]
    series = [
        (lambda r: float(r["train_loss"]) if r.get("train_loss") is not None else None,
         "train loss (train mode)", "tab:red"),
        (lambda r: _nll(r, "train_eval"), "train NLL (eval mode)", "tab:blue"),
        (lambda r: _nll(r, "val"), "val NLL", "tab:orange"),
        (lambda r: _nll(r, "test"), "test NLL (monitor)", "tab:green"),
    ]
    for getter, label, color in series:
        xs, m, lo, hi = mean_curve(hists, getter)
        if xs.size == 0:
            continue
        ax.plot(xs, m, color=color, lw=1.7, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.16)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / NLL")
    ax.set_title(f"{LABEL[model]} — loss")
    ax.legend(fontsize=8)
    fig.suptitle(f"{exp_title}  ·  mean ± std over {len(hists)} cells", fontsize=11, y=1.02)
    save(fig, dest / f"acc_loss_mean__{model}.png")


def plot_one_cell(exp_title: str, model: str, cell: dict, dest: Path) -> None:
    h = cell["history"]
    if not h:
        return
    epochs = [int(r["epoch"]) for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    for key, label, color in [
        ("train_eval", "train", "tab:blue"),
        ("val", "val", "tab:orange"),
        ("test", "test (monitor)", "tab:green"),
    ]:
        ys = [_acc(r, key) for r in h]
        ax.plot(epochs, ys, color=color, lw=1.4, label=label)
    be = cell.get("best_epoch")
    if be:
        ax.axvline(be, color="grey", ls="--", lw=1.0, label=f"best.pt epoch {be}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.2, 1.05)

    ax = axes[1]
    ax.plot(epochs, [r.get("train_loss") for r in h], color="tab:red", lw=1.4, label="train loss")
    ax.plot(epochs, [_nll(r, "val") for r in h], color="tab:orange", lw=1.4, label="val NLL")
    ax.plot(epochs, [_nll(r, "test") for r in h], color="tab:green", lw=1.4, label="test NLL")
    if be:
        ax.axvline(be, color="grey", ls="--", lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / NLL")
    ax.set_title("loss")
    ax.legend(fontsize=8)
    fig.suptitle(f"{exp_title}  ·  {LABEL[model]}  ·  {cell['cell']}", fontsize=10, y=1.02)
    save(fig, dest / "examples" / f"acc_loss__{cell['cell']}.png")


def is_learnable_key(name: str) -> bool:
    leaf = name.split(".")[-1]
    if leaf in SKIP_PARAM:
        return False
    # DualCD teacher is an EMA copy (requires_grad=False); not independently learned.
    low = name.lower()
    if "teacher" in low:
        return False
    return True


def module_group(name: str) -> str:
    parts = [p for p in name.split(".") if p not in {"_orig_mod"}]
    if parts and parts[0] in {"net", "model", "backbone", "module"}:
        return ".".join(parts[:2]) if len(parts) > 1 else parts[0]
    return parts[0] if parts else name


def collect_params(ckpt_root: Path, model: str):
    """Accumulate a pooled histogram + per-group RMS over all best.pt of this model."""
    import torch

    counts = np.zeros(len(BINS) - 1, dtype=np.float64)
    n_weights = 0
    n_cells = 0
    # group -> list of per-cell rms
    group_rms: Dict[str, List[float]] = defaultdict(list)
    group_mean: Dict[str, List[float]] = defaultdict(list)
    group_std: Dict[str, List[float]] = defaultdict(list)
    group_frac_small: Dict[str, List[float]] = defaultdict(list)
    overall = []  # per-cell (mean, std, rms, p01, p99, frac_|w|<1e-3)

    paths = sorted(ckpt_root.glob(f"{model}__*/best.pt"))
    for pt in paths:
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        state = ck.get("model_state") or {}
        by_g: Dict[str, List] = defaultdict(list)
        flats = []
        for k, v in state.items():
            if not is_learnable_key(k):
                continue
            if not hasattr(v, "dtype") or not v.dtype.is_floating_point:
                continue
            t = v.detach().float().reshape(-1)
            if t.numel() == 0:
                continue
            flats.append(t)
            by_g[module_group(k)].append(t)
        if not flats:
            continue
        cat = torch.cat(flats)
        hist, _ = np.histogram(cat.numpy(), bins=BINS)
        counts += hist
        n_weights += int(cat.numel())
        n_cells += 1
        abs_cat = cat.abs()
        overall.append({
            "cell": pt.parent.name,
            "n": int(cat.numel()),
            "mean": float(cat.mean()),
            "std": float(cat.std(unbiased=False)),
            "rms": float(torch.sqrt((cat * cat).mean())),
            "p01": float(torch.quantile(cat, 0.01)),
            "p99": float(torch.quantile(cat, 0.99)),
            "frac_abs_lt_1e3": float((abs_cat < 1e-3).float().mean()),
            "frac_abs_lt_1e2": float((abs_cat < 1e-2).float().mean()),
            "max_abs": float(abs_cat.max()),
        })
        for g, ts in by_g.items():
            gcat = torch.cat(ts)
            group_rms[g].append(float(torch.sqrt((gcat * gcat).mean())))
            group_mean[g].append(float(gcat.mean()))
            group_std[g].append(float(gcat.std(unbiased=False)))
            group_frac_small[g].append(float((gcat.abs() < 1e-3).float().mean()))
        del ck, state, flats, cat
    return {
        "counts": counts, "n_weights": n_weights, "n_cells": n_cells,
        "overall": overall, "group_rms": group_rms, "group_mean": group_mean,
        "group_std": group_std, "group_frac_small": group_frac_small,
    }


def plot_param_hist(exp_title: str, pooled: Dict[str, dict], dest: Path) -> None:
    present = [m for m in MODELS if (pooled.get(m) or {}).get("n_cells", 0) > 0]
    if not present:
        return
    n = len(present)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    width = BINS[1] - BINS[0]
    for i, model in enumerate(present):
        ax = axes[i]
        d = pooled[model]
        c = d["counts"]
        ax.bar(centers, c / c.sum(), width=width, color="#4C78A8", edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(f"{LABEL[model]}  n={d['n_cells']} ckpts")
        ax.set_xlabel("weight value")
        if i % ncols == 0:
            ax.set_ylabel("fraction (log)")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{exp_title} — learnable parameter values in best.pt "
                 f"(pooled over cells; BN running stats / DualCD teacher excluded)",
                 fontsize=11)
    save(fig, dest / "params_hist_all_models.png")


def plot_param_layers(exp_title: str, model: str, d: dict, dest: Path) -> None:
    groups = sorted(d["group_rms"].keys(), key=lambda g: -np.mean(d["group_rms"][g]))
    if not groups:
        return
    # keep at most 18 groups so the bar chart stays readable
    groups = groups[:18]
    rms = [float(np.mean(d["group_rms"][g])) for g in groups]
    fig, ax = plt.subplots(figsize=(8.0, max(3.6, 0.32 * len(groups) + 1.4)))
    y = np.arange(len(groups))
    ax.barh(y, rms, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMS of weights (mean over cells)")
    ax.set_title(f"{exp_title} · {LABEL[model]} · per-module weight RMS in best.pt")
    save(fig, dest / f"params_layers__{model}.png")


def write_param_csv(pooled: Dict[str, dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        cells = pooled[model]["overall"]
        if not cells:
            continue
        rows.append({
            "model": model,
            "n_cells": len(cells),
            "n_params": cells[0]["n"],
            "mean_of_means": float(np.mean([c["mean"] for c in cells])),
            "mean_of_stds": float(np.mean([c["std"] for c in cells])),
            "mean_rms": float(np.mean([c["rms"] for c in cells])),
            "mean_p01": float(np.mean([c["p01"] for c in cells])),
            "mean_p99": float(np.mean([c["p99"] for c in cells])),
            "mean_frac_|w|<1e-3": float(np.mean([c["frac_abs_lt_1e3"] for c in cells])),
            "mean_frac_|w|<1e-2": float(np.mean([c["frac_abs_lt_1e2"] for c in cells])),
            "mean_max_abs": float(np.mean([c["max_abs"] for c in cells])),
        })
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_readme(path: Path) -> None:
    path.write_text(
        """# 训练过程可视化（acc / loss / 参数分布）

这两套 SHU 2022 实验训练时**只把曲线写进了 `result.json`，当时没有自动出图**。
参数直方图训练过程中**没有记录**，这里是对每个 cell 的 `best.pt` 做的事后快照。

## 有什么

每个实验子目录（`exp1_cross_session/`、`exp2_cross_subject/`）：

- `acc_loss_mean__<model>.png` — 该模型 25 个 cell 的 **准确率** 和 **loss/NLL** 均值±标准差
  - 左：train（eval 模式）/ val（选模）/ test（只监控）准确率
  - 右：train loss（训练模式，含 dropout）以及 train/val/test 的 NLL
- `examples/` — 每个模型一张单 cell 例子（实验一 sub-001，实验二 split41 fold0）
- `params_hist_all_models.png` — 7 个模型在 `best.pt` 里所有可学习参数的数值直方图
- `params_layers__<model>.png` — 按模块名聚合的权重 RMS
- `params_summary.csv` — 每个模型的 mean / std / RMS / 分位数

## 怎么读

- 测试曲线**从未**用于早停或选 `best.pt`。
- `train loss` 和 `train NLL (eval)` 不是同一件事：前者带 dropout，后者是 eval 模式。
- 参数直方图的横轴截在 [-1.5, 1.5]；个别层（例如未约束的 bias）可能超出，见 CSV 的 `mean_max_abs`。
- BatchNorm 的 `running_mean` / `running_var` 不是可学习参数，已排除。
""",
        encoding="utf-8",
    )


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    write_readme(OUT / "README.md")

    print("plotting acc / loss from result.json")
    for exp in EXPS:
        dest = OUT / exp["tag"]
        dest.mkdir(parents=True, exist_ok=True)
        by_model = load_histories(exp["run"])
        print(f"  {exp['tag']}: " + ", ".join(f"{m}={len(by_model[m])}" for m in MODELS))
        for model in MODELS:
            cells = by_model.get(model) or []
            plot_acc_loss(exp["title"], model, cells, dest)
            # one example cell, stable choice
            example = None
            for c in cells:
                if "sub-001" in c["cell"] or "split41__fold0" in c["cell"]:
                    example = c
                    break
            if example is None and cells:
                example = cells[0]
            if example:
                plot_one_cell(exp["title"], model, example, dest)

    print("loading best.pt for parameter histograms (slow)")
    import torch  # noqa: F401  — import once here so acc/loss plots don't wait on CUDA

    for exp in EXPS:
        dest = OUT / exp["tag"]
        pooled = {}
        for model in MODELS:
            print(f"  {exp['tag']} {model}", flush=True)
            pooled[model] = collect_params(exp["ckpt"], model)
            print(f"    cells={pooled[model]['n_cells']} weights_pooled={pooled[model]['n_weights']}",
                  flush=True)
            plot_param_layers(exp["title"], model, pooled[model], dest)
        plot_param_hist(exp["title"], pooled, dest)
        write_param_csv(pooled, dest / "params_summary.csv")

    print("wrote", OUT)


if __name__ == "__main__":
    main()

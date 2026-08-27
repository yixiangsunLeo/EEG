#!/usr/bin/env python
"""Copy SHU 2022 exp1/exp2 results into a handoff folder and build tables + figures.

Does not copy checkpoints (best.pt / last.pt, ~2.4 GB). Metrics and per-epoch
curves live in each cell's result.json and are included.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT = Path("/share/home/yuan/SYX/eeg-mi-online")
DEST = Path("/share/home/yuan/SYX/SHU2022_两次实验结果_交给学长_20260820")

EXP1_SRC = PROJECT / "outputs/experiments/shu/cross_session_3to2_v1"
EXP2_SRC = PROJECT / "outputs/experiments/shu/cross_subject_5fold_v1"

MODELS = [
    "eegnet_official",
    "eegnex",
    "eeg_deformer",
    "atcnet",
    "deepconvnet",
    "fbcnet",
    "s4erp",
    "dualcd_s4_flatten",
    "dualcd_transformer",
]
MODEL_LABEL = {
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
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece"]
CHANCE = 0.50
PAPER_CS_P05 = 0.516  # Ma et al. Sci Data 9:531, p=0.05, N=2500

COLORS = {
    "train_eval": "tab:blue",
    "val": "tab:orange",
    "test": "tab:green",
}


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def pct(xs: List[float], thr: float) -> float:
    if not xs:
        return float("nan")
    return 100.0 * sum(1 for x in xs if x > thr) / len(xs)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def acc_at_epoch(history: List[Dict], epoch: int, split: str) -> Optional[float]:
    for row in history:
        if int(row.get("epoch", -1)) == int(epoch):
            block = row.get(split) or {}
            v = block.get("accuracy")
            return float(v) if v is not None else None
    return None


def series_acc(history: List[Dict], split: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in history:
        block = row.get(split) or {}
        v = block.get("accuracy")
        out.append(float(v) if v is not None else None)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Load cells
# ---------------------------------------------------------------------------

def load_exp1(src: Path) -> List[Dict[str, Any]]:
    cells = []
    for p in sorted((src / "cells").glob("*/result.json")):
        d = load_json(p)
        model = d["model"]
        subj = (d.get("fold") or {}).get("subject") or p.parent.name
        info = d.get("train_info") or {}
        hist = d.get("history") or []
        best_ep = int(info.get("best_epoch") or 0)
        pooled = (d.get("pooled_test") or {}).get("best") or {}
        last_p = (d.get("pooled_test") or {}).get("last") or {}
        ses = {}
        for item in d.get("per_session") or []:
            ses[item["test_session"]] = item
        row = {
            "model": model,
            "subject": subj,
            "cell": p.parent.name,
            "n_train": None,
            "n_val": None,
            "n_test_s4": (ses.get("ses-04") or {}).get("n_trials"),
            "n_test_s5": (ses.get("ses-05") or {}).get("n_trials"),
            "n_test_pooled": (d.get("pooled_test") or {}).get("n_trials"),
            "best_epoch": best_ep,
            "best_val": float(info.get("best_score") or float("nan")),
            "epochs_run": int(info.get("epochs_run") or 0),
            "early_stopped": bool(info.get("early_stopped")),
            "n_params": int((d.get("model_desc") or {}).get("n_params") or 0),
            "train_seconds": float(info.get("train_seconds") or 0),
            "train_acc_at_best": acc_at_epoch(hist, best_ep, "train_eval"),
            "val_acc_at_best": acc_at_epoch(hist, best_ep, "val"),
            "test_acc_at_best_curve": acc_at_epoch(hist, best_ep, "test"),
            "history": hist,
        }
        # n_train / n_val live on the per-session rows
        rows = d.get("rows") or []
        if rows:
            row["n_train"] = rows[0].get("n_train")
            row["n_val"] = rows[0].get("n_val")
        for m in METRICS:
            row[f"pooled_{m}"] = pooled.get(m)
            row[f"last_{m}"] = last_p.get(m)
            row[f"s4_{m}"] = ((ses.get("ses-04") or {}).get("best") or {}).get(m)
            row[f"s5_{m}"] = ((ses.get("ses-05") or {}).get("best") or {}).get(m)
        if row["train_acc_at_best"] is not None and row["val_acc_at_best"] is not None:
            row["overfit_gap"] = float(row["train_acc_at_best"]) - float(row["val_acc_at_best"])
        else:
            row["overfit_gap"] = None
        cells.append(row)
    return cells


def load_exp2(src: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (per-subject rows, per-cell rows)."""
    subjects: List[Dict[str, Any]] = []
    cells: List[Dict[str, Any]] = []
    for p in sorted((src / "cells").glob("*/result.json")):
        d = load_json(p)
        tag = p.parent.name  # model__splitSS__foldK__seed0
        # split seed is not inside result.json; parse the directory name
        split_seed = None
        fold_id = (d.get("fold") or {}).get("fold")
        parts = tag.split("__")
        for part in parts:
            if part.startswith("split") and part[5:].isdigit():
                split_seed = int(part[5:])
            if part.startswith("fold") and part[4:].isdigit():
                fold_id = int(part[4:])
        model = d["model"]
        info = d.get("train_info") or {}
        hist = d.get("history") or []
        best_ep = int(info.get("best_epoch") or 0)
        cell = {
            "model": model,
            "split_seed": split_seed,
            "fold": fold_id,
            "cell": tag,
            "n_train": None,
            "n_val": None,
            "best_epoch": best_ep,
            "best_val": float(info.get("best_score") or float("nan")),
            "epochs_run": int(info.get("epochs_run") or 0),
            "early_stopped": bool(info.get("early_stopped")),
            "n_params": int((d.get("model_desc") or {}).get("n_params") or 0),
            "train_seconds": float(info.get("train_seconds") or 0),
            "train_acc_at_best": acc_at_epoch(hist, best_ep, "train_eval"),
            "val_acc_at_best": acc_at_epoch(hist, best_ep, "val"),
            "test_acc_at_best_curve": acc_at_epoch(hist, best_ep, "test"),
            "history": hist,
        }
        if cell["train_acc_at_best"] is not None and cell["val_acc_at_best"] is not None:
            cell["overfit_gap"] = float(cell["train_acc_at_best"]) - float(cell["val_acc_at_best"])
        else:
            cell["overfit_gap"] = None
        for r in d.get("rows") or []:
            cell["n_train"] = r.get("n_train")
            cell["n_val"] = r.get("n_val")
            sub = {
                "model": model,
                "split_seed": split_seed,
                "fold": fold_id,
                "test_subject": r.get("test_subject"),
                "n_train": r.get("n_train"),
                "n_val": r.get("n_val"),
                "n_test": r.get("n_test"),
                "best_epoch": r.get("best_epoch"),
                "epochs_run": r.get("epochs_run"),
                "early_stopped": r.get("early_stopped"),
                "train_seconds": r.get("train_seconds"),
            }
            for m in METRICS:
                sub[m] = r.get(m)
            sub["last_accuracy"] = r.get("last_accuracy")
            sub["last_macro_f1"] = r.get("last_macro_f1")
            subjects.append(sub)
        cells.append(cell)
    return subjects, cells


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def exp1_tables(cells: List[Dict], out: Path) -> Dict[str, Any]:
    per_subj_fields = [
        "model", "subject", "n_train", "n_val", "n_test_s4", "n_test_s5", "n_test_pooled",
        "best_epoch", "best_val", "epochs_run", "early_stopped", "n_params", "train_seconds",
        "train_acc_at_best", "val_acc_at_best", "test_acc_at_best_curve", "overfit_gap",
        "pooled_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1", "pooled_auc",
        "pooled_nll", "pooled_brier", "pooled_ece",
        "s4_accuracy", "s5_accuracy", "s4_macro_f1", "s5_macro_f1",
        "last_accuracy", "last_macro_f1",
    ]
    write_csv(out / "exp1_per_subject.csv", cells, per_subj_fields)

    summary_rows = []
    headline = {}
    for model in MODELS:
        xs = [c for c in cells if c["model"] == model]
        acc = [float(c["pooled_accuracy"]) for c in xs]
        f1 = [float(c["pooled_macro_f1"]) for c in xs]
        auc = [float(c["pooled_auc"]) for c in xs]
        s4 = [float(c["s4_accuracy"]) for c in xs]
        s5 = [float(c["s5_accuracy"]) for c in xs]
        last = [float(c["last_accuracy"]) for c in xs if c.get("last_accuracy") is not None]
        be = [int(c["best_epoch"]) for c in xs]
        er = [int(c["epochs_run"]) for c in xs]
        gap = [float(c["overfit_gap"]) for c in xs if c.get("overfit_gap") is not None]
        tr = [float(c["train_acc_at_best"]) for c in xs if c.get("train_acc_at_best") is not None]
        va = [float(c["val_acc_at_best"]) for c in xs if c.get("val_acc_at_best") is not None]
        te = [float(c["test_acc_at_best_curve"]) for c in xs if c.get("test_acc_at_best_curve") is not None]
        m_acc, s_acc = mean_std(acc)
        m_f1, s_f1 = mean_std(f1)
        m_auc, s_auc = mean_std(auc)
        m_s4, s_s4 = mean_std(s4)
        m_s5, s_s5 = mean_std(s5)
        m_last, s_last = mean_std(last)
        row = {
            "model": model,
            "label": MODEL_LABEL[model],
            "n_subjects": len(xs),
            "acc_mean": m_acc, "acc_std": s_acc,
            "f1_mean": m_f1, "f1_std": s_f1,
            "auc_mean": m_auc, "auc_std": s_auc,
            "s4_acc_mean": m_s4, "s4_acc_std": s_s4,
            "s5_acc_mean": m_s5, "s5_acc_std": s_s5,
            "last_acc_mean": m_last, "last_acc_std": s_last,
            "best_epoch_median": float(np.median(be)) if be else float("nan"),
            "best_epoch_mean": float(np.mean(be)) if be else float("nan"),
            "epochs_run_median": float(np.median(er)) if er else float("nan"),
            "n_early_stopped": sum(1 for c in xs if c["early_stopped"]),
            "train_acc_at_best_mean": mean_std(tr)[0],
            "val_acc_at_best_mean": mean_std(va)[0],
            "test_curve_at_best_mean": mean_std(te)[0],
            "overfit_gap_mean": mean_std(gap)[0],
            "pct_above_50": pct(acc, 0.50),
            "pct_above_51.6": pct(acc, PAPER_CS_P05),
            "train_seconds_sum": sum(float(c["train_seconds"]) for c in xs),
        }
        summary_rows.append(row)
        headline[model] = row
    write_csv(out / "exp1_per_model.csv", summary_rows, list(summary_rows[0].keys()))
    return {"per_model": headline, "n_cells": len(cells)}


def exp2_tables(subjects: List[Dict], cells: List[Dict], out: Path) -> Dict[str, Any]:
    write_csv(out / "exp2_per_subject.csv", subjects, [
        "model", "split_seed", "fold", "test_subject", "n_train", "n_val", "n_test",
        "best_epoch", "epochs_run", "early_stopped", "train_seconds",
        "accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece",
        "last_accuracy", "last_macro_f1",
    ])

    # per (model, split_seed): mean over 25 subjects
    split_rows = []
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in subjects:
        grouped[(r["model"], int(r["split_seed"]))].append(r)
    for (model, ss), xs in sorted(grouped.items()):
        acc = [float(x["accuracy"]) for x in xs]
        f1 = [float(x["macro_f1"]) for x in xs]
        auc = [float(x["auc"]) for x in xs]
        m_acc, s_acc = mean_std(acc)
        split_rows.append({
            "model": model, "split_seed": ss, "n_subjects": len(xs),
            "acc_mean": m_acc, "acc_std_over_subjects": s_acc,
            "f1_mean": mean_std(f1)[0], "auc_mean": mean_std(auc)[0],
            "pct_above_50": pct(acc, 0.50),
        })
    write_csv(out / "exp2_per_split.csv", split_rows, list(split_rows[0].keys()))

    # headline: mean ± std over the 5 split means
    summary_rows = []
    headline = {}
    for model in MODELS:
        sm = [r for r in split_rows if r["model"] == model]
        accs = [float(r["acc_mean"]) for r in sm]
        f1s = [float(r["f1_mean"]) for r in sm]
        aucs = [float(r["auc_mean"]) for r in sm]
        all_subj = [float(x["accuracy"]) for x in subjects if x["model"] == model]
        cs = [c for c in cells if c["model"] == model]
        be = [int(c["best_epoch"]) for c in cs]
        er = [int(c["epochs_run"]) for c in cs]
        gap = [float(c["overfit_gap"]) for c in cs if c.get("overfit_gap") is not None]
        tr = [float(c["train_acc_at_best"]) for c in cs if c.get("train_acc_at_best") is not None]
        va = [float(c["val_acc_at_best"]) for c in cs if c.get("val_acc_at_best") is not None]
        te = [float(c["test_acc_at_best_curve"]) for c in cs if c.get("test_acc_at_best_curve") is not None]
        m_acc, s_acc = mean_std(accs)
        row = {
            "model": model,
            "label": MODEL_LABEL[model],
            "n_split_seeds": len(sm),
            "n_subject_scores": len(all_subj),
            "acc_mean_over_splits": m_acc,
            "acc_std_over_splits": s_acc,
            "f1_mean_over_splits": mean_std(f1s)[0],
            "f1_std_over_splits": mean_std(f1s)[1],
            "auc_mean_over_splits": mean_std(aucs)[0],
            "auc_std_over_splits": mean_std(aucs)[1],
            "acc_mean_all_125": mean_std(all_subj)[0],
            "acc_std_all_125": mean_std(all_subj)[1],
            "best_epoch_median": float(np.median(be)) if be else float("nan"),
            "best_epoch_mean": float(np.mean(be)) if be else float("nan"),
            "best_epoch_max": float(np.max(be)) if be else float("nan"),
            "epochs_run_median": float(np.median(er)) if er else float("nan"),
            "n_early_stopped": sum(1 for c in cs if c["early_stopped"]),
            "n_cells": len(cs),
            "train_acc_at_best_mean": mean_std(tr)[0],
            "val_acc_at_best_mean": mean_std(va)[0],
            "test_curve_at_best_mean": mean_std(te)[0],
            "overfit_gap_mean": mean_std(gap)[0],
            "pct_above_50": pct(all_subj, 0.50),
            "train_seconds_sum": sum(float(c["train_seconds"]) for c in cs),
        }
        summary_rows.append(row)
        headline[model] = row
    write_csv(out / "exp2_per_model.csv", summary_rows, list(summary_rows[0].keys()))

    # subject x model: mean over 5 splits
    subj_mean_rows = []
    by_ms: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in subjects:
        by_ms[(r["model"], r["test_subject"])].append(float(r["accuracy"]))
    for (model, subj), xs in sorted(by_ms.items()):
        m, s = mean_std(xs)
        subj_mean_rows.append({
            "model": model, "test_subject": subj, "n_splits": len(xs),
            "acc_mean": m, "acc_std": s,
        })
    write_csv(out / "exp2_subject_mean_over_splits.csv", subj_mean_rows,
              ["model", "test_subject", "n_splits", "acc_mean", "acc_std"])
    return {"per_model": headline, "n_subject_rows": len(subjects), "n_cells": len(cells)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bar(models: List[str], means: List[float], stds: List[float],
             title: str, path: Path, chance_extra: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(models))
    labels = [MODEL_LABEL[m] for m in models]
    ax.bar(x, [100 * v for v in means], yerr=[100 * v for v in stds],
           capsize=3, color="#4C78A8", ecolor="#333333", width=0.7)
    ax.axhline(100 * CHANCE, color="#C44E52", linestyle="--", linewidth=1.1,
               label="chance 50%")
    if chance_extra is not None:
        ax.axhline(100 * chance_extra, color="#E45756", linestyle=":", linewidth=1.1,
                   label=f"paper CS p=0.05 line {100 * chance_extra:.1f}% (N=2500)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(40, max(70, max(100 * (m + s) for m, s in zip(means, stds)) + 3))
    ax.legend(fontsize=8)
    savefig(path)


def plot_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str],
                 title: str, path: Path, vmin=0.35, vmax=0.85) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(col_labels) + 2.2),
                                    max(3.6, 0.38 * len(row_labels) + 1.6)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="accuracy")
    savefig(path)


def plot_grouped_train_val_test(models, train, val, test, title, path):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, [100 * v for v in train], width=w, label="train (eval mode @ best)", color="tab:blue")
    ax.bar(x, [100 * v for v in val], width=w, label="val @ best (selection)", color="tab:orange")
    ax.bar(x + w, [100 * v for v in test], width=w, label="test @ best (monitor)", color="tab:green")
    ax.axhline(50, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    savefig(path)


def plot_best_epoch_box(by_model: Dict[str, List[int]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    data = [by_model[m] for m in MODELS]
    ax.boxplot(data, labels=[MODEL_LABEL[m] for m in MODELS], showfliers=True)
    ax.set_ylabel("best epoch (val accuracy)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(path)


def plot_mean_curves(histories: List[List[Dict]], title: str, path: Path) -> None:
    """Mean ± std of accuracy; at epoch t only cells that reached t are included."""
    if not histories:
        return
    max_n = max(len(h) for h in histories)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for key, label, color in [
        ("train_eval", "train (eval mode)", COLORS["train_eval"]),
        ("val", "validation (selection)", COLORS["val"]),
        ("test", "test (monitor only)", COLORS["test"]),
    ]:
        means, lo, hi, ns = [], [], [], []
        xs = []
        for t in range(max_n):
            vals = []
            for h in histories:
                if t >= len(h):
                    continue
                v = (h[t].get(key) or {}).get("accuracy")
                if v is not None:
                    vals.append(float(v))
            if len(vals) < 3:
                break
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means.append(m)
            lo.append(m - s)
            hi.append(m + s)
            ns.append(len(vals))
            xs.append(t + 1)
        if not means:
            continue
        ax.plot(xs, means, color=color, linewidth=1.7, label=f"{label} (n starts at {ns[0]})")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=8)
    savefig(path)


def plot_cell_curves(history: List[Dict], title: str, path: Path, best_epoch: Optional[int]) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    plotted = False
    for key, label, color in [
        ("train_eval", "train", "tab:blue"),
        ("val", "validation", "tab:orange"),
        ("test", "test (monitor only)", "tab:green"),
    ]:
        ys = series_acc(history, key)
        if all(y is None for y in ys):
            continue
        ax.plot(epochs, ys, label=label, color=color, linewidth=1.3)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    if best_epoch and best_epoch > 0:
        ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1.0,
                   label=f"best.pt (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0.2, 1.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_split_stability(split_rows: List[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    seeds = sorted({int(r["split_seed"]) for r in split_rows})
    x = np.arange(len(MODELS))
    w = 0.14
    cmap = plt.cm.Blues(np.linspace(0.35, 0.9, len(seeds)))
    for i, ss in enumerate(seeds):
        ys = []
        for m in MODELS:
            hit = [r for r in split_rows if r["model"] == m and int(r["split_seed"]) == ss]
            ys.append(100 * hit[0]["acc_mean"] if hit else float("nan"))
        ax.bar(x + (i - 2) * w, ys, width=w, label=f"split seed {ss}", color=cmap[i])
    ax.axhline(50, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("Mean accuracy over 25 subjects (%)")
    ax.set_title("Cross-subject: accuracy of each subject-partition (split seeds 41–45)")
    ax.legend(fontsize=8, ncol=5)
    savefig(path)


def plot_exp1_s4_s5(cells: List[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(MODELS))
    w = 0.35
    s4, s4s, s5, s5s = [], [], [], []
    for m in MODELS:
        a = [float(c["s4_accuracy"]) for c in cells if c["model"] == m]
        b = [float(c["s5_accuracy"]) for c in cells if c["model"] == m]
        s4.append(100 * mean_std(a)[0])
        s4s.append(100 * mean_std(a)[1])
        s5.append(100 * mean_std(b)[0])
        s5s.append(100 * mean_std(b)[1])
    ax.bar(x - w / 2, s4, yerr=s4s, width=w, capsize=3, label="ses-04", color="#4C78A8")
    ax.bar(x + w / 2, s5, yerr=s5s, width=w, capsize=3, label="ses-05", color="#F58518")
    ax.axhline(50, color="#C44E52", linestyle="--", linewidth=1.0, label="chance 50%")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-session: test accuracy on ses-04 vs ses-05 (mean ± std over 25 subjects)")
    ax.legend(fontsize=8)
    savefig(path)


# ---------------------------------------------------------------------------
# Copy + drive
# ---------------------------------------------------------------------------

def copy_sources(dest: Path) -> None:
    exp1 = dest / "实验一_跨会话_cross_session_3to2" / "原始结果"
    exp2 = dest / "实验二_跨被试_cross_subject_5fold" / "原始结果"
    copytree(EXP1_SRC, exp1)
    copytree(EXP2_SRC, exp2)

    cfg = dest / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / "code/configs/experiments/shu_cross_session_3to2.yaml", cfg)
    shutil.copy2(PROJECT / "code/configs/experiments/shu_cross_subject_5fold.yaml", cfg)

    logs = dest / "slurm日志"
    logs.mkdir(parents=True, exist_ok=True)
    for name in [
        "shucs_g1-40278.out", "shucs_g2-40279.out", "shucs_g3-40280.out",
        "shuxs_g1-40281.out", "shuxs_g2-40282.out", "shuxs_g3-40283.out",
        "shucs_atcnet-41883.out", "shuxs_atcnet-41884.out",
    ]:
        src = PROJECT / "logs/slurm" / name
        if src.exists():
            shutil.copy2(src, logs / name)
        err = src.with_suffix(".err")
        if err.exists():
            shutil.copy2(err, logs / err.name)

    note = dest / "未打包_模型权重说明.txt"
    note.write_text(
        "本文件夹没有复制训练权重（best.pt / last.pt）。\n"
        "原因：两套实验合计约 2.4 GB，不适合作为邮件附件。\n\n"
        "权重仍在集群上：\n"
        "  /share/home/yuan/SYX/eeg-mi-online/checkpoints/shu/cross_session_3to2_v1/\n"
        "  /share/home/yuan/SYX/eeg-mi-online/checkpoints/shu/cross_subject_5fold_v1/\n\n"
        "每个 cell 的 result.json 里已经有：\n"
        "  - 最终指标（best.pt / last.pt 在测试集上的 accuracy / F1 / AUC 等）\n"
        "  - 逐 epoch 的 train / val / test 曲线（history）\n"
        "  - 早停信息、best_epoch、训练秒数\n"
        "画图和解读不依赖权重文件。\n",
        encoding="utf-8",
    )


def main() -> None:
    _style()
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"copying sources -> {DEST}")
    copy_sources(DEST)

    tables = DEST / "tables"
    fig1 = DEST / "实验一_跨会话_cross_session_3to2" / "图表"
    fig2 = DEST / "实验二_跨被试_cross_subject_5fold" / "图表"
    tables.mkdir(exist_ok=True)

    print("aggregating exp1")
    cells1 = load_exp1(EXP1_SRC)
    h1 = exp1_tables(cells1, tables)
    print(f"  exp1 cells: {len(cells1)}")

    print("aggregating exp2")
    subj2, cells2 = load_exp2(EXP2_SRC)
    h2 = exp2_tables(subj2, cells2, tables)
    print(f"  exp2 cells: {len(cells2)}  subject-rows: {len(subj2)}")

    # ---- exp1 figures ----
    print("figures exp1")
    means = [h1["per_model"][m]["acc_mean"] for m in MODELS]
    stds = [h1["per_model"][m]["acc_std"] for m in MODELS]
    plot_bar(MODELS, means, stds,
             "Cross-session (3→2): pooled ses-04+05 accuracy  (mean ± std over 25 subjects)",
             fig1 / "总览" / "01_model_bar_pooled_acc.png",
             chance_extra=PAPER_CS_P05)

    # heatmap model x subject
    subjects = sorted({c["subject"] for c in cells1})
    mat = np.full((len(MODELS), len(subjects)), np.nan)
    lookup = {(c["model"], c["subject"]): float(c["pooled_accuracy"]) for c in cells1}
    for i, m in enumerate(MODELS):
        for j, s in enumerate(subjects):
            mat[i, j] = lookup.get((m, s), np.nan)
    plot_heatmap(mat, [MODEL_LABEL[m] for m in MODELS], subjects,
                 "Cross-session: pooled test accuracy per subject",
                 fig1 / "总览" / "02_heatmap_model_x_subject.png")

    plot_exp1_s4_s5(cells1, fig1 / "总览" / "03_ses04_vs_ses05.png")
    plot_best_epoch_box({m: [int(c["best_epoch"]) for c in cells1 if c["model"] == m]
                         for m in MODELS},
                        "Cross-session: epoch of best.pt (val accuracy)",
                        fig1 / "总览" / "04_best_epoch.png")
    plot_grouped_train_val_test(
        MODELS,
        [h1["per_model"][m]["train_acc_at_best_mean"] for m in MODELS],
        [h1["per_model"][m]["val_acc_at_best_mean"] for m in MODELS],
        [h1["per_model"][m]["test_curve_at_best_mean"] for m in MODELS],
        "Cross-session: train / val / test accuracy at best.pt (mean over 25 subjects)",
        fig1 / "总览" / "05_train_val_test_at_best.png",
    )

    for m in MODELS:
        hists = [c["history"] for c in cells1 if c["model"] == m]
        plot_mean_curves(
            hists,
            f"{MODEL_LABEL[m]} — cross-session mean±std accuracy (25 subjects)",
            fig1 / "三曲线_模型均值" / f"curves__{m}__mean.png",
        )
        for c in cells1:
            if c["model"] != m:
                continue
            plot_cell_curves(
                c["history"],
                f"{MODEL_LABEL[m]}  {c['subject']}  pooled test acc={c['pooled_accuracy']:.3f}",
                fig1 / "三曲线_逐被试" / f"curves__{m}__{c['subject']}.png",
                c["best_epoch"],
            )

    # ---- exp2 figures ----
    print("figures exp2")
    means = [h2["per_model"][m]["acc_mean_over_splits"] for m in MODELS]
    stds = [h2["per_model"][m]["acc_std_over_splits"] for m in MODELS]
    plot_bar(MODELS, means, stds,
             "Cross-subject 5-fold: accuracy  (mean ± std over 5 subject-partitions)",
             fig2 / "总览" / "01_model_bar_acc.png")

    subjects = sorted({r["test_subject"] for r in subj2})
    # mean over 5 splits
    by = defaultdict(list)
    for r in subj2:
        by[(r["model"], r["test_subject"])].append(float(r["accuracy"]))
    mat = np.full((len(MODELS), len(subjects)), np.nan)
    for i, m in enumerate(MODELS):
        for j, s in enumerate(subjects):
            xs = by.get((m, s), [])
            mat[i, j] = float(np.mean(xs)) if xs else np.nan
    plot_heatmap(mat, [MODEL_LABEL[m] for m in MODELS], subjects,
                 "Cross-subject: accuracy per subject (mean over 5 split seeds)",
                 fig2 / "总览" / "02_heatmap_model_x_subject.png")

    split_rows = []
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in subj2:
        grouped[(r["model"], int(r["split_seed"]))].append(float(r["accuracy"]))
    for (model, ss), xs in grouped.items():
        split_rows.append({"model": model, "split_seed": ss, "acc_mean": float(np.mean(xs))})
    plot_split_stability(split_rows, fig2 / "总览" / "03_split_seed_stability.png")
    plot_best_epoch_box({m: [int(c["best_epoch"]) for c in cells2 if c["model"] == m]
                         for m in MODELS},
                        "Cross-subject: epoch of best.pt (val accuracy); patience=25",
                        fig2 / "总览" / "04_best_epoch.png")
    plot_grouped_train_val_test(
        MODELS,
        [h2["per_model"][m]["train_acc_at_best_mean"] for m in MODELS],
        [h2["per_model"][m]["val_acc_at_best_mean"] for m in MODELS],
        [h2["per_model"][m]["test_curve_at_best_mean"] for m in MODELS],
        "Cross-subject: train / val / test accuracy at best.pt (mean over 25 cells)",
        fig2 / "总览" / "05_train_val_test_at_best.png",
    )

    for m in MODELS:
        hists = [c["history"] for c in cells2 if c["model"] == m]
        plot_mean_curves(
            hists,
            f"{MODEL_LABEL[m]} — cross-subject mean±std accuracy (25 cells: 5 splits × 5 folds)",
            fig2 / "三曲线_模型均值" / f"curves__{m}__mean.png",
        )
        for c in cells2:
            if c["model"] != m:
                continue
            plot_cell_curves(
                c["history"],
                f"{MODEL_LABEL[m]}  split{c['split_seed']} fold{c['fold']}  "
                f"val@best={c['best_val']:.3f}",
                fig2 / "三曲线_逐cell" / f"curves__{c['cell']}.png",
                c["best_epoch"],
            )

    headline = {
        "exp1_n_cells": len(cells1),
        "exp2_n_cells": len(cells2),
        "exp2_n_subject_rows": len(subj2),
        "exp1": {m: {k: v for k, v in h1["per_model"][m].items() if k != "history"}
                 for m in MODELS},
        "exp2": {m: h2["per_model"][m] for m in MODELS},
    }
    with open(tables / "headline.json", "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2, ensure_ascii=False)
    print("wrote", tables / "headline.json")
    print("done ->", DEST)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Package SHU 2022 exp2 DualCD add-on: tables, figures, interpretation.

Writes to /share/home/yuan/SYX/SHU2022_exp2_DualCD结果_20260825/
Does not copy checkpoints (best.pt / last.pt).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import package_shu2022_handoff as P  # noqa: E402

PROJECT = Path("/share/home/yuan/SYX/eeg-mi-online")
DEST = Path("/share/home/yuan/SYX/SHU2022_exp2_DualCD结果_20260825")
EXP2_SRC = PROJECT / "outputs/experiments/shu/cross_subject_5fold_v1"

# WBCIC 3C 821 (different protocol; quoted only as ranking context)
REF_821 = {
    "eegnex": (0.6991, 0.1293),
    "eeg_deformer": (0.6849, 0.1414),
    "atcnet": (0.6839, 0.1265),
    "s4erp": (0.6826, 0.1370),
    "eegnet_official": (0.6636, 0.1245),
    "dualcd_transformer": (0.6513, 0.1251),
    "dualcd_s4_flatten": (0.6454, 0.1293),
}


def copy_sources(dest: Path) -> None:
    raw = dest / "原始结果"
    if raw.exists():
        shutil.rmtree(raw)
    shutil.copytree(EXP2_SRC, raw)

    cfg = dest / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT / "code/configs/experiments/shu_cross_subject_5fold.yaml", cfg
    )

    logs = dest / "slurm日志"
    logs.mkdir(parents=True, exist_ok=True)
    for name in [
        "shuxs_dino_smoke-42683.out",
        "shuxs_dino_flat-42684.out",
        "shuxs_dino_tf-42685.out",
        "shuxs_dino_tf-42685.err",
        "shuxs_dino_tf-44049.out",
    ]:
        src = PROJECT / "logs/slurm" / name
        if src.exists():
            shutil.copy2(src, logs / name)

    (dest / "未打包_模型权重说明.txt").write_text(
        "本文件夹没有复制训练权重（best.pt / last.pt）。\n"
        "DualCD 单个 cell 的权重约 250–500 MB，50 个 cell 合计数十 GB，不适合打包。\n\n"
        "权重仍在集群上：\n"
        "  /share/home/yuan/SYX/eeg-mi-online/checkpoints/shu/cross_subject_5fold_v1/\n\n"
        "每个 cell 的 result.json 里已经有最终指标、逐 epoch 的 train/val/test 曲线、\n"
        "早停信息、best_epoch、训练秒数，以及 DualCD 的分项 loss（dino/ibot/base 等）。\n"
        "画图和解读不依赖权重文件。\n",
        encoding="utf-8",
    )


def plot_rank_bar(headline: dict, path: Path) -> None:
    models = list(P.MODELS)
    means = [headline[m]["acc_mean_over_splits"] for m in models]
    stds = [headline[m]["acc_std_over_splits"] for m in models]
    colors = []
    for m in models:
        if m.startswith("dualcd"):
            colors.append("#E45756")
        elif m == "eegnex":
            colors.append("#4C78A8")
        else:
            colors.append("#9ECAE1")
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(models))
    ax.bar(
        x,
        [100 * v for v in means],
        yerr=[100 * v for v in stds],
        capsize=3,
        color=colors,
        ecolor="#333333",
        width=0.72,
    )
    ax.axhline(50, color="#C44E52", linestyle="--", linewidth=1.1, label="chance 50%")
    ax.set_xticks(x)
    ax.set_xticklabels([P.MODEL_LABEL[m] for m in models], rotation=22, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "SHU 2022 cross-subject: mean accuracy ± std over 5 subject-partitions"
    )
    ax.set_ylim(48, 70)
    ax.legend(fontsize=8)
    P.savefig(path)


def plot_flatten_vs_s4erp(subj: list, path: Path) -> None:
    by = defaultdict(dict)
    for r in subj:
        if r["model"] in ("s4erp", "dualcd_s4_flatten"):
            by[r["test_subject"]].setdefault(r["model"], []).append(float(r["accuracy"]))
    xs, ys, labels = [], [], []
    for sub, d in sorted(by.items()):
        if "s4erp" not in d or "dualcd_s4_flatten" not in d:
            continue
        xs.append(100.0 * float(np.mean(d["s4erp"])))
        ys.append(100.0 * float(np.mean(d["dualcd_s4_flatten"])))
        labels.append(sub)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(xs, ys, c="#4C78A8", s=36, zorder=3)
    lo, hi = 40, 95
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("S4ERP accuracy (mean over 5 splits, %)")
    ax.set_ylabel("DualCD-S4-Flatten accuracy (mean over 5 splits, %)")
    ax.set_title("Same S4 backbone: supervised S4ERP vs S4+DINO (Flatten)")
    ax.set_aspect("equal")
    P.savefig(path)


def plot_best_epoch_hist(cells: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bins = np.arange(0, 70, 2)
    for m, color, alpha in [
        ("dualcd_s4_flatten", "#E45756", 0.55),
        ("dualcd_transformer", "#F58518", 0.45),
        ("s4erp", "#4C78A8", 0.35),
        ("eegnex", "#54A24B", 0.30),
    ]:
        be = [int(c["best_epoch"]) for c in cells if c["model"] == m]
        ax.hist(be, bins=bins, color=color, alpha=alpha, label=P.MODEL_LABEL[m])
    ax.set_xlabel("best epoch (val accuracy)")
    ax.set_ylabel("number of cells (max 25 / model)")
    ax.set_title("When val accuracy peaked (patience=25)")
    ax.legend(fontsize=8)
    P.savefig(path)


def plot_dualcd_loss_parts(cells: list, model: str, path: Path) -> None:
    hists = [c["history"] for c in cells if c["model"] == model and c.get("history")]
    if not hists:
        return
    keys = [
        ("base", "supervised CE (base)", "tab:blue"),
        ("dino", "DINO", "tab:red"),
        ("proto", "proto", "tab:purple"),
        ("perturb", "perturb", "tab:green"),
        ("ibot", "iBOT", "tab:orange"),
        ("dkoleo", "KoLeo", "tab:brown"),
    ]
    max_n = max(len(h) for h in hists)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for key, label, color in keys:
        xs, means = [], []
        for t in range(max_n):
            vals = []
            for h in hists:
                if t >= len(h):
                    continue
                parts = h[t].get("loss_parts") or {}
                if key in parts and parts[key] is not None:
                    vals.append(float(parts[key]))
            if len(vals) < 3:
                break
            xs.append(t + 1)
            means.append(float(np.mean(vals)))
        if means:
            ax.plot(xs, means, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term (mean over cells still running)")
    ax.set_title(f"{P.MODEL_LABEL[model]} — DualCD loss parts")
    ax.legend(fontsize=8, ncol=2)
    P.savefig(path)


def fmt_pct(mean: float, std: float) -> str:
    return f"{100 * mean:.1f}±{100 * std:.1f}"


def write_readme(dest: Path, h2: dict, n_subj: int, n_cells: int) -> None:
    pm = h2["per_model"]
    ranked = sorted(P.MODELS, key=lambda m: -pm[m]["acc_mean_over_splits"])

    def row(m: str) -> str:
        d = pm[m]
        mark = " ← DualCD" if m.startswith("dualcd") else ""
        return (
            f"| `{m}` | {P.MODEL_LABEL[m]} | "
            f"**{fmt_pct(d['acc_mean_over_splits'], d['acc_std_over_splits'])}** | "
            f"{fmt_pct(d['f1_mean_over_splits'], d['f1_std_over_splits'])} | "
            f"{fmt_pct(d['auc_mean_over_splits'], d['auc_std_over_splits'])} | "
            f"{d['best_epoch_median']:.0f} | {d['epochs_run_median']:.0f} | "
            f"{100 * d['train_acc_at_best_mean']:.1f} / "
            f"{100 * d['val_acc_at_best_mean']:.1f} / "
            f"{100 * d['test_curve_at_best_mean']:.1f} | "
            f"{d['overfit_gap_mean'] * 100:.1f} | "
            f"{d['train_seconds_sum'] / 3600:.1f} |{mark}"
        )

    lines = [
        "# SHU 2022 实验二 DualCD 加训结果（2026-08-25）",
        "",
        "跨被试 5-fold（`shu_cross_subject_5fold_v1`）上，在已完成的 7 个监督模型之外，",
        "按**同一套配方**补训了 DualCD-S4-Flatten 和 DualCD-Transformer。",
        "两个 DualCD 模型各 25/25 格子，全部 `ok`。",
        "",
        "## 一句话结论",
        "",
        "**DualCD 没有超过同表上的监督模型。** Flatten **60.4±0.5%**，Transformer **60.7±1.2%**，",
        "都低于 EEGNeX（**64.5±0.4%**）和监督 S4ERP（**61.9±0.9%**）。",
        "S4 接上 DINO（Flatten）比不接 DINO 的 S4ERP **低约 1.5 个点**。",
        "这和 WBCIC 3C 821 上的排序一致：DINO 没有赢过 plain S4ERP / EEGNeX。",
        "**不能据此写 ICASSP 的 SOTA 故事**；最多作为负结果 / 消融。",
        "",
        "## 协议（读数字前先看这个）",
        "",
        "| 项 | 设定 |",
        "|---|---|",
        "| 数据 | SHU 2022（Ma et al. Sci Data 9:531），25 人 × 5 session，32ch × 1000 @250 Hz，2 类 |",
        "| 划分 | 每个 split：16 训练被试 / 4 验证被试 / 5 测试被试；每人 5 个 session 全用 |",
        "| 重复 | `split_seeds` 41–45（换的是**被试划分**，不是训练初始化） |",
        "| 训练种子 | `[0]`，固定 |",
        "| 优化 | Adam 1e-4，有效 batch 128，最多 500 epoch，无 scheduler / wd / clip |",
        "| 早停 | monitor **accuracy**，patience **25** |",
        "| 归一化 | `per_sample_zscore` |",
        "| DualCD 微批 | Flatten 32×4、Transformer 16×8（有效 batch 仍 128） |",
        "| 选模 | 只看 val；test 曲线只监控，从不参与 `best.pt` |",
        "",
        "Headline：**先对 25 个测试被试取均值，再对 5 个 split seed 取 mean±std**。",
        "被试之间的标准差大约 10 个点，那是人差，不是划分差；不要把 ±10 写进总表。",
        "",
        "**不要和下面这些数字直接比：**",
        "",
        "- 论文 cross-session 53.7%：那是只训 ses-01、测后面 session，不是这套 16:4:5。",
        "- 实验一 3→2（训 01+02+03、测 04+05）：也不是这套划分。",
        "- WBCIC 3C 821：11 人 LOSO、3 类、patience 100、monitor macro-F1。排序可参考，数字不可比。",
        "",
        "## 总表（实验二，9 模型）",
        "",
        "Acc / F1 / AUC = 5 个被试划分上的 mean±std（百分数）。",
        "train/val/test 是 `best.pt` 对应 epoch 上、对 25 个 cell 再平均。",
        "overfit gap = train_eval − val。GPU-h 是该模型 25 个 cell 的墙钟合计。",
        "",
        "| id | 显示名 | Acc (%) | macro-F1 (%) | AUC (%) | best epoch 中位 | 跑到的 epoch 中位 | train / val / test @ best | overfit (pp) | GPU-h |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for m in ranked:
        lines.append(row(m))

    d_flat = pm["dualcd_s4_flatten"]
    d_tf = pm["dualcd_transformer"]
    d_s4 = pm["s4erp"]
    d_nx = pm["eegnex"]

    lines += [
        "",
        "### DualCD 相对同表最优 / 同骨干",
        "",
        f"- vs EEGNeX：Flatten {100 * (d_flat['acc_mean_over_splits'] - d_nx['acc_mean_over_splits']):+.1f} pp，"
        f"Transformer {100 * (d_tf['acc_mean_over_splits'] - d_nx['acc_mean_over_splits']):+.1f} pp。",
        f"- vs 监督 S4ERP：Flatten {100 * (d_flat['acc_mean_over_splits'] - d_s4['acc_mean_over_splits']):+.1f} pp"
        "（同一套 S4 骨干 + DINO，没有涨点）。",
        f"- Flatten vs Transformer：{100 * (d_flat['acc_mean_over_splits'] - d_tf['acc_mean_over_splits']):+.1f} pp，"
        "两者差不多，Transformer 的划分波动稍大（split 43 到了 62.7%，其余约 60%）。",
        "",
        "## 怎么读这些曲线",
        "",
        "1. **Val 很早就到顶。** DualCD 的 `best.pt` 中位 epoch 是 9；大约一半格子在 epoch 1–5 就选中了。"
        "patience 25 之后在 ~30–35 epoch 停。监督 EEGNeX 中位 best epoch 是 55，还能再涨。",
        "2. **过拟合明显。** DualCD 在 best epoch 上 train ≈ 85%、val ≈ 62%、test ≈ 60–61%。"
        "监督 S4ERP 同样过拟合（train 92%），但 test 仍略高于 DualCD。",
        "3. **DINO loss 在降，分类不准。** 分项 loss 图里 DINO 项迅速下降，supervised CE（base）也在降，"
        "但 val/test accuracy 不跟着涨。说明自监督项在拟合视图，没有变成更好的跨被试特征。",
        "4. **选模有效，但救不了。** DualCD 的 last.pt 测试准确率比 best.pt 再低约 1.5 pp；"
        "早停没有选错方向，只是最好也就这样。",
        "5. **人差远大于模型差。** 好被试（sub-006 / sub-020）Flatten 也能到 80%+；"
        "差被试（sub-025 / sub-010）在机会水平附近。换 DualCD 改变不了这个结构。",
        "",
        "## 和 821 的对照（只比排序，不比绝对值）",
        "",
        "WBCIC 3C `paper_baseline_3c_821_v1`（11 折 LOSO，3 类，patience 100）：",
        "",
        "| 模型 | 821 Acc（折间 std） | 本次 SHU 跨被试 Acc（划分间 std） |",
        "|---|---:|---:|",
    ]
    for m in [
        "eegnex",
        "s4erp",
        "eegnet_official",
        "dualcd_transformer",
        "dualcd_s4_flatten",
    ]:
        a, s = REF_821[m]
        d = pm[m]
        lines.append(
            f"| {P.MODEL_LABEL[m]} | {100 * a:.1f}±{100 * s:.1f} | "
            f"{fmt_pct(d['acc_mean_over_splits'], d['acc_std_over_splits'])} |"
        )
    lines += [
        "",
        "两次都是：EEGNeX > 监督 S4ERP > DualCD。DualCD 的 best epoch 两次都非常早（821 上 mean 7.7 / 10.1）。",
        "所以这次不是「SHU 数据特殊」；更像 DualCD 这套损失在当前配方下系统性地早停、过拟合、涨不动。",
        "",
        "## 训练过程（事实，不是结论）",
        "",
        "- Smoke（2 epoch，独立目录）2026-08-21 23:57 通过，未写入正式 v1。",
        "- Flatten 作业 42684：08-21 23:57 → 08-23 18:41，25/25，正常结束。",
        "- Transformer 作业 42685：跑满 48h 超时；续跑 44049 于 08-24 11:48 接上，20:56 正常结束，25/25。",
        "- 正式数字只在 `cross_subject_5fold_v1`，不要用 smoke 的 2-epoch 数。",
        "",
        "## 文件夹里有什么",
        "",
        "```",
        "00_先读这个.md          本说明（结论 + 协议 + 读图）",
        "tables/                 CSV / headline.json / 总表.md",
        "图表/总览/              条形图、热力图、划分稳定性、过拟合、best epoch",
        "图表/三曲线_模型均值/   9 个模型的 train/val/test 均值±标准差",
        "图表/三曲线_DualCD逐cell/  DualCD 50 个格子的三条曲线",
        "图表/acc_loss_mean/     准确率 + NLL（含 DualCD）",
        "图表/DualCD_loss分项/   DINO / CE / proto 等",
        "原始结果/               225 个 result.json + splits + 配置快照（无权重）",
        "configs/                实验 yaml",
        "slurm日志/",
        "```",
        "",
        f"格子数：实验二 {n_cells}（9×25），被试分数行 {n_subj}（9×25×5）。失败 0。",
        "",
        "## 图表索引",
        "",
        "- `图表/总览/01_model_bar_acc.png` — 9 模型 Acc，红柱是 DualCD",
        "- `图表/总览/02_heatmap_model_x_subject.png` — 模型 × 被试（5 个 split 平均）",
        "- `图表/总览/03_split_seed_stability.png` — 五个划分是否稳",
        "- `图表/总览/04_best_epoch.png` — 选中 epoch 的箱线",
        "- `图表/总览/05_train_val_test_at_best.png` — 过拟合三联柱",
        "- `图表/总览/06_flatten_vs_s4erp_per_subject.png` — 同一人：S4ERP vs Flatten",
        "- `图表/总览/07_best_epoch_hist.png` — DualCD 的 best epoch 有多早",
        "",
        "## 若要接着做（不是这次的结果）",
        "",
        "这次已经把「同一配方、直接接 DINO」测完了。若还要救 DualCD，需要改配方",
        "（例如关掉部分 DINO 项、拉长 warmup、换 monitor、或先预训练再线性探测），",
        "那就是新实验，不能和这张表混报。",
        "",
    ]
    text = "\n".join(lines)
    (dest / "00_先读这个.md").write_text(text, encoding="utf-8")
    (dest / "README.md").write_text(text, encoding="utf-8")
    (dest / "tables" / "总表.md").write_text(
        "\n".join(
            [
                "# 实验二 headline",
                "",
                "| id | 显示名 | Acc (%) | macro-F1 (%) | AUC (%) | best epoch 中位 | GPU-h |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
            + [
                f"| `{m}` | {P.MODEL_LABEL[m]} | "
                f"{fmt_pct(pm[m]['acc_mean_over_splits'], pm[m]['acc_std_over_splits'])} | "
                f"{fmt_pct(pm[m]['f1_mean_over_splits'], pm[m]['f1_std_over_splits'])} | "
                f"{fmt_pct(pm[m]['auc_mean_over_splits'], pm[m]['auc_std_over_splits'])} | "
                f"{pm[m]['best_epoch_median']:.0f} | {pm[m]['train_seconds_sum'] / 3600:.1f} |"
                for m in ranked
            ]
            + [""]
        ),
        encoding="utf-8",
    )


def write_split_csv_from_rows(split_rows: list, path: Path) -> None:
    if not split_rows:
        return
    fields = list(split_rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(split_rows)


def main() -> None:
    P._style()
    DEST.mkdir(parents=True, exist_ok=True)
    print("copying result.json tree ->", DEST)
    copy_sources(DEST)

    tables = DEST / "tables"
    fig = DEST / "图表"
    tables.mkdir(exist_ok=True)

    print("aggregating exp2")
    subj2, cells2 = P.load_exp2(EXP2_SRC)
    h2 = P.exp2_tables(subj2, cells2, tables)
    print(f"  cells={len(cells2)}  subject-rows={len(subj2)}")

    print("overview figures")
    plot_rank_bar(h2["per_model"], fig / "总览" / "01_model_bar_acc.png")

    subjects = sorted({r["test_subject"] for r in subj2})
    by = defaultdict(list)
    for r in subj2:
        by[(r["model"], r["test_subject"])].append(float(r["accuracy"]))
    mat = np.full((len(P.MODELS), len(subjects)), np.nan)
    for i, m in enumerate(P.MODELS):
        for j, s in enumerate(subjects):
            xs = by.get((m, s), [])
            mat[i, j] = float(np.mean(xs)) if xs else np.nan
    P.plot_heatmap(
        mat,
        [P.MODEL_LABEL[m] for m in P.MODELS],
        subjects,
        "Cross-subject: accuracy per subject (mean over 5 split seeds)",
        fig / "总览" / "02_heatmap_model_x_subject.png",
    )

    split_rows = []
    grouped = defaultdict(list)
    for r in subj2:
        grouped[(r["model"], int(r["split_seed"]))].append(float(r["accuracy"]))
    for (model, ss), xs in grouped.items():
        split_rows.append({"model": model, "split_seed": ss, "acc_mean": float(np.mean(xs))})
    P.plot_split_stability(split_rows, fig / "总览" / "03_split_seed_stability.png")
    P.plot_best_epoch_box(
        {m: [int(c["best_epoch"]) for c in cells2 if c["model"] == m] for m in P.MODELS},
        "Cross-subject: epoch of best.pt (val accuracy); patience=25",
        fig / "总览" / "04_best_epoch.png",
    )
    P.plot_grouped_train_val_test(
        P.MODELS,
        [h2["per_model"][m]["train_acc_at_best_mean"] for m in P.MODELS],
        [h2["per_model"][m]["val_acc_at_best_mean"] for m in P.MODELS],
        [h2["per_model"][m]["test_curve_at_best_mean"] for m in P.MODELS],
        "Cross-subject: train / val / test accuracy at best.pt (mean over 25 cells)",
        fig / "总览" / "05_train_val_test_at_best.png",
    )
    plot_flatten_vs_s4erp(subj2, fig / "总览" / "06_flatten_vs_s4erp_per_subject.png")
    plot_best_epoch_hist(cells2, fig / "总览" / "07_best_epoch_hist.png")

    print("mean curves (9 models)")
    for m in P.MODELS:
        hists = [c["history"] for c in cells2 if c["model"] == m]
        P.plot_mean_curves(
            hists,
            f"{P.MODEL_LABEL[m]} — mean±std accuracy (25 cells: 5 splits × 5 folds)",
            fig / "三曲线_模型均值" / f"curves__{m}__mean.png",
        )

    print("DualCD per-cell curves")
    for c in cells2:
        if c["model"] not in ("dualcd_s4_flatten", "dualcd_transformer"):
            continue
        P.plot_cell_curves(
            c["history"],
            f"{P.MODEL_LABEL[c['model']]}  split{c['split_seed']} fold{c['fold']}  "
            f"val@best={c['best_val']:.3f}",
            fig / "三曲线_DualCD逐cell" / f"curves__{c['cell']}.png",
            c["best_epoch"],
        )

    print("acc / loss mean")
    sys.path.insert(0, str(SCRIPTS))
    import plot_training_diagnostics as D  # noqa: E402

    D._style()
    by_model = D.load_histories(EXP2_SRC)
    acc_dir = fig / "acc_loss_mean"
    for model in P.MODELS:
        D.plot_acc_loss("Cross-subject 5-fold", model, by_model.get(model) or [], acc_dir)

    print("DualCD loss parts")
    for m in ("dualcd_s4_flatten", "dualcd_transformer"):
        plot_dualcd_loss_parts(
            cells2, m, fig / "DualCD_loss分项" / f"loss_parts__{m}.png"
        )

    headline = {
        "exp2_n_cells": len(cells2),
        "exp2_n_subject_rows": len(subj2),
        "protocol": "kfold_subject 16:4:5, split_seeds 41-45, train seed 0",
        "headline_rule": "mean over 25 test subjects, then mean±std over 5 partitions",
        "exp2": {m: h2["per_model"][m] for m in P.MODELS},
    }
    with open(tables / "headline.json", "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2, ensure_ascii=False)

    write_readme(DEST, h2, len(subj2), len(cells2))
    print("done ->", DEST)


if __name__ == "__main__":
    main()

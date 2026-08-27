#!/usr/bin/env python
"""Aggregate cross-subject run(s) into the comparison tables, statistics and figures.

Replaces the earlier single-seed version, which hard-coded ``--seed 0``, globbed
``cells/*__seed{seed}/result.json`` (so it could not see multi-split-seed cell names) and
reported only ``mean +/- std``. The statistics live in
``code/summaries/cross_subject_bench.py``; this file is the CLI.

Outputs, per run directory::

    tables/per_cell.csv        one row per (model, split_seed, fold, seed, test subject)
    tables/per_model.csv       mean +/- std over subjects, per-split spread, params, GPU-h
    tables/per_subject.csv     subjects x models matrix of the primary metric
    tables/coverage.csv        cells/subjects/split seeds present per model (partial-run guard)
    tables/paired_tests.csv    pre-registered comparisons: Wilcoxon + Holm + Cohen's d_z
    tables/variant_selection.csv  which variant represents its family, chosen on VALIDATION
    tables/friedman_nemenyi.json  omnibus test, average ranks, critical difference
    figures/cd_diagram.png     Demsar critical-difference diagram
    REPORT_TABLE.md            everything above as Markdown, ready to paste

And once with ``--run`` given more than once::

    <first run>/tables/main_table.csv   cross-dataset table, one column block per dataset

Usage
-----
    python scripts/summarize_cross_subject.py \
        --run outputs/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1

    python scripts/summarize_cross_subject.py \
        --run outputs/experiments/wbci_shu/wbci_2c_cross_subject_bench_v1 \
        --run outputs/experiments/shu/shu_cross_subject_bench_v1 \
        --out-dir 5_papers/tables

The DSGNet Table II reference numbers are NOT printed here any more: they are three-class
SHUv5 LOSO values and this benchmark is two-class 5-fold on disjoint cohorts, so putting the
two in one table invites exactly the apples-to-oranges comparison
``docs/RESULTS_TRUTH.md`` §8 had to correct. Quote them in prose instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from code.summaries.cross_subject_bench import (  # noqa: E402
    METRICS,
    PRIMARY_METRIC,
    cd_diagram,
    coverage,
    friedman_nemenyi,
    load_run,
    main_table,
    paired_tests,
    per_model_table,
    select_by_validation,
    subject_matrix,
)


def _pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * float(x):.2f}"


class _Counter:
    """Sequential section numbers for a report whose middle sections are conditional."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def summarize_one(run_dir: Path, metric: str) -> Dict[str, object]:
    df = load_run(run_dir)
    tables = run_dir / "tables"
    figures = run_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables / "per_cell.csv", index=False)
    cov = coverage(df)
    cov.to_csv(tables / "coverage.csv")

    per_model = per_model_table(df)
    per_model.to_csv(tables / "per_model.csv")

    mat = subject_matrix(df, metric=metric)
    mat.to_csv(tables / "per_subject.csv")

    paired = paired_tests(mat)
    if not paired.empty:
        paired.to_csv(tables / "paired_tests.csv", index=False)

    sel = select_by_validation(df)
    if not sel.empty:
        sel.to_csv(tables / "variant_selection.csv", index=False)

    fn = friedman_nemenyi(mat)
    fn_json = {k: (v.to_dict() if isinstance(v, pd.Series) else v) for k, v in fn.items()}
    (tables / "friedman_nemenyi.json").write_text(
        json.dumps(fn_json, indent=2, ensure_ascii=False), encoding="utf-8")
    if "avg_ranks" in fn:
        cd_diagram(fn["avg_ranks"], fn["critical_difference"],
                   figures / "cd_diagram.png",
                   title=f"{run_dir.name} — {metric}, N={fn['N']} subjects, "
                         f"k={fn['k']} models")

    _write_report(run_dir, df, per_model, cov, paired, sel, fn, mat, metric)
    return {"df": df, "per_model": per_model, "mat": mat}


def _write_report(run_dir: Path, df: pd.DataFrame, per_model: pd.DataFrame,
                  cov: pd.DataFrame, paired: pd.DataFrame, sel: pd.DataFrame,
                  fn: Dict[str, object], mat: pd.DataFrame, metric: str) -> None:
    n_subj = int(mat.shape[0])
    # Sections 2-4 are conditional (a run without variant families or fusion models has
    # nothing to put there), so the number is assigned as they are emitted rather than
    # hard-coded — a report that jumps from "1." to "4." reads like a bug.
    section = _Counter()
    md: List[str] = [
        f"# {run_dir.name} — cross-subject comparison", "",
        f"* rows: **{len(df)}** over **{df['cell'].nunique()}** cells",
        f"* subjects: **{n_subj}** | split seeds: "
        f"**{sorted(set(df['split_seed'].dropna().tolist()))}** | "
        f"train seeds: **{sorted(set(df['seed'].dropna().tolist()))}**",
        f"* primary metric: **{metric}** (values below are percentages)",
        "",
        f"## {section.next()}. Per model — mean ± std over subjects",
        "",
        "| model | Acc | macro-F1 | AUROC | bal-Acc | Kappa | ±split | params | best ep | GPU-h |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in per_model.iterrows():
        kappa = r.get("cohen_kappa")
        kappa_s = "—" if pd.isna(kappa) else f"{float(kappa):.4f}"
        md.append(
            f"| `{m}` | **{_pct(r.get('accuracy'))} ± {_pct(r.get('accuracy_std'))}** "
            f"| {_pct(r.get('macro_f1'))} | {_pct(r.get('auc'))} "
            f"| {_pct(r.get('balanced_accuracy'))} | {kappa_s} "
            f"| {_pct(r.get('std_splits'))} | {int(r.get('n_params', 0)):,} "
            f"| {r.get('best_epoch_median'):.0f} | {r.get('gpu_hours'):.1f} |"
        )

    incomplete = cov[cov["n_subjects"] < n_subj]
    if len(incomplete):
        md += ["", "> ⚠ **Incomplete models — do not read these rows beside the complete ones.**",
               "", "| model | subjects | cells | split seeds |", "|:---|---:|---:|:---|"]
        for m, r in incomplete.iterrows():
            md.append(f"| `{m}` | {r['n_subjects']}/{n_subj} | {r['n_cells']} | {r['split_seeds']} |")

    if not sel.empty:
        md += ["", f"## {section.next()}. Variant selection (on VALIDATION, never on test)", "",
               "| family | model | val | test | val rank | test rank | selected |",
               "|:---|:---|---:|---:|---:|---:|:---:|"]
        for _, r in sel.iterrows():
            md.append(f"| {r['family']} | `{r['model']}` | {_pct(r['val_score'])} "
                      f"| {_pct(r['test_score'])} | {r['val_rank']} | {r['test_rank']} "
                      f"| {'**yes**' if r['selected'] else ''} |")
        if (sel["val_rank"] != sel["test_rank"]).any():
            md.append("")
            md.append("> Validation and test rankings disagree — which is precisely why the "
                      "representative is chosen on validation.")

    if not paired.empty:
        md += ["", f"## {section.next()}. Pre-registered paired comparisons (Wilcoxon, Holm-corrected, n={n_subj})",
               "",
               "| question | baseline → candidate | Δ pp | median Δ | d_z | W/T/L | p | p (Holm) | sig |",
               "|:---|:---|---:|---:|---:|:---:|---:|---:|:---:|"]
        for _, r in paired.iterrows():
            md.append(
                f"| {r['question']} | `{r['baseline']}` → `{r['candidate']}` "
                f"| **{100 * r['delta']:+.2f}** | {100 * r['median_delta']:+.2f} "
                f"| {r['d_z']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']} "
                f"| {r['p_value']:.4f} | {r['p_holm']:.4f} "
                f"| {'**yes**' if r['significant_holm_05'] else 'no'} |"
            )
        md += ["",
               "> Δ is candidate − baseline in percentage points. `d_z` is Cohen's d for "
               "paired data (mean difference / SD of differences); |d_z| < 0.2 is a small "
               "effect regardless of the p-value. W/T/L counts subjects where the candidate "
               "won / tied / lost."]

    if "avg_ranks" in fn:
        md += ["", f"## {section.next()}. Omnibus ranking (Friedman + Nemenyi)", "",
               f"* Friedman χ² = **{fn['friedman_stat']:.2f}**, p = **{fn['friedman_p']:.2e}** "
               f"(k={fn['k']} models, N={fn['N']} subjects)",
               f"* Nemenyi critical difference at α={fn['alpha']}: **{fn['critical_difference']:.2f}** ranks",
               "", "| rank | model | avg rank |", "|---:|:---|---:|"]
        for i, (m, r) in enumerate(fn["avg_ranks"].items(), start=1):
            md.append(f"| {i} | `{m}` | {r:.2f} |")
        md += ["", "Two models are indistinguishable at this α if their average ranks differ "
               f"by less than {fn['critical_difference']:.2f}; see `figures/cd_diagram.png`.",
               ""]
    elif "error" in fn:
        md += ["", f"## {section.next()}. Omnibus ranking — skipped: {fn['error']}", ""]

    md += ["", f"## {section.next()}. Protocol", "",
           "Split: subject-level k-fold, 6:2:2 train/val/test subjects, strict zero-shot "
           "(a test subject appears in neither train nor val). Seeds 41,42 re-draw the "
           "subject partition (ERP-Benchmark, arXiv:2601.00573); the training seed is fixed "
           "at 0, so the reported spread is the spread of the partition. Recipe: Adam 1e-4, "
           "batch 128, no scheduler / weight decay / gradient clipping, early stopping on "
           "validation macro-F1 with patience 25. Model selection never sees the test "
           "subjects; the per-epoch test curve is monitoring only.", ""]

    (run_dir / "REPORT_TABLE.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="run directory (repeat for a cross-dataset main table)")
    ap.add_argument("--metric", default=PRIMARY_METRIC, choices=METRICS,
                    help=f"primary metric for ranking and the CD diagram (default {PRIMARY_METRIC})")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the cross-dataset main table (default: first run)")
    args = ap.parse_args()

    results: Dict[str, pd.DataFrame] = {}
    for run in args.run:
        run_dir = Path(run)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        print(f"=== {run_dir} ===")
        res = summarize_one(run_dir, args.metric)
        results[run_dir.name] = res["df"]
        pm = res["per_model"]
        for m, r in pm.iterrows():
            print(f"  {m:22s} {args.metric} {_pct(r.get(args.metric))} "
                  f"± {_pct(r.get(args.metric + '_std'))}  "
                  f"({int(r['n_subjects'])} subjects, {int(r.get('n_params', 0)):,} params)")
        print(f"  -> {run_dir / 'REPORT_TABLE.md'}")

    if len(results) > 1:
        out_dir = Path(args.out_dir) if args.out_dir else (
            Path(args.run[0]) if Path(args.run[0]).is_absolute()
            else PROJECT_ROOT / args.run[0])
        out_dir = out_dir / "tables" if out_dir.name != "tables" else out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        mt = main_table(results)
        mt.to_csv(out_dir / "main_table.csv")
        print(f"\ncross-dataset main table -> {out_dir / 'main_table.csv'}")


if __name__ == "__main__":
    main()

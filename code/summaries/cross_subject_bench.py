"""Aggregation and statistics for the cross-subject benchmark.

What this module exists to do
----------------------------
Reporting ``mean +/- std over subjects`` is not enough to support the claim "model A beats
model B". With 25 subjects and a between-subject standard deviation of ~10pp, the *unpaired*
detectable difference is about 5.6pp — larger than the entire observed model range (5.4pp on
SHU, 6.2pp on the 3C cohort). Every model comparison in this project is therefore
underpowered unless it is **paired**: subject difficulty cancels in the per-subject
difference, which drops the standard error to ~1.0pp at n=25 and ~0.7pp at n=51.

Pairing is legitimate here because all models are trained on identical
``(split_seed, fold, subject)`` triples in a single run — that is the whole reason the plan
submits all 17 entries together instead of staging them.

Layers, from raw to reported
----------------------------
1. ``load_run``      — one row per (model, split_seed, fold, train_seed, test_subject).
2. ``subject_matrix``— subjects x models, averaged over split seeds. This is the unit of
   every statistical test below.
3. ``per_model_table``— the headline mean +/- std, plus spread over split seeds.
4. ``paired_tests``  — Wilcoxon signed-rank on the pre-registered comparisons only
   (``PREREGISTERED_COMPARISONS``), Holm-corrected inside that family, with Cohen's
   ``d_z`` and win/tie/loss counts.
5. ``friedman_nemenyi`` + ``cd_diagram`` — the omnibus test and the critical-difference
   figure that benchmark papers report (Demsar 2006). The Nemenyi critical value is computed
   from ``scipy.stats.studentized_range`` rather than a lookup table, so it is exact for any
   number of models.
6. ``select_by_validation`` — picks the representative of a variant family using the
   **validation** score recorded in each cell (``best_score``). Choosing the variant by test
   accuracy would be test-set selection; this is the legitimate alternative and it needs no
   extra compute because the value is already stored.

Multiple-comparison policy
--------------------------
Two families, deliberately separated:

* the pre-registered comparisons (7 of them) answer the questions the experiment was
  designed for, and are Holm-corrected among themselves;
* the all-pairs ranking is reported through Friedman + Nemenyi, which is the standard
  omnibus procedure and does not pretend that 136 pairwise tests are independent.

Running 136 Holm-corrected pairwise tests instead would be so conservative that nothing
could reach significance, which is a statement about the procedure rather than the models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Metrics carried through every table. The first three are the ERP-Benchmark
#: (arXiv:2601.00573) reporting set; Kappa matches DSGNet's Table II; the rest is calibration.
METRICS: List[str] = ["accuracy", "macro_f1", "auc", "balanced_accuracy", "cohen_kappa",
                      "nll", "brier", "ece"]

#: Metric used for ranking and the CD diagram unless overridden.
PRIMARY_METRIC = "accuracy"

#: Comparisons the experiment was designed to answer, declared here so the set cannot grow
#: after the numbers are known. ``(baseline, candidate, question)``.
PREREGISTERED_COMPARISONS: List[Tuple[str, str, str]] = [
    # Fusion direction A: does ATCNet's depthwise (band-specific) spatial filter bank beat
    # ShallowNet's dense one? Single-variable: everything after the stem is identical.
    ("s4erp", "s4erp_dw@f64d2", "fusion A: depthwise stem, 64 bands x 2"),
    ("s4erp", "s4erp_dw@f16d8", "fusion A: depthwise stem, 16 bands x 8"),
    ("s4erp", "s4erp_dw@f16d2p", "fusion A: depthwise stem, ATCNet's 16 x 2 + projection"),
    # Fusion direction B: does one shared causal SSM beat five independent TCN branches?
    ("atcnet", "atcnet_s4", "fusion B: shared causal SSM replaces 5 TCNs"),
    ("atcnet@noattn", "atcnet_s4@noattn", "fusion B, attention removed from both sides"),
    # ATCNet ablations: which component actually carries its advantage?
    ("atcnet", "atcnet@tcn8", "ablation: is the TCN over-parameterized? (73.1% of params)"),
    ("atcnet", "atcnet@nwin1", "ablation: how much is the 5-window logit ensemble worth?"),
    ("atcnet", "atcnet@noattn", "ablation: is the attention block decorative?"),
]

#: Variant families whose representative is chosen by validation score (never by test).
VARIANT_FAMILIES: Dict[str, List[str]] = {
    "s4erp_dw": ["s4erp_dw@f64d2", "s4erp_dw@f16d8", "s4erp_dw@f16d2p"],
}


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_run(run_dir: str | Path) -> pd.DataFrame:
    """Every completed row of a run: one per (model, split_seed, fold, seed, test_subject).

    Reads ``cells/*/result.json``, which is the authoritative per-cell record (the CSVs under
    ``runs/`` are per-model views written by whichever job happened to run that model).
    """
    run_dir = Path(run_dir)
    cells = sorted((run_dir / "cells").glob("*/result.json"))
    if not cells:
        raise SystemExit(f"no completed cells under {run_dir / 'cells'}")
    records: List[Dict] = []
    for p in cells:
        detail = json.loads(p.read_text(encoding="utf-8"))
        # `split_seed` landed in the per-cell record on 2026-08-27. Cells written before
        # that carry it only in their directory name (`<model>__split41__fold0__seed0`),
        # so fall back to parsing the tag rather than silently labelling everything 0.
        cell_split = detail.get("split_seed")
        if cell_split is None:
            tag_parts = [t for t in p.parent.name.split("__") if t.startswith("split")]
            cell_split = int(tag_parts[0][len("split"):]) if tag_parts else 0
        for r in detail.get("rows") or []:
            if r.get("status") != "ok":
                continue
            r = dict(r)
            r.setdefault("split_seed", cell_split)
            r["cell"] = p.parent.name
            records.append(r)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit(f"{run_dir}: cells exist but none has an ok row")
    for c in ("split_seed", "fold", "seed"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: how many (split_seed, fold) cells and subjects are present.

    A partially finished sweep must not be read as if it were complete — the 7-fold
    DualCD row in ``docs/RESULTS_TRUTH.md`` §2 is exactly the mistake this table prevents.
    """
    g = df.groupby("model")
    out = pd.DataFrame({
        "n_rows": g.size(),
        "n_cells": g["cell"].nunique(),
        "n_subjects": g["test_subject"].nunique(),
        "split_seeds": g["split_seed"].apply(lambda s: sorted(set(s.dropna().tolist()))),
    })
    return out.sort_index()


# --------------------------------------------------------------------------- #
# 2. Subjects x models matrix — the unit of every test
# --------------------------------------------------------------------------- #
def subject_matrix(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """``subjects x models`` matrix, averaging over split seeds (and train seeds).

    Averaging first, then testing over subjects, is what keeps the observations independent:
    the same subject under two split seeds is not two independent measurements.
    """
    if metric not in df.columns:
        raise KeyError(f"metric '{metric}' not in rows; have {sorted(set(df.columns) & set(METRICS))}")
    mat = df.pivot_table(index="test_subject", columns="model", values=metric, aggfunc="mean")
    return mat.sort_index()


def complete_cases(mat: pd.DataFrame, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Drop subjects that are missing for any of ``models`` (paired tests need full rows)."""
    sub = mat if models is None else mat[[m for m in models if m in mat.columns]]
    return sub.dropna(axis=0, how="any")


# --------------------------------------------------------------------------- #
# 3. Headline table
# --------------------------------------------------------------------------- #
def per_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std over subjects for every metric, plus the spread over split seeds.

    ``std_subjects`` is the sample standard deviation across subjects — the number that
    belongs in the paper ("mean +/- std over subjects"). ``std_splits`` is the standard
    deviation of the per-split-seed means, i.e. how much the *partition* matters; on the
    previous SHU run it was 0.38-1.21pp against ~10pp between subjects.
    """
    rows: List[Dict] = []
    for model, g in df.groupby("model"):
        subj = g.groupby("test_subject")[
            [m for m in METRICS if m in g.columns]
        ].mean()
        rec: Dict[str, object] = {"model": model, "n_subjects": int(len(subj))}
        for m in METRICS:
            if m in subj.columns:
                rec[m] = float(subj[m].mean())
                rec[f"{m}_std"] = float(subj[m].std(ddof=1)) if len(subj) > 1 else 0.0
        if "split_seed" in g.columns and g["split_seed"].nunique() > 1:
            per_split = g.groupby("split_seed")[PRIMARY_METRIC].mean()
            rec["std_splits"] = float(per_split.std(ddof=1))
        else:
            rec["std_splits"] = float("nan")
        rec["val_score"] = float(g["best_score"].mean()) if "best_score" in g else float("nan")
        rec["n_params"] = int(pd.to_numeric(g["n_params"], errors="coerce").max() or 0)
        rec["best_epoch_median"] = float(pd.to_numeric(g["best_epoch"],
                                                       errors="coerce").median())
        rec["gpu_hours"] = float(pd.to_numeric(g["train_seconds"], errors="coerce")
                                 .groupby(g["cell"]).first().sum() / 3600.0)
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("model")
    return out.sort_values(PRIMARY_METRIC, ascending=False)


# --------------------------------------------------------------------------- #
# 4. Paired tests on the pre-registered comparisons
# --------------------------------------------------------------------------- #
def _holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


@dataclass
class PairedResult:
    baseline: str
    candidate: str
    question: str
    n: int
    mean_baseline: float
    mean_candidate: float
    delta: float           # candidate - baseline, in the metric's own units
    median_delta: float
    d_z: float             # Cohen's d for paired data: mean(diff) / std(diff)
    wins: int
    ties: int
    losses: int
    p_value: float
    p_holm: float = float("nan")


def paired_tests(
    mat: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str, str]] = tuple(PREREGISTERED_COMPARISONS),
    tie_eps: float = 1e-12,
) -> pd.DataFrame:
    """Wilcoxon signed-rank per comparison, Holm-corrected within the family.

    Wilcoxon rather than a paired t-test: per-subject accuracy differences on EEG are
    routinely non-normal and heavy-tailed, and n is 25-51. Cohen's ``d_z`` is reported
    alongside because a p-value alone says nothing about whether a difference matters —
    every model-level effect in this project so far has been smaller than the
    between-subject spread.

    Comparisons whose models are absent from ``mat`` are skipped (a partially finished sweep
    should still produce the table it can).
    """
    from scipy.stats import wilcoxon

    results: List[PairedResult] = []
    for base, cand, question in comparisons:
        if base not in mat.columns or cand not in mat.columns:
            continue
        pair = mat[[base, cand]].dropna()
        if len(pair) < 3:
            continue
        b = pair[base].to_numpy(dtype=float)
        c = pair[cand].to_numpy(dtype=float)
        diff = c - b
        sd = float(np.std(diff, ddof=1))
        try:
            _, p = wilcoxon(c, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            # all differences zero -> no evidence of a difference
            p = 1.0
        results.append(PairedResult(
            baseline=base, candidate=cand, question=question, n=int(len(pair)),
            mean_baseline=float(b.mean()), mean_candidate=float(c.mean()),
            delta=float(diff.mean()), median_delta=float(np.median(diff)),
            d_z=(float(diff.mean() / sd) if sd > 0 else float("nan")),
            wins=int((diff > tie_eps).sum()), ties=int((np.abs(diff) <= tie_eps).sum()),
            losses=int((diff < -tie_eps).sum()), p_value=float(p),
        ))
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame([r.__dict__ for r in results])
    out["p_holm"] = _holm(out["p_value"].tolist())
    out["significant_holm_05"] = out["p_holm"] < 0.05
    return out.sort_values("p_holm")


# --------------------------------------------------------------------------- #
# 5. Omnibus ranking: Friedman + Nemenyi
# --------------------------------------------------------------------------- #
def friedman_nemenyi(mat: pd.DataFrame, alpha: float = 0.05) -> Dict[str, object]:
    """Friedman omnibus test + Nemenyi critical difference over the complete-case matrix.

    Ranks are computed per subject (1 = best model on that subject) and averaged. The
    Nemenyi critical difference is

    .. math:: CD = q_{\\alpha} \\sqrt{\\frac{k(k+1)}{6N}}

    with :math:`q_{\\alpha}` the studentized-range quantile at infinite degrees of freedom
    divided by :math:`\\sqrt 2`. Computing it from ``scipy.stats.studentized_range`` avoids
    the usual lookup table, which in Demsar (2006) only goes to k=10 while we have 17.
    """
    from scipy.stats import friedmanchisquare, studentized_range

    full = mat.dropna(axis=0, how="any")
    k, N = full.shape[1], full.shape[0]
    if k < 3 or N < 3:
        return {"error": f"need >=3 models and >=3 subjects, got k={k}, N={N}"}
    stat, p = friedmanchisquare(*[full[c].to_numpy() for c in full.columns])
    # rank 1 = best; higher metric is better for accuracy/f1/auc/kappa
    ranks = full.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()
    q_alpha = float(studentized_range.ppf(1 - alpha, k, 1e6)) / np.sqrt(2.0)
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))
    return {"friedman_stat": float(stat), "friedman_p": float(p), "k": int(k), "N": int(N),
            "avg_ranks": avg_ranks, "critical_difference": cd, "alpha": alpha}


def cd_diagram(avg_ranks: pd.Series, cd: float, path: str | Path,
               title: str = "") -> None:
    """Demsar-style critical-difference diagram (lower rank = better, drawn on the left)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = avg_ranks.sort_values()
    names = list(ranks.index)
    vals = ranks.to_numpy(dtype=float)
    k = len(names)
    lo, hi = float(np.floor(vals.min())), float(np.ceil(vals.max()))

    # Cliques: maximal runs whose rank span is below CD. Computed up front because the
    # figure height depends on how many bars end up being drawn.
    drawn: List[Tuple[int, int]] = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in drawn):
            drawn.append((i, j))

    n_left = (k + 1) // 2
    clique_y = -0.6 * (n_left + 1) - 0.35
    y_min = clique_y - 0.22 * max(len(drawn) - 1, 0) - 0.35

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * (2.4 - y_min)))
    ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.set_ylim(y_min, 2.4)
    ax.axis("off")

    # rank axis
    ax.plot([lo, hi], [0, 0], "k-", lw=1.2)
    for t in np.arange(lo, hi + 0.001, 1.0):
        ax.plot([t, t], [0, 0.16], "k-", lw=1.0)
        ax.text(t, 0.3, f"{t:.0f}", ha="center", va="bottom", fontsize=9)

    # CD ruler
    ax.plot([lo, lo + cd], [1.5, 1.5], "k-", lw=1.6)
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [1.38, 1.62], "k-", lw=1.6)
    ax.text(lo + cd / 2, 1.72, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    # model labels, alternating sides for readability
    for i, name in enumerate(names[:n_left]):
        y = -0.6 * (i + 1)
        r = float(ranks[name])
        ax.plot([r, r, lo - 0.35], [0, y, y], "k-", lw=0.9)
        ax.text(lo - 0.4, y, f"{name}  ({r:.2f})", ha="right", va="center", fontsize=8.5)
    for i, name in enumerate(names[n_left:]):
        y = -0.6 * (n_left - i)
        r = float(ranks[name])
        ax.plot([r, r, hi + 0.35], [0, y, y], "k-", lw=0.9)
        ax.text(hi + 0.4, y, f"({r:.2f})  {name}", ha="left", va="center", fontsize=8.5)

    for n, (i, j) in enumerate(drawn):
        y = clique_y - 0.22 * n
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], "k-", lw=3.2, solid_capstyle="round")

    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Variant selection by validation score
# --------------------------------------------------------------------------- #
def select_by_validation(
    df: pd.DataFrame,
    families: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    """Pick each variant family's representative by mean **validation** score.

    ``best_score`` is the monitored validation metric at the epoch that produced ``best.pt``,
    recorded per cell. Selecting on it is legitimate; selecting on test accuracy would
    inflate the headline by roughly the spread of the family. All variants stay in the
    tables — this only marks which one represents the family in the abstract.
    """
    families = VARIANT_FAMILIES if families is None else families
    rows: List[Dict] = []
    for family, members in families.items():
        present = [m for m in members if m in set(df["model"])]
        if not present:
            continue
        sub = df[df["model"].isin(present)]
        val = sub.groupby("model")["best_score"].mean()
        test = sub.groupby("model")[PRIMARY_METRIC].mean()
        winner = val.idxmax()
        for m in present:
            rows.append({
                "family": family, "model": m,
                "val_score": float(val[m]), "test_score": float(test[m]),
                "selected": bool(m == winner),
                "val_rank": int(val.rank(ascending=False)[m]),
                "test_rank": int(test.rank(ascending=False)[m]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. Cross-dataset main table
# --------------------------------------------------------------------------- #
def main_table(runs: Dict[str, pd.DataFrame], metrics: Sequence[str] = ("accuracy", "macro_f1", "auc")
               ) -> pd.DataFrame:
    """One row per model, one column block per dataset: ``mean +/- std`` strings.

    Kept as formatted strings because this is the paste-into-the-paper artifact; the numeric
    versions live in each run's ``per_model.csv``.
    """
    frames = []
    for dataset, df in runs.items():
        t = per_model_table(df)
        block = pd.DataFrame(index=t.index)
        for m in metrics:
            if m in t.columns:
                block[(dataset, m)] = [
                    f"{100 * v:.2f} ± {100 * s:.2f}"
                    for v, s in zip(t[m], t[f"{m}_std"])
                ]
        block[(dataset, "n_subj")] = t["n_subjects"]
        frames.append(block)
    out = pd.concat(frames, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out

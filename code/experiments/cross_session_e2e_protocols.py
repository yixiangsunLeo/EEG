"""Subject-specific cross-session protocol on the end-to-end trainer.

Train on one set of sessions and test on later sessions **of the same subject**, one
model per subject. This is the SHU 2022 dataset paper's "cross-session" (CS) condition,
generalized so the train/test session lists come from the config:

  * paper CS          — ``train_sessions=[ses-01]``, ``test_sessions=[ses-02..05]``
  * this project's run — ``train_sessions=[ses-01,ses-02,ses-03]``,
                         ``test_sessions=[ses-04,ses-05]``  (advisor, 2026-08-16)

Why not ``session_protocols.run_cross_session``
-----------------------------------------------
That function drives the finished Phase 0-2c work and must stay bit-reproducible. It also
enumerates every *directed session pair* (20 per subject here), retrains for each, and
runs on the frozen ``trainer.py`` — which has no resume and logs no per-epoch curves. The
advisor asked for three curves + resumable cells, so this protocol runs on
``code/training/e2e_trainer.py`` instead, exactly like the cross-subject track. Cell
layout, checkpoint policy and ``result.json`` format therefore match
``cross_subject_protocols.py``, and the same summarize/plot scripts work on both.

Split policy
------------
``train_sessions`` and ``test_sessions`` must be disjoint, and a subject is only used when
it has every one of them ``ok``. The validation slice is a **stratified trial slice of the
pooled training sessions** (``val_fraction``), which is the only place it can come from:
the dataset paper trains on source sessions alone and never touches the test sessions
before scoring. Its own within-session benchmark splits 8:1:1, so ``val_fraction=1/9``
reproduces that 8:1 train:val ratio on the training pool.

Leakage rules enforced here
---------------------------
  * test sessions never enter train or val (asserted per fold);
  * the val slice is carved from the training sessions only;
  * normalization is per-trial and fit-free, so no statistic crosses the split;
  * the per-epoch test curve is monitoring only — ``best.pt`` and early stopping read the
    validation loader alone (guard lives in ``e2e_trainer.py``);
  * every fold's session lists are written to ``splits/`` as JSON.

Per cell = (model, subject, seed). Each cell writes ``best.pt`` / ``last.pt`` plus a
``result.json`` completion marker, so an interrupted sweep resumes. Both checkpoints are
scored on every test session separately (the paper reports per test session) and pooled.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Subset

from ..datasets.session_splits import SessionRecord
from ..models.registry import build_model
from ..training.e2e_trainer import BEST_CKPT, LAST_CKPT, CellPaths, E2ESpec, train_end_to_end
from ..training.trainer import predict
from ..utils.io import save_json
from ..utils.paths import ses_id, sub_id
from ..utils.seed import set_seed
from .cross_subject_protocols import (
    SubjectData,
    _loader,
    _make_dataset,
    _trial_val_split,
    _val_evaluator,
    load_subject_data,
)
from .session_protocols import evaluate_predictions

CROSS_SESSION_RESULT_COLUMNS: List[str] = [
    "dataset", "model", "protocol", "subject", "train_sessions", "test_session",
    "seed", "n_train", "n_val", "n_test",
    "monitor", "best_score", "best_epoch", "epochs_run", "early_stopped",
    "n_params", "train_seconds",
    # metrics of the BEST checkpoint on this test session
    "accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece",
    # metrics of the LAST-epoch checkpoint on the same session
    "last_accuracy", "last_balanced_accuracy", "last_macro_f1", "last_auc",
    "status", "error",
]


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
@dataclass
class CrossSessionFold:
    """One subject's session-level split (train sessions -> test sessions)."""

    subject: str
    train_sessions: List[str]
    test_sessions: List[str]

    def assert_disjoint(self) -> None:
        tr, te = set(self.train_sessions), set(self.test_sessions)
        if not tr or not te:
            raise AssertionError(
                f"{self.subject}: empty train_sessions or test_sessions "
                f"(train={sorted(tr)}, test={sorted(te)})"
            )
        if tr & te:
            raise AssertionError(
                f"{self.subject}: session leak between train and test ({sorted(tr & te)})"
            )


def normalize_session_list(sessions: Sequence[str | int]) -> List[str]:
    """Config sessions (``1`` / ``"ses-01"`` / ``"01"``) -> sorted canonical ids."""
    return sorted({ses_id(s) for s in sessions})


def make_cross_session_folds(
    subject_data: "Dict[str, SubjectData]",
    *,
    train_sessions: Sequence[str | int],
    test_sessions: Sequence[str | int],
    logger=None,
) -> List[CrossSessionFold]:
    """One fold per subject that has every train and test session available.

    A subject missing any required session is skipped loudly rather than silently
    trained on a smaller session set, which would make its row incomparable.
    """
    tr = normalize_session_list(train_sessions)
    te = normalize_session_list(test_sessions)
    if set(tr) & set(te):
        raise ValueError(
            f"train_sessions and test_sessions must be disjoint (got {tr} vs {te})"
        )
    folds: List[CrossSessionFold] = []
    for subj in sorted(subject_data):
        have = {ses_id(s) for s in subject_data[subj].sessions}
        missing = sorted((set(tr) | set(te)) - have)
        if missing:
            if logger is not None:
                logger.warning("skip %s: missing ok sessions %s (has %s)",
                               subj, missing, sorted(have))
            continue
        folds.append(CrossSessionFold(subject=subj, train_sessions=list(tr),
                                      test_sessions=list(te)))
    for f in folds:
        f.assert_disjoint()
    if not folds:
        raise ValueError(
            f"no subject has all of train={tr} + test={te}; check the manifest / config."
        )
    return folds


def _subject_index(subject: str) -> int:
    """Stable small int per subject id, for the per-epoch seed base."""
    m = re.search(r"(\d+)", str(subject))
    if m:
        return int(m.group(1))
    return int(hashlib.sha256(str(subject).encode("utf-8")).hexdigest()[:6], 16)


def cell_signature(fold: CrossSessionFold, *, dataset: str, model_name: str,
                   model_params: Dict, data_dims: Dict, spec: E2ESpec,
                   protocol: str, seed: int) -> str:
    """Fingerprint of everything a resumed cell must not have changed."""
    payload = {
        "dataset": dataset, "model": model_name, "protocol": protocol,
        "seed": int(seed), "subject": fold.subject,
        "train_sessions": sorted(fold.train_sessions),
        "test_sessions": sorted(fold.test_sessions),
        "data_dims": {k: data_dims[k] for k in sorted(data_dims)},
        "model_params": {k: model_params[k] for k in sorted(model_params)},
        "normalization": spec.normalization,
        "monitor": spec.monitor, "monitor_mode": spec.monitor_mode,
        "optimizer": spec.optimizer, "lr": spec.lr, "weight_decay": spec.weight_decay,
        "batch_size": spec.batch_size, "scheduler": spec.scheduler,
        "grad_clip_norm": spec.grad_clip_norm, "val_fraction": spec.val_fraction,
        "micro_batch_size": spec.micro_batch_size,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def run_cell(
    subject_data: "Dict[str, SubjectData]",
    fold: CrossSessionFold,
    *,
    dataset: str,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: E2ESpec,
    seed: int,
    device: torch.device,
    ckpt_dir: Path,
    result_json: Path,
    protocol: str = "cross_session",
    resume: bool = True,
    logger=None,
) -> List[Dict[str, object]]:
    """Train one subject's model and score it on each held-out session."""
    fold.assert_disjoint()
    paths = CellPaths(ckpt_dir=Path(ckpt_dir), result_json=Path(result_json))
    signature = cell_signature(fold, dataset=dataset, model_name=model_name,
                               model_params=model_params, data_dims=data_dims,
                               spec=spec, protocol=protocol, seed=seed)
    if resume and paths.is_complete():
        with open(paths.result_json, "r", encoding="utf-8") as f:
            done = json.load(f)
        stored = done.get("cell_signature")
        if stored is not None and stored != signature:
            raise RuntimeError(
                f"{paths.result_json} was produced under a different configuration "
                f"(stored signature {stored}, current {signature}). Use a fresh --out for "
                "the new setting, or --no-resume to overwrite."
            )
        if logger is not None:
            logger.info("skip completed cell: %s", paths.result_json)
        return list(done.get("rows") or [])

    set_seed(int(seed))
    pin = device.type == "cuda"
    subj = fold.subject

    train_ds, ytr = _make_dataset(subject_data, [subj], sessions=fold.train_sessions)
    if train_ds is None:
        raise ValueError(f"{subj}: no trials in train_sessions {fold.train_sessions}")
    tr_idx, va_idx = _trial_val_split(ytr, spec.val_fraction, int(seed))

    micro_bs = int(spec.micro_batch_size or spec.batch_size)
    train_loader = _loader(Subset(train_ds, tr_idx.tolist()), batch_size=micro_bs,
                           shuffle=True, num_workers=spec.num_workers, pin=pin)
    val_loader = (None if va_idx is None else
                  _loader(Subset(train_ds, va_idx.tolist()), batch_size=spec.batch_size,
                          shuffle=False, num_workers=spec.num_workers, pin=pin))
    n_train = int(len(tr_idx))
    n_val = 0 if va_idx is None else int(len(va_idx))

    # Monitoring-only loaders for the three curves. The test curve pools every held-out
    # session into one line (advisor's choice); the per-session numbers are computed
    # separately below and stored in result.json.
    train_eval_loader = test_curve_loader = None
    if getattr(spec, "curves", False):
        cap = int(getattr(spec, "train_eval_max_trials", 2000) or 0)
        if cap > 0 and n_train > cap:
            # Deterministic subset so the curve is comparable across epochs and resumes.
            sub = np.linspace(0, n_train - 1, num=cap, dtype=int).tolist()
            train_eval_ds = Subset(train_ds, [int(tr_idx[i]) for i in sub])
        else:
            train_eval_ds = Subset(train_ds, tr_idx.tolist())
        train_eval_loader = _loader(train_eval_ds, batch_size=spec.batch_size, shuffle=False,
                                    num_workers=spec.num_workers, pin=pin)
        test_pool_ds, _ = _make_dataset(subject_data, [subj], sessions=fold.test_sessions)
        if test_pool_ds is not None:
            test_curve_loader = _loader(test_pool_ds, batch_size=spec.batch_size,
                                        shuffle=False, num_workers=spec.num_workers, pin=pin)

    model = build_model(model_name, params=dict(model_params), **data_dims)
    desc = model.describe() if hasattr(model, "describe") else {}
    if logger is not None:
        logger.info("[cell] %s %s | train=%s (%d trials, val %d) test=%s | params=%s",
                    model_name, subj, ",".join(fold.train_sessions), n_train, n_val,
                    ",".join(fold.test_sessions), desc.get("n_params"))

    info = train_end_to_end(
        model, train_loader, val_loader, spec=spec, device=device,
        ckpt_dir=paths.ckpt_dir, resume=resume, evaluate_fn=_val_evaluator(),
        epoch_seed_base=int(seed) * 100_000 + _subject_index(subj) * 1_000,
        run_signature=signature,
        train_eval_loader=train_eval_loader, test_loader=test_curve_loader,
        logger=logger,
    )

    # Score BOTH checkpoints on every held-out session separately (the dataset paper
    # reports per test session, 's2'..'s5'), then once on the pooled test sessions.
    rows: List[Dict[str, object]] = []
    per_session_detail: List[Dict[str, object]] = []
    last_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_state = (torch.load(paths.best, map_location=device, weights_only=False)["model_state"]
                  if paths.best.exists() else None)

    def _score(sessions: Sequence[str]):
        ds, y = _make_dataset(subject_data, [subj], sessions=list(sessions))
        if ds is None:
            return None, None, 0
        loader = _loader(ds, batch_size=spec.batch_size, shuffle=False,
                         num_workers=spec.num_workers, pin=pin)
        if best_state is not None:
            model.load_state_dict(best_state)
        yt, yp, pr, _ = predict(model, loader, device=device)
        best_m = evaluate_predictions(yt, yp, pr)
        model.load_state_dict(last_state)
        yt2, yp2, pr2, _ = predict(model, loader, device=device)
        last_m = evaluate_predictions(yt2, yp2, pr2)
        return best_m, last_m, int(y.numel())

    common = {
        "dataset": dataset, "model": model_name, "protocol": protocol,
        "subject": subj, "train_sessions": "+".join(fold.train_sessions),
        "seed": int(seed), "n_train": n_train, "n_val": n_val,
        "monitor": info["monitor"], "best_score": float(info["best_score"]),
        "best_epoch": int(info["best_epoch"]), "epochs_run": int(info["epochs_run"]),
        "early_stopped": bool(info["early_stopped"]),
        "n_params": int(desc.get("n_params", 0)),
        "train_seconds": float(info["train_seconds"]),
        "status": "ok", "error": "",
    }

    for ses in fold.test_sessions:
        best_m, last_m, n_test = _score([ses])
        if best_m is None:
            continue
        rows.append({
            **common, "test_session": ses, "n_test": n_test,
            **{k: float(v) for k, v in best_m.items()},
            "last_accuracy": float(last_m["accuracy"]),
            "last_balanced_accuracy": float(last_m["balanced_accuracy"]),
            "last_macro_f1": float(last_m["macro_f1"]),
            "last_auc": float(last_m["auc"]),
        })
        per_session_detail.append({
            "test_session": ses, "n_trials": n_test,
            "best": {k: float(v) for k, v in best_m.items()},
            "last": {k: float(v) for k, v in last_m.items()},
        })

    pooled_best, pooled_last, n_pooled = _score(fold.test_sessions)
    pooled = None
    if pooled_best is not None:
        pooled = {
            "n_trials": n_pooled,
            "best": {k: float(v) for k, v in pooled_best.items()},
            "last": {k: float(v) for k, v in pooled_last.items()},
        }

    if best_state is not None:
        model.load_state_dict(best_state)

    save_json({
        "dataset": dataset, "model": model_name, "fold": asdict(fold), "seed": int(seed),
        "protocol": protocol, "cell_signature": signature, "data_dims": data_dims,
        "model_params": model_params, "model_desc": desc, "spec": asdict(spec),
        "train_info": {k: v for k, v in info.items() if k != "history"},
        "history": info["history"], "per_session": per_session_detail,
        "pooled_test": pooled, "rows": rows,
        "checkpoints": {"best": str(paths.best), "last": str(paths.last)},
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, paths.result_json)
    return rows


# --------------------------------------------------------------------------- #
# Sweep over models x subjects x seeds
# --------------------------------------------------------------------------- #
def run_cross_session_e2e(
    records: Sequence[SessionRecord],
    *,
    dataset: str,
    models: Sequence[str],
    model_params_all: Dict[str, Dict],
    data_dims: Dict,
    spec: E2ESpec,
    seeds: Sequence[int],
    device: torch.device,
    train_sessions: Sequence[str | int],
    test_sessions: Sequence[str | int],
    subjects_subset: Optional[Sequence[str]] = None,
    normalization: str = "none",
    micro_batch_per_model: Optional[Dict[str, int]] = None,
    out_dir: Path,
    ckpt_root: Optional[Path] = None,
    resume: bool = True,
    protocol: str = "cross_session",
    logger=None,
) -> Dict[str, object]:
    """Run the whole cross-session sweep, resuming whatever is already on disk."""
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"
    cells_dir = out_dir / "cells"
    splits_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    subject_data = load_subject_data(
        records, n_channels=data_dims["n_channels"], n_times=data_dims["n_times"],
        normalization=normalization, logger=logger,
    )
    folds = make_cross_session_folds(
        subject_data, train_sessions=train_sessions, test_sessions=test_sessions,
        logger=logger,
    )
    save_json({
        "dataset": dataset, "protocol": protocol,
        "train_sessions": normalize_session_list(train_sessions),
        "test_sessions": normalize_session_list(test_sessions),
        "val_fraction": spec.val_fraction, "normalization": normalization,
        "n_subjects": len(folds), "subjects": [f.subject for f in folds],
        "trials_per_subject": {s: subject_data[s].n_trials for s in sorted(subject_data)},
        "folds": [asdict(f) for f in folds],
    }, splits_dir / "cross_session_folds.json")

    selected = folds
    if subjects_subset:
        # accepts 1 / '1' / 'sub-001', like every other subject filter in the project
        want = {sub_id(s) for s in subjects_subset}
        selected = [f for f in folds if f.subject in want]
    if logger is not None:
        logger.info("cross-session plan: dataset=%s train=%s test=%s subjects=%d/%d "
                    "models=%s seeds=%s",
                    dataset, normalize_session_list(train_sessions),
                    normalize_session_list(test_sessions), len(selected), len(folds),
                    list(models), list(seeds))

    all_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for model_name in models:
        mp = dict((model_params_all or {}).get(model_name, {}))
        micro_bs = (micro_batch_per_model or {}).get(model_name)
        spec_m = replace(spec, micro_batch_size=int(micro_bs)) if micro_bs else spec
        for fold in selected:
            for seed in seeds:
                tag = f"{model_name}__{fold.subject}__seed{int(seed)}"
                ck = (Path(ckpt_root) / tag) if ckpt_root is not None else (cells_dir / tag / "ckpt")
                try:
                    all_rows += run_cell(
                        subject_data, fold, dataset=dataset, model_name=model_name,
                        model_params=mp, data_dims=data_dims, spec=spec_m, seed=int(seed),
                        device=device, ckpt_dir=ck,
                        result_json=cells_dir / tag / "result.json",
                        protocol=protocol, resume=resume, logger=logger,
                    )
                except Exception as exc:  # keep the sweep alive; record and continue
                    msg = f"{type(exc).__name__}: {exc}"
                    failures.append({"cell": tag, "error": msg})
                    if logger is not None:
                        logger.exception("cell FAILED %s: %s", tag, msg)
                    all_rows.append({
                        "dataset": dataset, "model": model_name, "protocol": protocol,
                        "subject": fold.subject,
                        "train_sessions": "+".join(fold.train_sessions),
                        "test_session": "+".join(fold.test_sessions),
                        "seed": int(seed), "status": "failed", "error": msg,
                    })
    return {
        "rows": all_rows, "folds": [asdict(f) for f in folds], "failures": failures,
        "n_subjects": len(folds),
        "n_cells_ok": sum(1 for r in all_rows if r.get("status") == "ok"),
        "n_cells_failed": len(failures),
        "checkpoint_names": {"best": BEST_CKPT, "last": LAST_CKPT},
    }

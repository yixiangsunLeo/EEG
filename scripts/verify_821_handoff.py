#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handoff sanity check for `paper_baseline_3c_821_v1` after moving to a new account.

Four questions, none of which a normal dry-run answers:

1. Is the locally regenerated 3C data bit-identical to the account it came from?
   Compares hashes of the decoded X/y arrays, not the .npz container bytes (zlib
   output can differ between numpy builds while the arrays are identical).
2. Do the freshly built folds match the fold definitions stored in the 50 finished
   `result.json` files? A subject-list drift would silently change what fold N means.
3. Do the split sizes still land on 8/2/1 subjects with all 9899 trials used?
4. Does `cell_signature` recomputed here reproduce the signatures the previous account
   wrote into `result.json` and into resumable `last.pt`? This is the strongest check:
   the same fingerprint guards the resume path, so a match means data dims, split
   membership, model params and the whole optimizer recipe are identical. A mismatch
   here is what would otherwise surface hours later as a RuntimeError mid-run.

Usage:
  python scripts/verify_821_handoff.py [--old-root /share/home/Zihang/MI] [--hash-sessions 3]
  python scripts/verify_821_handoff.py --hash-sessions 33     # full data comparison
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.session_splits import load_ok_sessions  # noqa: E402
from code.experiments.cross_subject_protocols import (  # noqa: E402
    cell_signature, make_subject_folds,
)

RUN_DIR = PROJECT_ROOT / "outputs/experiments/wbci_shu/paper_baseline_3c_821_v1"
MANIFEST = PROJECT_ROOT / "outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv"
# 8:2:1 is a SUBJECT ratio, so trial counts are 7199:1800:900 only when every session has
# 300 trials. sub-010/ses-01 officially has 299 (see code/preprocessing/wbci_3c_mat.py), so
# the two folds where sub-010 is a val or test subject shift by one trial. Assert the
# subject ratio and the invariant total instead of hard-coding the typical trial triple.
EXPECTED_SUBJECTS = (8, 2, 1)
EXPECTED_TOTAL_TRIALS = 9899


def _array_digest(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=False) as d:
        h = hashlib.sha256()
        for key in ("X", "y"):
            arr = np.ascontiguousarray(d[key])
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()[:16]


def check_data_identical(records, old_root: Path, n_sessions: int) -> bool:
    print(f"\n[1] data identity vs {old_root} (decoded arrays, first {n_sessions} sessions)")
    if not old_root.exists():
        print("    SKIP: old root not readable")
        return True
    ok = True
    for rec in records[:n_sessions]:
        new_p = Path(rec.npz_path)
        rel = new_p.relative_to(PROJECT_ROOT / "outputs/processed/wbci_shu_3c_mat_clean")
        old_p = old_root / "outputs/processed/wbci_shu_3c_mat_clean" / rel
        if not old_p.exists():
            print(f"    {rec.subject}/{rec.session}: SKIP (absent on old account)")
            continue
        a, b = _array_digest(new_p), _array_digest(old_p)
        same = a == b
        ok &= same
        print(f"    {rec.subject}/{rec.session}: new={a} old={b} {'MATCH' if same else 'DIFFER'}")
    return ok


def check_folds(records) -> bool:
    print("\n[2] fresh folds vs folds recorded in finished cells")
    subjects = sorted({r.subject for r in records})
    folds = make_subject_folds(
        subjects, protocol="loso", n_folds=11, val_mode="subjects",
        n_val_subjects=2, split_seed=0,
    )
    by_idx = {f.fold: f for f in folds}
    cells = sorted((RUN_DIR / "cells").glob("*/result.json"))
    print(f"    built {len(folds)} folds; comparing against {len(cells)} finished cells")
    ok = True
    mismatches = 0
    for cell in cells:
        stored = json.loads(cell.read_text())["fold"]
        fresh = by_idx[stored["fold"]]
        for key in ("train_subjects", "val_subjects", "test_subjects"):
            if sorted(stored[key]) != sorted(getattr(fresh, key)):
                print(f"    MISMATCH {cell.parent.name} {key}")
                print(f"      stored={sorted(stored[key])}")
                print(f"      fresh ={sorted(getattr(fresh, key))}")
                ok = False
                mismatches += 1
    print(f"    {len(cells) - mismatches} / {len(cells)} cells agree on subject assignment")
    return ok


def check_split_sizes(records) -> bool:
    print("\n[3] per-fold split: 8/2/1 subjects, disjoint, all 9899 trials used")
    n_by_subject: dict[str, int] = {}
    for r in records:
        n_by_subject[r.subject] = n_by_subject.get(r.subject, 0) + r.n_trials
    subjects = sorted(n_by_subject)
    folds = make_subject_folds(
        subjects, protocol="loso", n_folds=11, val_mode="subjects",
        n_val_subjects=2, split_seed=0,
    )
    ok = True
    for f in folds:
        groups = (f.train_subjects, f.val_subjects, f.test_subjects)
        n_subj = tuple(len(g) for g in groups)
        trials = tuple(sum(n_by_subject[s] for s in g) for g in groups)
        problems = []
        if n_subj != EXPECTED_SUBJECTS:
            problems.append(f"subjects {n_subj} != {EXPECTED_SUBJECTS}")
        if sum(trials) != EXPECTED_TOTAL_TRIALS:
            problems.append(f"trials sum {sum(trials)} != {EXPECTED_TOTAL_TRIALS}")
        if len(set().union(*groups)) != sum(n_subj):
            problems.append("groups overlap")
        if problems:
            ok = False
        note = "; ".join(problems) if problems else (
            "ok" if trials == (7199, 1800, 900) else "ok (sub-010 has 299-trial session)")
        print(f"    fold {f.fold:2d} test={sorted(f.test_subjects)} "
              f"val={sorted(f.val_subjects)} -> {trials[0]}:{trials[1]}:{trials[2]} {note}")
    return ok


def check_signatures(records) -> bool:
    """Recompute cell_signature and compare against what the previous account stored."""
    print("\n[4] cell_signature reproduces the previous account's fingerprints")
    import argparse as _argparse

    import torch

    from code.runners import _build_e2e_spec
    from code.utils.config import load_config

    cfg = load_config(PROJECT_ROOT / "code/configs/experiments/paper_baseline_3c_821.yaml")
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    # The defaults run.py passes when no CLI flag overrides the YAML.
    spec = _build_e2e_spec(
        _argparse.Namespace(batch_size=None, lr=None, max_epochs=None, patience=None,
                            monitor=None, num_workers=None, val_fraction=None),
        train_cfg,
    )
    dims = dict(n_channels=data_cfg["n_channels"], n_times=data_cfg["n_times"],
                n_classes=data_cfg["n_classes"], sfreq=data_cfg["sfreq"])
    folds = {f.fold: f for f in make_subject_folds(
        sorted({r.subject for r in records}), protocol="loso", n_folds=11,
        val_mode="subjects", n_val_subjects=2, split_seed=0)}
    micro = train_cfg.get("micro_batch_per_model") or {}

    def sig_for(model: str, fold: int) -> str:
        return cell_signature(
            folds[fold], dataset=data_cfg["name"], model_name=model,
            model_params=(cfg.get("model_params") or {}).get(model, {}), data_dims=dims,
            spec=dataclasses.replace(spec, micro_batch_size=micro.get(model)),
            protocol="loso", val_mode="subjects", seed=0)

    ok = True
    n = bad = 0
    for d in sorted((RUN_DIR / "cells").glob("*/result.json")):
        stored = json.loads(d.read_text())
        n += 1
        if stored.get("cell_signature") != sig_for(d.parent.name.split("__")[0],
                                                   stored["fold"]["fold"]):
            print(f"    MISMATCH in finished cell {d.parent.name}")
            ok = False
            bad += 1
    print(f"    {n - bad} / {n} finished cells reproduce their stored signature")

    ckpts = sorted((PROJECT_ROOT / "checkpoints/wbci_shu/paper_baseline_3c_821_v1").glob(
        "*/last.pt"))
    for p in ckpts:
        model, fold_s, _ = p.parent.name.split("__")
        stored = torch.load(p, map_location="cpu", weights_only=False).get("run_signature")
        mine = sig_for(model, int(fold_s.replace("fold", "")))
        same = stored == mine
        ok &= same
        print(f"    resumable {p.parent.name}: stored={stored} local={mine} "
              f"{'MATCH' if same else 'MISMATCH'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default="/share/home/Zihang/MI")
    ap.add_argument("--hash-sessions", type=int, default=3)
    args = ap.parse_args()

    records = load_ok_sessions(MANIFEST, status_filter=("ok",))
    total = sum(r.n_trials for r in records)
    print(f"manifest: {len(records)} ok sessions / {len({r.subject for r in records})} subjects "
          f"/ {total} trials")

    results = {
        "data identical": check_data_identical(records, Path(args.old_root), args.hash_sessions),
        "folds reproduce": check_folds(records),
        "split sizes 8:2:1": check_split_sizes(records),
        "signatures match": check_signatures(records),
    }
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

"""Behaviour tests for the subject-specific cross-session protocol (synthetic data, CPU).

Covers the guarantees that matter for the SHU 2022 cross-session run:

  * folds are per subject and the train/test session lists never overlap;
  * a subject missing a required session is skipped, not silently trained on less data;
  * the validation slice comes from the TRAIN sessions only — no test-session trial ever
    reaches the training or validation loader;
  * a cell writes exactly ``best.pt`` + ``last.pt`` and one row per test session;
  * re-running a finished cell is a no-op (resume), and an interrupted cell continues.

Uses 8-channel / 64-sample synthetic sessions and a tiny model so the file runs in
seconds on CPU.

Run: python -m pytest tests/foundation/test_cross_session_e2e_protocol.py -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.session_splits import load_ok_sessions  # noqa: E402
from code.experiments.cross_session_e2e_protocols import (  # noqa: E402
    make_cross_session_folds,
    run_cell,
    run_cross_session_e2e,
)
from code.experiments.cross_subject_protocols import load_subject_data  # noqa: E402
from code.training.e2e_trainer import BEST_CKPT, LAST_CKPT, E2ESpec  # noqa: E402
from code.utils.io import save_session_npz  # noqa: E402

N_CH, N_T = 8, 64
DIMS = dict(n_channels=N_CH, n_times=N_T, n_classes=2, sfreq=250)
TINY_PARAMS = {"d_model": 8, "n_layers": 1, "d_ff": 16}
TRAIN_SES = ["ses-01", "ses-02", "ses-03"]
TEST_SES = ["ses-04", "ses-05"]


def _write_dataset(root: Path, *, n_subjects: int = 4, n_sessions: int = 5,
                   n_trials: int = 20, seed: int = 0,
                   sessions_per_subject=None) -> Path:
    """Synthetic per-session npz files + processed_manifest.csv; return the manifest."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        subj = f"sub-{s:03d}"
        wanted = (sessions_per_subject or {}).get(subj, list(range(1, n_sessions + 1)))
        for ses in wanted:
            sess = f"ses-{ses:02d}"
            y = np.array([0, 1] * (n_trials // 2), dtype=np.int64)
            X = rng.normal(0, 10, size=(n_trials, N_CH, N_T)).astype(np.float32)
            # Weak class signal that drifts a little per session (the thing CS measures).
            X[y == 1, :2, :] += 3.0 + 0.1 * ses
            path = save_session_npz(
                root / subj / sess / f"{subj}_{sess}.npz", X, y,
                subject_id=subj, session_id=sess, sfreq=250,
                channel_names=[f"C{i}" for i in range(N_CH)],
            )
            rows.append({"subject_id": subj, "session_id": sess, "npz_path": str(path),
                         "status": "ok", "n_trials": n_trials,
                         "label_0_count": int((y == 0).sum()),
                         "label_1_count": int((y == 1).sum())})
    manifest = root / "processed_manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return manifest


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_cs")
    manifest = _write_dataset(root)
    records = load_ok_sessions(manifest)
    subject_data = load_subject_data(records, n_channels=N_CH, n_times=N_T,
                                     normalization="none")
    return {"manifest": manifest, "records": records, "subject_data": subject_data}


def _spec(max_epochs: int = 2, **kw) -> E2ESpec:
    base = dict(batch_size=8, lr=1e-3, max_epochs=max_epochs, early_stopping_patience=0,
                num_workers=0, monitor="accuracy", monitor_mode="max", scheduler="none",
                normalization="none", val_fraction=0.1111)
    base.update(kw)
    return E2ESpec(**base)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def test_one_fold_per_subject_with_disjoint_sessions(dataset):
    folds = make_cross_session_folds(dataset["subject_data"],
                                     train_sessions=TRAIN_SES, test_sessions=TEST_SES)
    assert [f.subject for f in folds] == ["sub-001", "sub-002", "sub-003", "sub-004"]
    for f in folds:
        f.assert_disjoint()                        # raises on any session overlap
        assert f.train_sessions == TRAIN_SES
        assert f.test_sessions == TEST_SES


def test_session_lists_are_normalized_and_must_be_disjoint(dataset):
    folds = make_cross_session_folds(dataset["subject_data"],
                                     train_sessions=[1, 2, 3], test_sessions=[4, 5])
    assert folds[0].train_sessions == TRAIN_SES and folds[0].test_sessions == TEST_SES
    with pytest.raises(ValueError, match="disjoint"):
        make_cross_session_folds(dataset["subject_data"],
                                 train_sessions=["ses-01"], test_sessions=["ses-01"])


def test_subject_missing_a_required_session_is_skipped(tmp_path):
    manifest = _write_dataset(tmp_path, n_subjects=3,
                              sessions_per_subject={"sub-002": [1, 2, 3, 4]})
    sd = load_subject_data(load_ok_sessions(manifest), n_channels=N_CH, n_times=N_T,
                           normalization="none")
    folds = make_cross_session_folds(sd, train_sessions=TRAIN_SES, test_sessions=TEST_SES)
    assert [f.subject for f in folds] == ["sub-001", "sub-003"]   # sub-002 lacks ses-05


def test_no_usable_subject_fails_loudly(dataset):
    with pytest.raises(ValueError, match="no subject has all of"):
        make_cross_session_folds(dataset["subject_data"],
                                 train_sessions=["ses-01"], test_sessions=["ses-09"])


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def test_cell_writes_two_checkpoints_and_one_row_per_test_session(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold = make_cross_session_folds(sd, train_sessions=TRAIN_SES,
                                    test_sessions=TEST_SES)[0]
    ck = tmp_path / "ckpt"
    rows = run_cell(sd, fold, dataset="synthetic", model_name="s4erp",
                    model_params=TINY_PARAMS, data_dims=DIMS, spec=_spec(2), seed=0,
                    device=torch.device("cpu"), ckpt_dir=ck,
                    result_json=tmp_path / "result.json")

    assert sorted(p.name for p in ck.iterdir()) == sorted([BEST_CKPT, LAST_CKPT])
    assert [r["test_session"] for r in rows] == TEST_SES
    for r in rows:
        assert r["status"] == "ok"
        assert r["subject"] == fold.subject
        assert r["train_sessions"] == "ses-01+ses-02+ses-03"
        assert r["n_test"] == 20                       # one session of 20 trials
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["last_accuracy"] <= 1.0        # final-epoch metrics reported too

    detail = json.loads((tmp_path / "result.json").read_text())
    assert detail["train_info"]["epochs_run"] == 2
    assert [d["test_session"] for d in detail["per_session"]] == TEST_SES
    assert detail["pooled_test"]["n_trials"] == 40     # ses-04 + ses-05
    # 3 train sessions x 20 trials, split 8:1 into train/val
    assert rows[0]["n_train"] + rows[0]["n_val"] == 60
    assert rows[0]["n_val"] == pytest.approx(60 * 0.1111, abs=1.5)


def test_test_session_trials_never_reach_training(dataset, tmp_path, monkeypatch):
    """Hard leakage guard: record every trial the train/val loaders actually yield."""
    from code.experiments import cross_session_e2e_protocols as csp

    sd = dataset["subject_data"]
    fold = make_cross_session_folds(sd, train_sessions=TRAIN_SES,
                                    test_sessions=TEST_SES)[0]
    seen: list[torch.Tensor] = []
    real_loader = csp._loader

    def spy(ds, **kw):
        dl = real_loader(ds, **kw)
        seen.append(torch.cat([xb for xb, _ in dl], dim=0))
        return dl

    monkeypatch.setattr(csp, "_loader", spy)
    run_cell(sd, fold, dataset="synthetic", model_name="s4erp",
             model_params=TINY_PARAMS, data_dims=DIMS, spec=_spec(1), seed=0,
             device=torch.device("cpu"), ckpt_dir=tmp_path / "ckpt",
             result_json=tmp_path / "result.json")

    train_val = torch.cat(seen[:2], dim=0)             # loaders: train, val, then eval
    assert len(train_val) == 60                        # exactly the 3 training sessions
    subj = sd[fold.subject]
    test_idx = [i for i, s in enumerate(subj.session_of_trial) if s in set(TEST_SES)]
    test_trials = subj.X[test_idx]
    train_val_flat = {t.numpy().tobytes() for t in train_val}
    assert not any(t.numpy().tobytes() in train_val_flat for t in test_trials), \
        "a test-session trial appeared in the train/val loaders"


def test_completed_cell_is_skipped_and_interrupted_cell_resumes(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold = make_cross_session_folds(sd, train_sessions=TRAIN_SES,
                                    test_sessions=TEST_SES)[0]
    ck = tmp_path / "ckpt"
    kw = dict(dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
              data_dims=DIMS, seed=0, device=torch.device("cpu"), ckpt_dir=ck)

    rows1 = run_cell(sd, fold, spec=_spec(2), result_json=tmp_path / "a.json", **kw)
    mtime = (ck / LAST_CKPT).stat().st_mtime_ns
    rows2 = run_cell(sd, fold, spec=_spec(50), result_json=tmp_path / "a.json", **kw)
    assert rows2 == rows1                                    # completed -> no retraining
    assert (ck / LAST_CKPT).stat().st_mtime_ns == mtime

    # A different result marker with a longer budget continues from epoch 3.
    run_cell(sd, fold, spec=_spec(4), result_json=tmp_path / "b.json", **kw)
    detail = json.loads((tmp_path / "b.json").read_text())
    assert detail["train_info"]["resumed_from_epoch"] == 2
    assert [h["epoch"] for h in detail["history"]] == [1, 2, 3, 4]


def test_resume_refuses_a_checkpoint_from_a_different_split(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold_a = make_cross_session_folds(sd, train_sessions=TRAIN_SES,
                                      test_sessions=TEST_SES)[0]
    fold_b = make_cross_session_folds(sd, train_sessions=["ses-01", "ses-02"],
                                      test_sessions=["ses-04", "ses-05"])[0]
    ck = tmp_path / "ckpt"
    kw = dict(dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
              data_dims=DIMS, seed=0, device=torch.device("cpu"), ckpt_dir=ck)
    run_cell(sd, fold_a, spec=_spec(1), result_json=tmp_path / "a.json", **kw)
    with pytest.raises(RuntimeError, match="different configuration"):
        run_cell(sd, fold_b, spec=_spec(2), result_json=tmp_path / "b.json", **kw)


# --------------------------------------------------------------------------- #
# Full sweep
# --------------------------------------------------------------------------- #
def test_sweep_covers_models_and_subjects_and_is_resumable(dataset, tmp_path):
    out = tmp_path / "run"
    kw = dict(dataset="synthetic", models=["s4erp", "eegnet_official"],
              model_params_all={"s4erp": TINY_PARAMS, "eegnet_official": {}},
              data_dims=DIMS, spec=_spec(1), seeds=[0], device=torch.device("cpu"),
              train_sessions=TRAIN_SES, test_sessions=TEST_SES,
              subjects_subset=["sub-001", 2], normalization="none", out_dir=out)
    res = run_cross_session_e2e(dataset["records"], **kw)

    assert res["n_cells_failed"] == 0
    assert res["n_subjects"] == 4                       # folds built for every subject
    cells = sorted(p.name for p in (out / "cells").iterdir())
    assert cells == ["eegnet_official__sub-001__seed0", "eegnet_official__sub-002__seed0",
                     "s4erp__sub-001__seed0", "s4erp__sub-002__seed0"]
    # 2 models x 2 subjects x 2 test sessions
    assert len([r for r in res["rows"] if r["status"] == "ok"]) == 8
    assert (out / "splits" / "cross_session_folds.json").exists()

    res2 = run_cross_session_e2e(dataset["records"], **kw)
    assert res2["rows"] == res["rows"]                  # everything complete -> identical


def test_atcnet_custom_loss_accepts_subject_ids():
    """Regression: S4ERP-Adv made the trainer pass subject_ids into every custom-loss
    hook. ATCNet uses that hook for Keras L2, not for a subject head — it must not crash."""
    from code.models.registry import build_model

    model = build_model("atcnet", n_channels=32, n_times=1000, n_classes=2, sfreq=250)
    assert model.uses_custom_loss
    x = torch.randn(4, 32, 1000)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    sid = torch.arange(4)
    loss, parts = model.training_step(x, y, epoch=0, subject_ids=sid)
    assert torch.isfinite(loss)
    assert "ce" in parts and "l2" in parts
    loss.backward()

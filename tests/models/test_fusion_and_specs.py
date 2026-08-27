"""Tests for the P0 additions: model aliases, parameter resolution, and both fusion models.

These are the checks that would otherwise only be caught by a 3-hour GPU smoke test, so they
run on CPU with tiny batches. The parameter counts are asserted exactly: they are quoted in
the plan and in the model yamls, and a silent change to a stem or a head would invalidate the
"512x smaller spatial filter" claim without any test failing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from code.models.eeg_foundation.encoders import DepthwiseShallowNetEmbedding
from code.models.model_specs import (
    resolve_all,
    resolve_model_params,
    safe_tag,
    split_alias,
    variant_names,
)
from code.models.registry import BENCHMARK_MODEL_NAMES, build_model

DIMS_58 = dict(n_channels=58, n_times=1000, n_classes=2, sfreq=250)
DIMS_32 = dict(n_channels=32, n_times=1000, n_classes=2, sfreq=250)


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("atcnet", ("atcnet", "")),
    ("atcnet@tcn8", ("atcnet", "tcn8")),
    ("s4erp_dw@f16d2p", ("s4erp_dw", "f16d2p")),
])
def test_split_alias(name, expected):
    assert split_alias(name) == expected


@pytest.mark.parametrize("bad", ["", "atcnet@", "@tcn8", "atcnet@a@b"])
def test_split_alias_rejects_malformed(bad):
    with pytest.raises(ValueError):
        split_alias(bad)


def test_safe_tag_is_path_safe():
    assert safe_tag("atcnet@tcn8") == "atcnet-tcn8"
    assert safe_tag("atcnet") == "atcnet"
    assert "@" not in safe_tag("s4erp_dw@f64d2")


def test_undefined_alias_raises_rather_than_running_the_plain_model():
    """A typo must not land in the results table disguised as an ablation."""
    with pytest.raises(ValueError, match="not defined"):
        resolve_model_params("atcnet@tcn9")


def test_undefined_alias_raises_even_when_experiment_overrides_are_present():
    """A leftover model_params block used to skip the 'not defined' check."""
    with pytest.raises(ValueError, match="not defined"):
        resolve_model_params("atcnet@tcn88", experiment_params={"tcn_filters": 8})


# --------------------------------------------------------------------------- #
# Parameter resolution: single source of truth
# --------------------------------------------------------------------------- #
def test_variants_declared_in_yaml():
    assert set(variant_names("atcnet")) == {"tcn8", "nwin1", "noattn"}
    assert set(variant_names("atcnet_s4")) == {"noattn"}
    assert set(variant_names("s4erp_dw")) == {"f64d2", "f16d8", "f16d2p"}


@pytest.mark.parametrize("name,key,value", [
    ("atcnet", "tcn_filters", 32),
    ("atcnet@tcn8", "tcn_filters", 8),
    ("atcnet@nwin1", "n_windows", 1),
    ("atcnet@noattn", "attention", None),
    ("s4erp_dw@f64d2", "dw_F1", 64),
    ("s4erp_dw@f16d8", "dw_D", 8),
])
def test_variant_overrides_defaults(name, key, value):
    assert resolve_model_params(name)[key] == value


def test_timepatch_gets_mi_bins_not_erp_bins():
    """The ERP defaults (0-750 ms) would dump ~90% of a 4 s trial into the last bin.

    This regression exists because the value lived in the model yaml while experiment
    configs overrode `model_params` without it, so the yaml never took effect.
    """
    bins = resolve_model_params("dualcd_s4_timepatch")["bin_boundaries_ms"]
    assert bins == [0, 500, 1000, 1500, 2000, 3000, 4000]
    assert bins[-1] == 4000, "bins must cover the whole 4 s MI trial"


def test_experiment_params_override_yaml():
    got = resolve_model_params("atcnet", experiment_params={"tcn_filters": 99})
    assert got["tcn_filters"] == 99


def test_resolve_all_covers_the_whole_roster():
    res = resolve_all(BENCHMARK_MODEL_NAMES, dataset="wbci_shu_2c")
    assert set(res) == set(BENCHMARK_MODEL_NAMES)
    assert len(BENCHMARK_MODEL_NAMES) == 18
    assert "atcnet_s4@noattn" in BENCHMARK_MODEL_NAMES


def test_experiment_yamls_match_the_canonical_roster():
    """A yaml that silently drops an entry would un-pair the Holm family."""
    import yaml

    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "code" / "configs" / "experiments"
    for name in ("wbci_2c_cross_subject_bench.yaml", "shu_cross_subject_bench.yaml"):
        models = yaml.safe_load((root / name).read_text())["models"]
        assert models == BENCHMARK_MODEL_NAMES, name


# --------------------------------------------------------------------------- #
# Fusion direction A: depthwise spatial front-end
# --------------------------------------------------------------------------- #
def test_depthwise_stem_preserves_patch_num_and_width():
    """patch_num must match the dense stem, or the backbone and head would need changing."""
    dense_patch_num = (1000 - 28) // 2 + 1
    stem = DepthwiseShallowNetEmbedding(c_in=58, d_model=128, F1=16, D=2)
    out = stem(torch.zeros(2, 1000, 58))
    assert out.shape == (2, dense_patch_num, 128)


@pytest.mark.parametrize("F1,D,expected_stem", [
    (64, 2, 25_984),
    (16, 8, 24_736),
    (16, 2, 6_592),
])
def test_depthwise_stem_param_counts(F1, D, expected_stem):
    stem = DepthwiseShallowNetEmbedding(c_in=58, d_model=128, F1=F1, D=D)
    assert sum(p.numel() for p in stem.parameters()) == expected_stem


def test_depthwise_stem_is_512x_cheaper_than_dense_spatial_conv():
    """The headline claim of fusion direction A, asserted rather than asserted in prose."""
    from code.models.eeg_foundation.encoders import ShallowNetEmbedding

    dense = ShallowNetEmbedding(c_in=58, d_model=128)
    dense_spatial = sum(p.numel() for p in dense.shallow_net[1].parameters())
    dw = DepthwiseShallowNetEmbedding(c_in=58, d_model=128, F1=16, D=2)
    dw_spatial = sum(p.numel() for p in dw.shallow_net[1].parameters())
    assert dense_spatial == 950_400
    assert dw_spatial == 1_888
    assert dense_spatial / dw_spatial > 400


def test_depthwise_grouping_is_actually_band_specific():
    """Output filter j must depend only on temporal filter j // D, not on all of them."""
    stem = DepthwiseShallowNetEmbedding(c_in=8, d_model=4, F1=2, D=2)
    w = stem.shallow_net[1].weight
    assert w.shape == (4, 1, 8, 1), "depthwise kernel must have in_channels/groups == 1"


@pytest.mark.parametrize("variant,expected_total", [
    ("s4erp_dw@f64d2", 426_370),
    ("s4erp_dw@f16d8", 425_122),
    ("s4erp_dw@f16d2p", 406_978),
])
def test_s4erp_dw_total_params_and_forward(variant, expected_total):
    m = build_model(variant, params=resolve_model_params(variant), **DIMS_58)
    assert m.describe()["n_params"] == expected_total
    out = m(torch.zeros(2, 58, 1000))
    assert out["logits"].shape == (2, 2)
    assert out["features"].shape == (2, 62_336)


def test_s4erp_dw_is_a_single_variable_change_from_s4erp():
    """Only the stem may differ; the S4 encoder and classifier must be identical in size."""
    a = build_model("s4erp", params=resolve_model_params("s4erp"), **DIMS_58)
    b = build_model("s4erp_dw@f64d2", params=resolve_model_params("s4erp_dw@f64d2"),
                    **DIMS_58)
    enc = lambda m: sum(p.numel() for p in m.model.encoder.parameters())      # noqa: E731
    head = lambda m: sum(p.numel() for p in m.model.classifier.parameters())  # noqa: E731
    assert enc(a) == enc(b)
    assert head(a) == head(b)
    assert sum(p.numel() for p in a.model.embedding.parameters()) \
        > sum(p.numel() for p in b.model.embedding.parameters())


# --------------------------------------------------------------------------- #
# Fusion direction B: shared causal SSM
# --------------------------------------------------------------------------- #
def test_atcnet_s4_reads_out_at_the_official_window_ends():
    """ATCNet's 5 windows end at tokens 12..16; the SSM must be read at the same places."""
    m = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4"), **DIMS_58)
    d = m.describe()
    assert d["Tc"] == 17
    assert d["readout_idx"] == [12, 13, 14, 15, 16]


def test_atcnet_s4_timescales_rescaled_for_short_sequence():
    """The default log_dt spans 6.8-546 steps, tuned for S4ERP's 487 patches.

    Left unchanged, 7 of 8 modes would have tau >> Tc=17 and the layer would collapse to a
    running mean — a null result that would be misread as "the SSM does not work here".
    """
    m = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4"), **DIMS_58)
    d = m.describe()
    assert d["tau_max_tokens"] <= d["Tc"] * 1.2, d
    assert d["tau_min_tokens"] < 2.0, d


def test_atcnet_s4_is_much_smaller_than_atcnet():
    a = build_model("atcnet", params=resolve_model_params("atcnet"), **DIMS_58)
    b = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4"), **DIMS_58)
    na = a.describe()["n_params"]
    nb = b.describe()["n_params"]
    assert na == 114_554
    assert nb == 27_834
    assert na / nb > 4.0


def test_atcnet_s4_attention_is_causal():
    """A non-causal mask would let readout 12 see token 16, collapsing the 5-way ensemble."""
    from code.models.atcnet.atcnet_s4 import _CausalMHABlock

    torch.manual_seed(0)
    block = _CausalMHABlock(embed_dim=8, dropout=0.0).eval()
    x = torch.randn(1, 6, 8)
    base = block(x)
    x2 = x.clone()
    x2[:, -1, :] += 10.0                      # perturb only the LAST position
    perturbed = block(x2)
    # every position before the last must be unchanged
    assert torch.allclose(base[:, :-1, :], perturbed[:, :-1, :], atol=1e-5)
    assert not torch.allclose(base[:, -1, :], perturbed[:, -1, :], atol=1e-5)


def test_atcnet_s4_noattn_variant_drops_the_attention_block():
    m = build_model("atcnet_s4@noattn",
                    params=resolve_model_params("atcnet_s4@noattn"), **DIMS_58)
    assert m.model.attention_block is None
    assert m.describe()["n_params"] == 25_642


def test_atcnet_and_atcnet_s4_carry_the_same_l2_scheme():
    """Otherwise the ablation is a two-variable change and means nothing.

    ATCNet folds L2(conv 0.009) + L2(dense 0.5) into the loss. Measured at initialization
    that term is ~10-17x the cross-entropy — it dominates early training. A variant without
    it would differ from ``atcnet`` in the temporal stage AND in the regularizer.
    """
    x, y = torch.randn(4, 58, 1000), torch.tensor([0, 1, 0, 1])
    for name in ("atcnet", "atcnet_s4", "atcnet@noattn", "atcnet_s4@noattn"):
        m = build_model(name, params=resolve_model_params(name), **DIMS_58).train()
        assert m.uses_custom_loss, f"{name}: L2 penalty is not wired into the loss"
        loss, parts = m.training_step(x, y, epoch=0)
        assert set(parts) == {"ce", "l2"}, f"{name}: {parts}"
        assert parts["l2"] > 0, name
        assert torch.isfinite(loss), name


def test_atcnet_s4_l2_excludes_the_ssm_dynamics_parameters():
    """Dynamics and the skip must not be penalized.

    ``log_dt``: shrinking it pushes dt toward 1 and destroys the timescale
    calibration that ``reinit_log_dt_for_length`` sets up. ``D``: initialized at 1
    while the SSM kernel is ~1e-4, so it is the only live temporal path; L2 on D
    would collapse the stage. ``B``/``C`` stand in for the TCN conv kernels and
    stay under the 0.009 penalty.
    """
    m = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4"), **DIMS_58)
    m.model.l2_penalty().backward()
    for block in m.model.s4:
        for pname in ("log_dt", "A_log", "A_sign", "D"):
            g = getattr(block.s4, pname).grad
            assert g is None or torch.all(g == 0), f"L2 reached {pname}"
        # B/C are parametrized (max_norm); the leaf is `.parametrizations.*.original`.
        for pname in ("B", "C"):
            g = block.s4.parametrizations[pname].original.grad
            assert g is not None and torch.any(g != 0), f"L2 did not reach {pname}"
        if block.has_ffn:
            assert block.ffn[0].weight.grad is not None


def test_atcnet_s4_ssm_maps_carry_tcn_max_norm():
    """ATCNet TCN kernels are max_norm(0.6); B/C stand in for those kernels."""
    from code.models.atcnet.atcnet_torch import CONV_MAX_NORM

    m = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4"), **DIMS_58)
    layer = m.model.s4[0].s4
    with torch.no_grad():
        layer.parametrizations.B.original.fill_(10.0)
        layer.parametrizations.C.original.fill_(10.0)
    b_norm = torch.linalg.vector_norm(layer.B, dim=1).max().item()
    c_norm = torch.linalg.vector_norm(layer.C, dim=1).max().item()
    assert b_norm <= CONV_MAX_NORM + 1e-5
    assert c_norm <= CONV_MAX_NORM + 1e-5


# --------------------------------------------------------------------------- #
# The whole roster builds and runs on both datasets' shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dims,dataset", [(DIMS_58, "wbci_shu_2c"), (DIMS_32, "shu")])
def test_every_roster_entry_builds_and_reports_params(dims, dataset):
    """No zero parameter counts, and one identical params dict works on 58ch and 32ch."""
    params = resolve_all(BENCHMARK_MODEL_NAMES, dataset=dataset)
    for name in BENCHMARK_MODEL_NAMES:
        m = build_model(name, params=params[name], **dims)
        desc = m.describe() if hasattr(m, "describe") else {}
        assert desc.get("n_params", 0) > 0, f"{name} reports n_params={desc.get('n_params')}"


@pytest.mark.slow
@pytest.mark.parametrize("dims", [DIMS_58, DIMS_32])
def test_every_roster_entry_forward_pass(dims):
    params = resolve_all(BENCHMARK_MODEL_NAMES, dataset="wbci_shu_2c")
    x = torch.zeros(2, dims["n_channels"], dims["n_times"])
    for name in BENCHMARK_MODEL_NAMES:
        m = build_model(name, params=params[name], **dims).eval()
        with torch.no_grad():
            out = m(x)
        logits = out["logits"] if isinstance(out, dict) else out
        assert logits.shape == (2, dims["n_classes"]), name
        assert torch.isfinite(logits).all(), name


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_every_roster_entry_runs_on_gpu():
    """Forward, loss, and ``describe()`` on cuda, for the whole roster.

    ``describe()`` is diagnostics, but the trainer calls it on every best-checkpoint
    save, so a tensor built without ``device=`` there kills the cell after training.
    A CPU-only run cannot see it: both operands are on the CPU. This is what took out
    all four ``atcnet_s4`` cells of smoke 47237.
    """
    dev = torch.device("cuda")
    params = resolve_all(BENCHMARK_MODEL_NAMES, dataset="wbci_shu_2c")
    x = torch.randn(2, DIMS_58["n_channels"], DIMS_58["n_times"], device=dev)
    y = torch.tensor([0, 1], device=dev)
    for name in BENCHMARK_MODEL_NAMES:
        m = build_model(name, params=params[name], **DIMS_58).to(dev).train()
        loss, _ = m.training_step(x, y, epoch=0)
        assert torch.isfinite(loss), name
        desc = m.describe() if hasattr(m, "describe") else {}
        assert desc.get("n_params", 0) > 0, name


# --------------------------------------------------------------------------- #
# Signature guard
# --------------------------------------------------------------------------- #
def _sig_kw():
    from code.experiments.cross_subject_protocols import CrossSubjectFold

    return CrossSubjectFold(0, ["sub-001"], ["sub-002"], ["sub-003"]), dict(
        dataset="d", model_name="m", model_params={}, data_dims={"n_classes": 2},
        protocol="kfold_subject", val_mode="subjects", seed=0)


def test_cell_signature_always_covers_patience():
    """v1 let patience change without invalidating a cell, silently mixing two selection
    rules into one results table."""
    from dataclasses import replace

    from code.experiments.cross_subject_protocols import SIGNATURE_VERSION, cell_signature
    from code.training.e2e_trainer import E2ESpec

    assert SIGNATURE_VERSION >= 2
    fold, kw = _sig_kw()
    for scheduler in ("none", "cosine"):
        spec = E2ESpec(max_epochs=500, early_stopping_patience=25, scheduler=scheduler)
        assert (cell_signature(fold, spec=spec, **kw)
                != cell_signature(fold, spec=replace(spec, early_stopping_patience=50), **kw))


def test_max_epochs_only_defines_identity_when_a_scheduler_is_active():
    """Its role depends on the scheduler.

    Under ``scheduler: none`` it is a pure cap, so raising it and resuming is a strict
    continuation of the same trajectory — the way a wall-clock-killed job is restarted.
    Under a cosine schedule it *is* the annealing horizon, so changing it changes the
    learning rate at every remaining step and the cell is no longer the same experiment.
    """
    from dataclasses import replace

    from code.experiments.cross_subject_protocols import cell_signature
    from code.training.e2e_trainer import E2ESpec

    fold, kw = _sig_kw()
    no_sched = E2ESpec(max_epochs=500, early_stopping_patience=25, scheduler="none")
    assert (cell_signature(fold, spec=no_sched, **kw)
            == cell_signature(fold, spec=replace(no_sched, max_epochs=200), **kw))

    cosine = replace(no_sched, scheduler="cosine")
    assert (cell_signature(fold, spec=cosine, **kw)
            != cell_signature(fold, spec=replace(cosine, max_epochs=200), **kw))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_holm_is_monotone_and_conservative():
    from code.summaries.cross_subject_bench import _holm

    p = [0.001, 0.01, 0.04, 0.5]
    adj = _holm(p)
    assert np.all(adj >= np.asarray(p))
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "must be non-decreasing"
    assert np.all(adj <= 1.0)


def test_holm_uses_the_pre_registered_family_size():
    """A partial sweep must not make the remaining tests easier to call significant."""
    from code.summaries.cross_subject_bench import _holm

    p = [0.04]
    assert _holm(p, n_family=1)[0] == 0.04
    assert _holm(p, n_family=8)[0] == min(1.0, 0.32)


def test_nemenyi_cd_shrinks_with_more_subjects():
    """CD ~ sqrt(k(k+1)/6N): more subjects -> a smaller detectable rank difference."""
    import pandas as pd

    from code.summaries.cross_subject_bench import friedman_nemenyi

    rng = np.random.default_rng(0)
    def cd_for(n):
        mat = pd.DataFrame(rng.random((n, 5)), columns=list("abcde"))
        return friedman_nemenyi(mat)["critical_difference"]

    assert cd_for(80) < cd_for(20)


def test_friedman_ranks_flip_for_lower_is_better_metrics():
    """NLL/Brier/ECE: rank 1 must be the smallest value, not the largest."""
    import pandas as pd

    from code.summaries.cross_subject_bench import friedman_nemenyi

    mat = pd.DataFrame({
        "good": [0.1, 0.2, 0.15],
        "mid": [0.5, 0.4, 0.45],
        "bad": [1.0, 0.9, 1.1],
    })
    hi = friedman_nemenyi(mat, higher_is_better=True)["avg_ranks"]
    lo = friedman_nemenyi(mat, higher_is_better=False)["avg_ranks"]
    assert hi["bad"] < hi["good"]
    assert lo["good"] < lo["bad"]


def test_friedman_drops_incomplete_models_not_subjects():
    """One unfinished DualCD must not collapse N for the whole CD diagram."""
    import pandas as pd

    from code.summaries.cross_subject_bench import friedman_nemenyi

    mat = pd.DataFrame({
        "a": [0.7, 0.6, 0.8, 0.5],
        "b": [0.6, 0.5, 0.7, 0.4],
        "c": [0.5, 0.4, 0.6, 0.3],
        "unfinished": [0.9, None, None, None],
    })
    out = friedman_nemenyi(mat)
    assert "unfinished" in out["dropped_models"]
    assert "unfinished" not in out["avg_ranks"].index
    assert out["N"] == 4
    assert out["k"] == 3


def test_paired_tests_wins_follow_metric_direction_and_report_a_ci():
    import pandas as pd

    from code.summaries.cross_subject_bench import paired_tests

    mat = pd.DataFrame({
        "atcnet": [0.70, 0.60, 0.65, 0.55],
        "atcnet_s4": [0.72, 0.63, 0.64, 0.58],
    })
    cmp = [("atcnet", "atcnet_s4", "fusion B")]
    acc = paired_tests(mat, comparisons=cmp, higher_is_better=True)
    nll = paired_tests(mat, comparisons=cmp, higher_is_better=False)
    assert int(acc.iloc[0]["wins"]) == 3
    assert int(acc.iloc[0]["losses"]) == 1
    assert int(nll.iloc[0]["wins"]) == 1
    assert int(nll.iloc[0]["losses"]) == 3
    assert acc.iloc[0]["delta_ci95_lo"] < acc.iloc[0]["delta"] < acc.iloc[0]["delta_ci95_hi"]


def test_preregistered_family_is_eight_and_includes_the_noattn_pair():
    from code.summaries.cross_subject_bench import PREREGISTERED_COMPARISONS

    assert len(PREREGISTERED_COMPARISONS) == 8
    assert ("atcnet@noattn", "atcnet_s4@noattn",
            "fusion B, attention removed from both sides") in PREREGISTERED_COMPARISONS


def test_save_json_is_strict_and_atomic(tmp_path):
    from code.utils.io import load_json, save_json

    path = tmp_path / "t.json"
    save_json({"kappa": float("nan"), "ok": 1.0}, path)
    got = load_json(path)
    assert got["kappa"] is None
    assert got["ok"] == 1.0

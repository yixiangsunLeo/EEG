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


# --------------------------------------------------------------------------- #
# Parameter resolution: single source of truth
# --------------------------------------------------------------------------- #
def test_variants_declared_in_yaml():
    assert set(variant_names("atcnet")) == {"tcn8", "nwin1", "noattn"}
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
    assert len(BENCHMARK_MODEL_NAMES) == 17


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
    assert nb == 23_482
    assert na / nb > 4.5


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
    m = build_model("atcnet_s4", params=resolve_model_params("atcnet_s4@noattn"), **DIMS_58)
    assert m.model.attention_block is None
    assert m.describe()["n_params"] == 21_290


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


def test_nemenyi_cd_shrinks_with_more_subjects():
    """CD ~ sqrt(k(k+1)/6N): more subjects -> a smaller detectable rank difference."""
    import pandas as pd

    from code.summaries.cross_subject_bench import friedman_nemenyi

    rng = np.random.default_rng(0)
    def cd_for(n):
        mat = pd.DataFrame(rng.random((n, 5)), columns=list("abcde"))
        return friedman_nemenyi(mat)["critical_difference"]

    assert cd_for(80) < cd_for(20)

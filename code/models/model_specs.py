"""Single source of truth for model structural parameters, plus the model-alias mechanism.

Two problems this module solves
------------------------------

**1. Parameters were maintained in two places.** ``code/configs/models/<name>.yaml`` held
"this model's standard hyperparameters" while every experiment config repeated them inline
under ``model_params``. ``code/configs/models/README.md`` even warned "两处都要改，别只改一处".
That warning was already violated: ``dualcd_s4_timepatch.yaml`` carries MI-appropriate
``bin_boundaries_ms``, but the experiment configs' inline ``model_params`` omit the key, so
the model silently fell back to the source package's ERP defaults (0-750 ms) and dumped
~90% of a 4 s trial's patches into the last bin.

Now the yaml is authoritative and the experiment config only lists model *names*. An
experiment may still override a value, but it has to do so explicitly and the override is
recorded in the cell signature.

**2. One model, several configurations.** Ablations like "ATCNet with ``tcn_filters=8``"
used to need a whole separate ``run_id``, because ``model_params`` is keyed by model name
and a run cannot hold two different ``atcnet`` entries. An **alias** ``atcnet@tcn8`` fixes
that: the part before ``@`` selects the implementation, the part after selects a named
variant defined in the model's own yaml. Aliases are distinct model names everywhere
downstream (``model_params`` keys, cell directory tags, ``cell_signature``), so several
configurations of one model coexist in a single run and stay pairwise comparable.

YAML layout
-----------

``defaults`` / ``per_dataset`` / ``variants`` are optional; any other top-level key is
treated as a default, which keeps every pre-existing (flat) yaml working unchanged::

    name: "atcnet"
    defaults:
      tcn_filters: 32
      n_windows: 5
    per_dataset:            # keyed by the experiment's `data.name`
      shu: {}               # empty = no dataset-specific change
    variants:
      tcn8:   {tcn_filters: 8}      # -> atcnet@tcn8
      nwin1:  {n_windows: 1}        # -> atcnet@nwin1
      noattn: {attention: null}     # -> atcnet@noattn

Merge order, later wins::

    flat top-level keys -> defaults -> per_dataset[dataset] -> variants[alias]
                        -> experiment model_params[full_name]

``per_dataset`` exists because a model *may* need a structural change to fit a dataset
(compatibility, not tuning). For WBCIC-SHU 2C and SHU 2022 nothing is needed: both are
1000 samples @ 250 Hz and the channel count is injected by the experiment layer, so every
entry is currently empty. Keeping the mechanism means a third dataset does not require a
code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.logging_utils import get_logger

logger = get_logger("models.model_specs")

#: Separator between the implementation name and the named variant, e.g. ``atcnet@tcn8``.
ALIAS_SEP = "@"

#: Top-level yaml keys that are structure, not hyperparameters.
_RESERVED = frozenset({"name", "description", "defaults", "per_dataset", "variants"})

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"


def split_alias(name: str) -> Tuple[str, str]:
    """``'atcnet@tcn8'`` -> ``('atcnet', 'tcn8')``; ``'atcnet'`` -> ``('atcnet', '')``.

    Raises on an empty base or a bare trailing separator so a typo fails loudly instead of
    silently resolving to the plain model.
    """
    raw = (name or "").strip()
    if ALIAS_SEP not in raw:
        if not raw:
            raise ValueError("empty model name")
        return raw, ""
    base, _, variant = raw.partition(ALIAS_SEP)
    base, variant = base.strip(), variant.strip()
    if not base or not variant:
        raise ValueError(
            f"malformed model alias {name!r}; expected '<model>{ALIAS_SEP}<variant>' "
            f"(e.g. 'atcnet{ALIAS_SEP}tcn8')"
        )
    if ALIAS_SEP in variant:
        raise ValueError(f"model alias {name!r} has more than one '{ALIAS_SEP}'")
    return base, variant


def base_name(name: str) -> str:
    """Implementation name behind a possibly-aliased model name."""
    return split_alias(name)[0]


def safe_tag(name: str) -> str:
    """Filesystem/shell-safe form of a model name: ``'atcnet@tcn8'`` -> ``'atcnet-tcn8'``.

    Used for cell directory names, CSV filenames and Slurm job names. The logical name
    (with ``@``) is what goes into ``model_params``, result rows and ``cell_signature``;
    only paths use this form, and the mapping is injective because ``@`` may appear once
    and variant names are not allowed to contain ``-`` ambiguity in practice.
    """
    return split_alias(name)[0] if ALIAS_SEP not in name else name.replace(ALIAS_SEP, "-")


def load_model_yaml(base: str, models_dir: Optional[Path] = None) -> Dict:
    """Raw contents of ``configs/models/<base>.yaml`` (empty dict if the file is absent)."""
    import yaml

    path = Path(models_dir or _DEFAULT_MODELS_DIR) / f"{base}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping, got {type(data).__name__}")
    return data


def resolve_model_params(
    name: str,
    *,
    dataset: Optional[str] = None,
    models_dir: Optional[Path] = None,
    experiment_params: Optional[Dict] = None,
) -> Dict:
    """Merged structural parameters for one (possibly aliased) model name.

    ``experiment_params`` is the experiment config's ``model_params[name]`` block and wins
    over everything in the yaml.
    """
    base, variant = split_alias(name)
    spec = load_model_yaml(base, models_dir)

    merged: Dict = {k: v for k, v in spec.items() if k not in _RESERVED}
    for key, block in (("defaults", spec.get("defaults")),):
        if block:
            if not isinstance(block, dict):
                raise ValueError(f"models/{base}.yaml: '{key}' must be a mapping")
            merged.update(block)

    per_dataset = spec.get("per_dataset") or {}
    if per_dataset and not isinstance(per_dataset, dict):
        raise ValueError(f"models/{base}.yaml: 'per_dataset' must be a mapping")
    if dataset and dataset in per_dataset:
        block = per_dataset[dataset] or {}
        if not isinstance(block, dict):
            raise ValueError(f"models/{base}.yaml: per_dataset['{dataset}'] must be a mapping")
        merged.update(block)

    variants = spec.get("variants") or {}
    if variants and not isinstance(variants, dict):
        raise ValueError(f"models/{base}.yaml: 'variants' must be a mapping")
    overrides = dict(experiment_params or {})
    if variant:
        if variant not in variants:
            # A typo here would silently run the plain model under an ablation's name
            # (even if the experiment config happens to carry some other override) and
            # land in the results table as if it were the ablation. Fail instead.
            # Ad-hoc variants belong in the model's yaml, not in experiment_params.
            raise ValueError(
                f"model alias '{name}' is not defined: models/{base}.yaml has no "
                f"variants['{variant}']. Known variants: {sorted(variants) or 'none'}"
            )
        block = variants[variant] or {}
        if not isinstance(block, dict):
            raise ValueError(
                f"models/{base}.yaml: variants['{variant}'] must be a mapping"
            )
        merged.update(block)

    merged.update(overrides)
    merged.pop("name", None)
    return merged


def resolve_all(
    names: Sequence[str],
    *,
    dataset: Optional[str] = None,
    models_dir: Optional[Path] = None,
    experiment_params_all: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """``resolve_model_params`` for a whole model list, keyed by the full (aliased) name."""
    experiment_params_all = experiment_params_all or {}
    unknown = sorted(set(experiment_params_all) - set(names))
    if unknown:
        logger.warning(
            "model_params has entries for models not in this run's list: %s", unknown
        )
    out: Dict[str, Dict] = {}
    for name in names:
        out[name] = resolve_model_params(
            name, dataset=dataset, models_dir=models_dir,
            experiment_params=experiment_params_all.get(name),
        )
    return out


def variant_names(base: str, models_dir: Optional[Path] = None) -> List[str]:
    """Named variants declared in ``configs/models/<base>.yaml``."""
    return sorted((load_model_yaml(base, models_dir).get("variants") or {}))


__all__ = [
    "ALIAS_SEP", "split_alias", "base_name", "safe_tag", "load_model_yaml",
    "resolve_model_params", "resolve_all", "variant_names",
]

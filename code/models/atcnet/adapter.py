"""Project-facing wrapper for the official ATCNet port.

ATCNet is the strongest published baseline on the SHUv5 three-class LOSO table of the
DSGNet paper (Acc 0.6834 vs DSGNet 0.6856), so it is the calibration point for our
end-to-end runs. ``atcnet_torch.ATCNetOfficial`` is a 1:1 port of the official Keras
``ATCNet_``; this file only translates it into the project contract
``{logits, features, confidence}`` (see ``.cursor/rules/30-model-experiments``).

Upstream folds its ``L2`` kernel regularizers into the loss, so we expose the same
penalty through the trainer's ``uses_custom_loss`` / ``training_step`` hook instead of
silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eeg_foundation.losses import GradientReversalLayer
from .atcnet_s4 import ATCNetS4
from .atcnet_torch import ATCNetOfficial

#: Official recommended recipe (main_TrainValTest.py L413): Adam lr=0.001, batch 64.
DEFAULT_LR = 1e-3


@dataclass
class ATCNetConfig:
    """ATCNet structural hyperparameters — all values are the upstream defaults.

    Data dims come from the experiment layer, never hard-coded here.
    """

    # data dims (shared by every model in this project)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # upstream ATCNet_ signature (models.py L34-L37)
    n_windows: int = 5
    eegn_F1: int = 16
    eegn_D: int = 2
    eegn_kernelSize: int = 64
    eegn_poolSize: int = 7
    eegn_dropout: float = 0.3
    tcn_depth: int = 2
    tcn_kernelSize: int = 4
    tcn_filters: int = 32
    tcn_dropout: float = 0.3
    attention: Optional[str] = "mha"
    fuse: str = "average"

    #: Add the official Keras L2 kernel penalties to the loss (upstream behaviour).
    use_official_l2: bool = True


class ATCNetClassifier(nn.Module):
    """Official ATCNet with this project's dict contract.

    ``forward(x)``: ``x`` is ``[B, C, T]``, returns
    ``{"logits": [B, n_classes], "features": [B, d], "confidence": None}``.

    ``features``: ATCNet has no single penultimate embedding — each sliding window feeds
    its own dense head and the *logits* are averaged. We report the mean of the per-window
    TCN features (the tensors the dense heads consume), which is the closest analogue to
    the other models' penultimate features. Nothing in the forward maths is changed.
    """

    def __init__(self, config: ATCNetConfig | None = None):
        super().__init__()
        cfg = config or ATCNetConfig()
        self.config = cfg

        self.model = ATCNetOfficial(
            n_classes=cfg.n_classes,
            in_chans=cfg.n_channels,
            in_samples=cfg.n_times,
            n_windows=cfg.n_windows,
            eegn_F1=cfg.eegn_F1,
            eegn_D=cfg.eegn_D,
            eegn_kernelSize=cfg.eegn_kernelSize,
            eegn_poolSize=cfg.eegn_poolSize,
            eegn_dropout=cfg.eegn_dropout,
            tcn_depth=cfg.tcn_depth,
            tcn_kernelSize=cfg.tcn_kernelSize,
            tcn_filters=cfg.tcn_filters,
            tcn_dropout=cfg.tcn_dropout,
            attention=cfg.attention,
            fuse=cfg.fuse,
        )
        self.feature_dim = int(
            cfg.tcn_filters * (cfg.n_windows if cfg.fuse == "concat" else 1)
        )
        self.uses_custom_loss = bool(cfg.use_official_l2)

    def default_lr(self) -> float:
        return DEFAULT_LR

    def describe(self) -> Dict[str, object]:
        return {
            "name": "atcnet",
            "source": "official Altaheri/EEG-ATCNet ATCNet_ (Keras) ported to PyTorch",
            "n_params": sum(p.numel() for p in self.parameters()),
            "feature_dim": self.feature_dim,
            "n_windows": int(self.model.n_windows),
            "Tc": int(self.model.Tc),
            "Tw": int(self.model.Tw),
            "official_l2": bool(self.uses_custom_loss),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        logits = self.model(x)
        window_feats = self.model.last_window_features
        if self.config.fuse == "concat":
            features = torch.cat(window_feats, dim=1)
        else:
            features = (window_feats[0] if len(window_feats) == 1
                        else torch.stack(window_feats, dim=0).mean(dim=0))
        return {"logits": logits, "features": features, "confidence": None}

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0,
                      subject_ids: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE + the official Keras L2 kernel penalties (Keras adds them to the loss).

        ``subject_ids`` is accepted and ignored: the e2e trainer always has the
        argument for S4ERP-Adv, and ATCNet has no subject-adversarial head.
        """
        ce = F.cross_entropy(self.forward(x)["logits"], y)
        l2 = self.model.l2_penalty()
        return ce + l2, {"ce": float(ce.detach().item()),
                         "l2": float(l2.detach().item())}


@dataclass
class ATCNetS4Config:
    """``atcnet_s4`` — ATCNet's conv block + one shared causal SSM (fusion direction B).

    Conv-block fields mirror ``ATCNetConfig`` so a config block can be copied across;
    ``tcn_*`` are replaced by ``s4_*``. See ``atcnet_s4.ATCNetS4`` for the design rationale.
    """

    # data dims (shared by every model in this project)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # convolutional (CV) block — identical to ATCNet's upstream defaults
    eegn_F1: int = 16
    eegn_D: int = 2
    eegn_kernelSize: int = 64
    eegn_poolSize: int = 7
    eegn_dropout: float = 0.3

    # one shared causal attention block ("mha") or none
    attention: Optional[str] = "mha"

    # shared causal SSM replacing the 5 independent TCN branches
    s4_layers: int = 2
    s4_state_dim: int = 8
    s4_d_ff: Optional[int] = 32         # None -> no feed-forward; 32 restores channel mix
    s4_dropout: float = 0.3

    # readout
    n_windows: int = 5
    fuse: str = "average"

    #: Apply ATCNet's official L2 kernel penalties (conv 0.009 / dense 0.5). Keep this True:
    #: with it off, `atcnet` vs `atcnet_s4` differs in the temporal stage AND in the
    #: dominant regularizer (the penalty is ~17x the cross-entropy at initialization).
    use_official_l2: bool = True


class ATCNetS4Classifier(nn.Module):
    """``ATCNetS4`` behind this project's ``{logits, features, confidence}`` contract.

    ``features`` is the mean of the per-readout vectors, matching what
    ``ATCNetClassifier`` reports so the two are directly comparable.

    Like ``ATCNetClassifier``, the official Keras ``L2`` kernel penalties are folded into the
    loss through the trainer's custom-loss hook — see ``ATCNetS4.l2_penalty`` for the
    layer-for-layer mapping and for why leaving them out would confound the ablation.
    """

    def __init__(self, config: ATCNetS4Config | None = None):
        super().__init__()
        cfg = config or ATCNetS4Config()
        self.config = cfg
        self.model = ATCNetS4(
            n_classes=cfg.n_classes,
            in_chans=cfg.n_channels,
            in_samples=cfg.n_times,
            n_windows=cfg.n_windows,
            eegn_F1=cfg.eegn_F1,
            eegn_D=cfg.eegn_D,
            eegn_kernelSize=cfg.eegn_kernelSize,
            eegn_poolSize=cfg.eegn_poolSize,
            eegn_dropout=cfg.eegn_dropout,
            attention=cfg.attention,
            s4_layers=cfg.s4_layers,
            s4_state_dim=cfg.s4_state_dim,
            s4_d_ff=cfg.s4_d_ff,
            s4_dropout=cfg.s4_dropout,
            fuse=cfg.fuse,
            use_official_l2=cfg.use_official_l2,
        )
        self.feature_dim = int(
            cfg.eegn_F1 * cfg.eegn_D * (cfg.n_windows if cfg.fuse == "concat" else 1)
        )
        self.uses_custom_loss = bool(cfg.use_official_l2)

    def default_lr(self) -> float:
        return DEFAULT_LR

    def describe(self) -> Dict[str, object]:
        d = {
            "name": "atcnet_s4",
            "source": "ours — ATCNet conv block + one shared causal diagonal SSM",
            "n_params": sum(p.numel() for p in self.parameters()),
            "feature_dim": self.feature_dim,
            "n_windows": int(self.model.n_windows),
            "Tc": int(self.model.Tc),
            "readout_idx": list(self.model.readout_idx),
            "attention": self.config.attention,
            "official_l2": bool(self.uses_custom_loss),
        }
        d.update(self.model.timescales())
        return d

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0,
                      subject_ids: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE + the official L2 kernel penalties, exactly as ``ATCNetClassifier`` does.

        ``subject_ids`` is accepted and ignored (the e2e trainer always passes the argument).
        """
        ce = F.cross_entropy(self.forward(x)["logits"], y)
        l2 = self.model.l2_penalty()
        return ce + l2, {"ce": float(ce.detach().item()),
                         "l2": float(l2.detach().item())}

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        logits = self.model(x)
        window_feats = self.model.last_window_features
        if self.config.fuse == "concat":
            features = torch.cat(window_feats, dim=1)
        else:
            features = (window_feats[0] if len(window_feats) == 1
                        else torch.stack(window_feats, dim=0).mean(dim=0))
        return {"logits": logits, "features": features, "confidence": None}


@dataclass
class ATCNetAdvConfig(ATCNetConfig):
    """ATCNet + subject-adversarial head. Structural defaults are ATCNet's own."""

    n_subjects: int = 11
    lambda_adv: float = 1.0
    adv_warmup_epochs: int = 5
    adv_ramp_epochs: int = 50      # epoch at which lambda reaches lambda_adv
    adv_hidden: int = 64
    #: Which tensor the discriminator sees. ``conv`` = the [B, Tc*F2] block that every
    #: sliding window reads (544-d at 58ch/1000pt); ``fused`` = the [B, tcn_filters]
    #: vector the dense heads consume (32-d). See the class docstring for why ``conv``.
    adv_tap: str = "conv"


class ATCNetAdversarial(ATCNetClassifier):
    """ATCNet with a gradient-reversed subject discriminator (DANN, Ganin 2016).

    Same backbone, same official L2 penalty, plus ``CE(subject)`` after a GRL so the
    encoder is pushed toward subject-invariant features.

    **Where the discriminator taps matters.** ``features`` is the *mean of the per-window
    TCN outputs* — it sits downstream of the window split, so it is not the tensor the
    windows share and it is only ``tcn_filters`` wide (32). The conv-block output is what
    every window actually reads, and it is ``Tc * F2`` wide (17*32 = 544). Subject identity
    lives in the spatio-temporal filters, i.e. exactly there, so ``adv_tap="conv"`` is the
    default; ``"fused"`` is kept as an ablation.
    """

    def __init__(self, config: ATCNetAdvConfig | None = None):
        cfg = config or ATCNetAdvConfig()
        super().__init__(cfg)
        self.adv_cfg = cfg

        if cfg.adv_tap not in ("conv", "fused"):
            raise ValueError(f"adv_tap must be 'conv' or 'fused', got {cfg.adv_tap!r}")
        self.adv_tap = cfg.adv_tap
        f2 = int(cfg.eegn_F1 * cfg.eegn_D)
        adv_in = int(self.model.Tc * f2) if cfg.adv_tap == "conv" else self.feature_dim

        self.grl = GradientReversalLayer(lambda_=0.0)
        self.subject_head = nn.Sequential(
            nn.Linear(adv_in, cfg.adv_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.eegn_dropout),
            nn.Linear(cfg.adv_hidden, cfg.n_subjects),
        )
        self.adv_in = adv_in
        # The adversarial branch is always part of the loss, so the plain-CE path in the
        # trainer would silently drop it; force the custom-loss hook regardless of L2.
        self.uses_custom_loss = True

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({
            "name": "atcnet_adv",
            "adv_tap": self.adv_tap,
            "adv_in_dim": self.adv_in,
            "n_subjects": int(self.adv_cfg.n_subjects),
            "lambda_adv": float(self.adv_cfg.lambda_adv),
            "adv_head_params": sum(p.numel() for p in self.subject_head.parameters()),
        })
        return d

    def update_adv_lambda(self, epoch: int) -> float:
        """Zero for ``adv_warmup_epochs``, then linear up to ``lambda_adv``.

        Matches the S4ERP-Adv schedule so the two adversarial arms are comparable.
        """
        cfg = self.adv_cfg
        if epoch < cfg.adv_warmup_epochs:
            lam = 0.0
        else:
            span = max(cfg.adv_ramp_epochs - cfg.adv_warmup_epochs, 1)
            lam = min(1.0, (epoch - cfg.adv_warmup_epochs) / span) * cfg.lambda_adv
        self.grl.set_lambda(float(lam))
        return float(lam)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        out = super().forward(x)
        if self.training:
            if self.adv_tap == "conv":
                conv = self.model.last_conv_features        # [B, Tc, F2]
                z = conv.reshape(conv.size(0), -1)
            else:
                z = out["features"]
            out["subject_logits"] = self.subject_head(self.grl(z))
        return out

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0,
                      subject_ids: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``CE + official L2 + CE(subject)``; adversarial strength lives in the GRL."""
        self.update_adv_lambda(int(epoch))
        out = self.forward(x)
        ce = F.cross_entropy(out["logits"], y)
        l2 = self.model.l2_penalty()
        loss = ce + l2
        parts = {"ce": float(ce.detach().item()), "l2": float(l2.detach().item())}
        if subject_ids is not None and "subject_logits" in out:
            adv = F.cross_entropy(out["subject_logits"], subject_ids)
            loss = loss + adv
            parts["adv"] = float(adv.detach().item())
            parts["lambda"] = float(self.grl.lambda_)
        return loss, parts

"""ATCNet with its five independent TCN branches replaced by one **shared causal SSM**.

Fusion direction B: ATCNet is the base, its temporal-aggregation stage is swapped for the
state-space layer used by S4ERP. ``atcnet_torch.py`` (the verbatim 1:1 port of the official
Keras ``ATCNet_``) is deliberately **not** touched — its value is being a faithful
reference — so everything new lives here and only reuses its building blocks.

Why this is the right swap
--------------------------
Measured/derived properties of the official ATCNet at 1000 samples @ 250 Hz:

* the convolutional block emits ``Tc = (1000 // 8) // 7 = 17`` tokens of width ``F2 = 32``;
* one token covers 239 samples ≈ **956 ms** with a stride of 56 samples ≈ **224 ms**;
* a sliding window of ``Tw = Tc - n_windows + 1 = 13`` tokens therefore spans
  **≈ 3,644 ms, i.e. 91% of the 4 s trial**;
* window ``i`` is read out at its *last* step, which is index ``i + Tw - 1`` of the original
  sequence — so with ``n_windows=5`` the five readouts are indices **12, 13, 14, 15, 16**,
  the last five tokens;
* the TCN's receptive field is ``1 + 2*(k-1)*(1 + 2) = 19 > Tw``, so that last step already
  integrates the entire window.

Conclusion: **the five windows are five 224 ms-shifted copies of nearly the same view, i.e.
an ensemble, not a multi-scale decomposition.** Yet ATCNet spends five *independent* sets of
TCN weights on them — 83,840 parameters, **73.1% of the whole model** — to implement what
should be a shift-equivariant operator.

A single causal SSM resolves the tension: weight sharing costs 1/5 the parameters, causality
guarantees that the readout at position ``t`` sees only ``[0, t]`` (so the five readouts still
differ, which is what keeps the ensemble diverse), and each readout's receptive field is
unbounded rather than clipped at 13 steps. The five independent ``Linear`` heads are kept —
they are 330 parameters and they are where the ensemble actually lives.

Budget at 58 channels / 2 classes (measured after the FFN mixer)::

    ConvBlock                     19,424   (unchanged)
    shared MHA (optional)          ~2,200   was 10,960 for 5 independent blocks
    shared SSM + FFN (2 layers)    ~6,000   was 83,840 for 5 independent TCNs
    5 x Linear(32, 2)                 330   (unchanged)
    total                         ~28,000   vs ATCNet's 114,554  ->  ~4x smaller

The SSM is the project's ``S4Block``. Note that implementation is a **diagonal** SSM (it
keeps only ``diag(A)`` of the HiPPO-LegS matrix), not the DPLR construction of Gu et al.
2022 — see ``code/models/eeg_foundation/s4_layers.py`` and the naming discussion in
``docs/PLAN_cross_subject_benchmark_and_fusion.md`` §5.5.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

from ..eeg_foundation.s4_layers import S4Block
from .atcnet_torch import (
    CONV_MAX_NORM,
    CONV_WEIGHT_DECAY,
    DENSE_WEIGHT_DECAY,
    _ConvBlock,
    _KerasMHA,
    _MaxNormKernel,
)


def reinit_log_dt_for_length(s4_layer: nn.Module, seq_len: int) -> None:
    """Re-draw an ``S4Layer``'s ``log_dt`` so its timescales match a short sequence.

    **This is not optional.** ``S4Layer.__init__`` uses
    ``log_dt ~ U(log(0.1) - 4, -4)``, i.e. ``dt in [0.0018, 0.018]``. The discretized decay
    is ``A_bar[h, n] = exp(-(n + 1) * dt_h)``, so the time constant of mode ``n`` is

    .. math:: \\tau(n, dt) = \\frac{1}{(n + 1)\\, dt}\\quad\\text{steps}

    which for the default range spans **6.8 to 546 steps**. That is calibrated for S4ERP's
    ``patch_num = 487``. On ATCNet's ``Tc = 17`` tokens, 7 of the 8 modes have
    ``tau >> L`` and their kernels are flat — the layer degenerates into a running mean, and
    the resulting null result would be misread as "the SSM does not work here".

    Rescaling: to cover time constants from ``1`` step up to the sequence length ``L`` while
    the ``(n + 1)`` factor already spans ``1..N``, sample ``dt`` log-uniformly in
    ``[1 / L, N / L]``. Then ``tau`` ranges from ``L / N**2`` (fastest mode, largest ``dt``)
    to ``L`` (slowest mode, smallest ``dt``) — the whole window, and nothing wasted beyond it.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    n_states = int(s4_layer.state_dim)
    dt_min = 1.0 / float(seq_len)
    dt_max = float(n_states) / float(seq_len)
    with torch.no_grad():
        u = torch.rand_like(s4_layer.log_dt)
        s4_layer.log_dt.copy_(math.log(dt_min) + u * (math.log(dt_max) - math.log(dt_min)))


class _CausalKerasMHA(_KerasMHA):
    """``_KerasMHA`` with a causal mask.

    Without the mask, attention at readout position 12 would see position 16 and all five
    readouts would collapse onto the same global view, destroying the ensemble diversity
    that is the entire point of keeping five heads. (This is not train/test leakage — it is
    all within one trial — but it does defeat the architecture.)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.key_dim).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.key_dim)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.key_dim)
        return self.out_proj(out)


class _CausalMHABlock(nn.Module):
    """Pre-norm residual block around ``_CausalKerasMHA`` (mirrors ATCNet's ``mha_block``)."""

    def __init__(self, embed_dim: int, key_dim: int = 8, num_heads: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha = _CausalKerasMHA(embed_dim, key_dim=key_dim, num_heads=num_heads,
                                   dropout=dropout)
        self.drop = nn.Dropout(0.3)      # hard-coded upstream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mha(self.norm(x)))


class ATCNetS4(nn.Module):
    """ATCNet's conv block + one shared causal SSM + ``n_windows`` independent readouts.

    ``forward`` returns pre-softmax logits, matching ``ATCNetOfficial``.

    Parameters mirror ``ATCNetOfficial`` where they play the same role, so a config can be
    copied across. ``tcn_*`` are replaced by ``s4_*``.

    Parameters
    ----------
    n_windows : int
        Number of readout positions. They are the **last** ``n_windows`` tokens, which is
        exactly where the official model reads its sliding windows.
    attention : {"mha", None}
        ``"mha"`` inserts one shared causal attention block before the SSM (ATCNet has five
        independent, unmasked ones). ``None`` drops it, which makes the comparison against
        ``atcnet@noattn`` a single-variable test of "5 independent TCNs vs 1 shared SSM".
    s4_layers, s4_state_dim, s4_d_ff, s4_dropout
        The shared SSM stack. ``s4_d_ff=32`` is a pointwise channel mixer: a diagonal
        SSM is SISO per feature, while ATCNet's ``Conv1d(32, 32)`` mixes channels.
        ``None`` drops the mixer and makes the ``@noattn`` pair a two-variable change.
    use_official_l2 : bool
        Apply ATCNet's ``L2`` kernel penalties (see ``l2_penalty``). Default True, and it
        matters: dropping it would make the comparison against ``atcnet`` a two-variable
        change (temporal stage *and* regularization).
    """

    def __init__(
        self,
        n_classes: int,
        in_chans: int = 58,
        in_samples: int = 1000,
        n_windows: int = 5,
        eegn_F1: int = 16,
        eegn_D: int = 2,
        eegn_kernelSize: int = 64,
        eegn_poolSize: int = 7,
        eegn_dropout: float = 0.3,
        attention: Optional[str] = "mha",
        s4_layers: int = 2,
        s4_state_dim: int = 8,
        s4_d_ff: Optional[int] = 32,
        s4_dropout: float = 0.3,
        fuse: str = "average",
        use_official_l2: bool = True,
    ):
        super().__init__()
        self.use_official_l2 = bool(use_official_l2)
        if attention not in (None, "mha"):
            raise ValueError(f"attention must be 'mha' or None, got {attention!r}")
        if fuse not in ("average", "concat"):
            raise ValueError(f"fuse must be 'average' or 'concat', got {fuse!r}")

        F2 = eegn_F1 * eegn_D
        self.F2 = F2
        self.n_windows = int(n_windows)
        self.fuse = fuse

        self.conv_block = _ConvBlock(
            in_chans=in_chans, F1=eegn_F1, D=eegn_D, kern_length=eegn_kernelSize,
            pool_size=eegn_poolSize, dropout=eegn_dropout,
        )
        self.Tc = (in_samples // 8) // eegn_poolSize
        if self.Tc < self.n_windows:
            raise ValueError(
                f"n_windows={n_windows} needs at least that many tokens after the conv "
                f"block, but Tc={self.Tc} (in_samples={in_samples})."
            )
        #: Readout positions: the last `n_windows` tokens = the official model's window ends.
        self.readout_idx: List[int] = list(range(self.Tc - self.n_windows, self.Tc))

        self.attention_block = (_CausalMHABlock(F2) if attention == "mha" else None)

        # One shared stack, applied to the whole Tc-token sequence.
        self.s4 = nn.Sequential(*[
            S4Block(F2, state_dim=s4_state_dim, d_ff=s4_d_ff, dropout=s4_dropout)
            for _ in range(int(s4_layers))
        ])
        for block in self.s4:
            reinit_log_dt_for_length(block.s4, self.Tc)
            # TCN kernels carry max_norm(0.6); B/C are the linear maps that replaced them.
            register_parametrization(block.s4, "B", _MaxNormKernel(CONV_MAX_NORM))
            register_parametrization(block.s4, "C", _MaxNormKernel(CONV_MAX_NORM))

        if fuse == "average":
            self.dense = nn.ModuleList(
                [nn.Linear(F2, n_classes) for _ in range(self.n_windows)]
            )
        else:
            self.dense = nn.ModuleList([nn.Linear(F2 * self.n_windows, n_classes)])
        for lin in self.dense:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"expected [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1).unsqueeze(1)                  # [B, 1, T, C]

        feat = self.conv_block(x).squeeze(-1)                # [B, F2, Tc]
        feat = feat.permute(0, 2, 1)                         # [B, Tc, F2]
        self.last_conv_features = feat

        if self.attention_block is not None:
            feat = self.attention_block(feat)
        feat = self.s4(feat)                                 # [B, Tc, F2], causal

        window_feats = [feat[:, t, :] for t in self.readout_idx]
        self.last_window_features = window_feats

        if self.fuse == "average":
            outs = [self.dense[i](f) for i, f in enumerate(window_feats)]
            return (outs[0] if len(outs) == 1
                    else torch.stack(outs, dim=0).mean(dim=0))
        return self.dense[0](torch.cat(window_feats, dim=1))

    def l2_penalty(self) -> torch.Tensor:
        """ATCNet's ``L2`` kernel penalties, applied to the corresponding layers here.

        **Not optional for a fair comparison.** Upstream ATCNet folds
        ``L2(conv_weightDecay=0.009)`` over its conv kernels and ``L2(dense_weightDecay=0.5)``
        over its classifier weights into the loss. At initialization that term measures
        ~12.2 against a cross-entropy of ~0.69 — it is **17x the data term**, not a nudge.
        A variant without it would differ from ``atcnet`` in two ways at once (temporal stage
        *and* the dominant regularizer), and the ablation would be uninterpretable.

        The mapping is layer-for-layer:

        * conv block kernels -> ``0.009``. Identical module to ATCNet's, identical treatment.
        * dense head weights -> ``0.5``. Identical modules (``n_windows`` x ``Linear``).
        * SSM ``B``/``C`` -> ``0.009``, standing in for the TCN conv kernels they replace.
          ``A_log`` / ``A_sign`` / ``log_dt`` / ``D`` are deliberately **excluded**:
          the first three parameterize the state dynamics (decay rates and timestep),
          not a linear map — shrinking ``log_dt`` toward 0 would push ``dt`` toward 1
          and destroy the timescale calibration that ``reinit_log_dt_for_length`` exists
          to set. ``D`` is the SSM skip (initialized at 1). At init the convolution
          kernel is ~1e-4, so ``D`` is the only live temporal path; penalizing it would
          drive the skip to 0 and collapse the stage to (near) identity. ATCNet's TCN
          residual is an unscaled add, which has no analogue of a penalized skip gain.
        * the attention block is excluded, matching upstream (only conv + dense are penalized).
        """
        conv_term = sum((w ** 2).sum() for w in self.conv_block.conv_kernels())
        for block in self.s4:
            conv_term = conv_term + (block.s4.B ** 2).sum() + (block.s4.C ** 2).sum()
            if block.has_ffn:
                conv_term = conv_term + sum(
                    (mod.weight ** 2).sum()
                    for mod in block.ffn.modules() if isinstance(mod, nn.Linear)
                )
        dense_term = sum((lin.weight ** 2).sum() for lin in self.dense)
        return CONV_WEIGHT_DECAY * conv_term + DENSE_WEIGHT_DECAY * dense_term

    def timescales(self) -> Dict[str, float]:
        """Min/max SSM time constants in tokens — the P1 smoke check for ``log_dt``.

        Must land in the 1..Tc range, not the 6.8..546 range of the S4ERP default.
        """
        taus: List[float] = []
        for block in self.s4:
            dt = torch.exp(block.s4.log_dt).detach()
            n = torch.arange(1, block.s4.state_dim + 1, dtype=dt.dtype, device=dt.device)
            taus += (1.0 / (n[None, :] * dt[:, None])).flatten().tolist()
        return {"tau_min_tokens": float(min(taus)), "tau_max_tokens": float(max(taus)),
                "Tc": float(self.Tc)}

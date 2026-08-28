#!/usr/bin/env python3
"""10 Hz SSVEP Oz Welch PSD — Wang 2016 benchmark dataset.

Data: Wang et al., IEEE TNSRE 2017 (doi:10.1109/TNSRE.2016.2627556)
Each .mat is [64 ch × 1500 samp × 40 targets × 6 blocks], 250 Hz.
Epoch layout: −0.5 s cue, 0–5 s flicker, 5–5.5 s blank.
10 Hz is target index 2 (0-based) in the 5×8 speller layout used by MOABB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.ticker import MultipleLocator
from mne.preprocessing import ICA
from scipy.io import loadmat
from scipy.signal import welch

# International 10-20 labels in Wang2016 / 64-channels.loc order (1-based 1..64)
CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5",
    "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3",
    "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2",
]

# MOABB Wang2016 event map: target index 0-based; "10" Hz -> event id 3 -> index 2
# Speller is 5 rows × 8 cols: 8,9,10,...,15 then 8.2,9.2,... (not 8.0,8.2,8.4,...)
FREQ_HZ_BY_TARGET = np.array(
    [
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
        8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
        8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
        8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
    ]
)

SFREQ = 250.0
N_PRE_STIM = 125  # 0.5 s cue / baseline, discarded
N_STIM = 1250  # 5.0 s flicker
STIM_HZ = 10.0
N_FFT = 1024
N_OVERLAP = 256


def find_10hz_target_index(data: np.ndarray, oz_idx: int = 61) -> int:
    """SNR at 10 Hz (peak / neighbor median), used only as a sanity check."""
    n_targets = data.shape[2]
    scores = np.zeros(n_targets)
    for t in range(n_targets):
        trials = data[oz_idx, N_PRE_STIM : N_PRE_STIM + N_STIM, t, :]
        specs = []
        for b in range(trials.shape[1]):
            f, pxx = welch(
                trials[:, b],
                fs=SFREQ,
                nperseg=N_FFT,
                noverlap=N_OVERLAP,
                nfft=N_FFT,
            )
            specs.append(pxx)
        mean_p = np.mean(specs, axis=0)
        i10 = int(np.argmin(np.abs(f - STIM_HZ)))
        neigh = (np.abs(f - STIM_HZ) > 1.0) & (np.abs(f - STIM_HZ) < 3.0)
        noise = np.median(mean_p[neigh]) if np.any(neigh) else np.median(mean_p)
        scores[t] = mean_p[i10] / max(noise, np.finfo(float).tiny)
    best = int(np.argmax(scores))
    print(f"[freq-check] top-5 10 Hz SNR targets: {np.argsort(scores)[::-1][:5].tolist()}")
    return best


def load_freq_table(data_dir: Path) -> np.ndarray:
    fp = data_dir / "Freq_Phase.mat"
    if fp.exists():
        mat = loadmat(str(fp), squeeze_me=True)
        freqs = np.asarray(mat["freqs"], dtype=float).ravel()
        print(f"[Freq_Phase] {freqs.tolist()}")
        return freqs
    return FREQ_HZ_BY_TARGET.copy()


def load_10hz_epochs_array(mat_paths: list[Path], freq_table: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return full 6 s epochs in volts (includes 0.5 s pre-stim for filter padding)."""
    oz_idx = CH_NAMES.index("Oz")
    labeled = int(np.argmin(np.abs(freq_table - STIM_HZ)))
    chunks = []
    chosen = None
    for path in mat_paths:
        mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
        if "data" not in mat:
            raise KeyError(f"{path} has no 'data' key; keys={list(mat)}")
        data = np.asarray(mat["data"], dtype=np.float64)
        if data.ndim != 4 or data.shape[0] != 64 or data.shape[1] != 1500:
            raise ValueError(f"{path}: unexpected data shape {data.shape}")
        if chosen is None:
            snr_best = find_10hz_target_index(data, oz_idx=oz_idx)
            print(
                f"[freq-check] Freq_Phase 10 Hz index={labeled} ({freq_table[labeled]} Hz); "
                f"SNR-argmax index={snr_best} ({freq_table[snr_best]} Hz)"
            )
            # Ground truth is the experimental label, not the SNR argmax.
            chosen = labeled
        # keep full 6 s so FIR/ICA have pre/post padding; crop to flicker after filter
        full = np.transpose(data[:, :, chosen, :], (2, 0, 1))  # (blocks, ch, time)
        chunks.append(full)
        print(f"[load] {path.name}: {full.shape[0]} trials of 10 Hz")
    arr_uv = np.concatenate(chunks, axis=0)
    arr_v = arr_uv * 1e-6  # Neuroscan export is µV
    return arr_v, chosen, oz_idx


def make_epochs(arr_v: np.ndarray) -> mne.EpochsArray:
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    # File time 0 is 0.5 s before flicker onset.
    epochs = mne.EpochsArray(arr_v, info, tmin=-0.5, verbose="error")
    epochs.set_montage(montage, on_missing="ignore")
    epochs.set_eeg_reference("average", projection=False, verbose="error")
    return epochs


def preprocess(epochs: mne.EpochsArray) -> mne.EpochsArray:
    """Average ref, 1–45 Hz FIR, ICA EOG, crop to flicker, ±80 µV reject.

    Recording already applied a 50 Hz notch. A second FIR notch on 6 s epochs
    is longer than the signal and distorts the spectrum, so it is omitted.
    1 Hz high-pass (not 5 Hz) is used so the 0–8 Hz 1/f floor remains visible
    on a 0–45 Hz axis; 45 Hz low-pass still removes EMG / line residuals.
    """
    n_in = len(epochs)
    epochs.filter(1.0, 45.0, fir_design="firwin", verbose="error")

    ica = ICA(n_components=20, method="fastica", random_state=97, max_iter="auto")
    ica.fit(epochs, verbose="error")
    try:
        eog_indices, eog_scores = ica.find_bads_eog(
            epochs, ch_name=["Fp1", "Fp2"], verbose="error"
        )
        ica.exclude = list(eog_indices)
        score_max = float(np.max(np.abs(eog_scores))) if np.size(eog_scores) else 0.0
        print(f"[ICA] excluded EOG components: {ica.exclude} (score max={score_max:.3f})")
    except Exception as exc:
        print(f"[ICA] EOG detection skipped ({exc})")
    epochs = ica.apply(epochs, verbose="error")

    # Keep only the 5 s flicker, drop the 0.5 s pre-stim cue/baseline.
    epochs = epochs.copy().crop(tmin=0.0, tmax=5.0)
    epochs.drop_bad(reject=dict(eeg=80e-6), verbose="error")
    n_good = len(epochs)
    print(f"[reject] kept {n_good} / {n_in} epochs (|amp| < 80 µV)")
    if n_good < 6:
        raise RuntimeError(f"Too few epochs after reject: {n_good}")
    return epochs


def compute_oz_psd(epochs: mne.EpochsArray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=45.0,
        n_fft=N_FFT,
        n_overlap=N_OVERLAP,
        picks=["Oz"],
        verbose="error",
    )
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, 1, n_freqs), V²/Hz
    psd_mean_v = psds.mean(axis=0)[0]
    psd_mean_uv = psd_mean_v * 1e12  # → µV²/Hz
    psd_db = 10.0 * np.log10(np.maximum(psd_mean_uv, np.finfo(float).tiny))
    return freqs, psd_db


def plot_psd(freqs: np.ndarray, psd_db: np.ndarray, n_epochs: int, out_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)

    ax.plot(
        freqs,
        psd_db,
        color="#1f3a93",
        lw=1.8,
        solid_capstyle="round",
        label="Oz channel, 10-Hz SSVEP",
    )
    for f0, lab in ((10, "10 Hz"), (20, "20 Hz"), (30, "30 Hz")):
        ax.axvline(f0, color="0.65", linestyle="--", lw=1.0, zorder=0)
        i = int(np.argmin(np.abs(freqs - f0)))
        ax.scatter(
            [freqs[i]],
            [psd_db[i]],
            s=28,
            c="#c0392b",
            zorder=3,
            edgecolors="none",
        )

    ax.set_xlim(0, 45)
    ax.set_xticks(np.arange(0, 46, 5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Power (dB)")  # 10 log10(µV²/Hz)

    y_min, y_max = float(np.min(psd_db)), float(np.max(psd_db))
    pad = max(3.0, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.4", fontsize=10)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.text(
        0.02,
        0.04,
        f"n = {n_epochs} epochs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout()
    extras = [
        Path("/share/home/yuan/SYX/10Hz_SSVEP_Oz_PSD.png"),
        Path("/share/home/yuan/SYX/ssvep_10hz/10Hz_SSVEP_Oz_PSD.png"),
    ]
    save_paths = [out_png, *extras]
    for path in save_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"[save] {path}")
    print(
        f"[peaks] 10 Hz={psd_db[np.argmin(np.abs(freqs-10))]:.2f} dB, "
        f"20 Hz={psd_db[np.argmin(np.abs(freqs-20))]:.2f} dB, "
        f"30 Hz={psd_db[np.argmin(np.abs(freqs-30))]:.2f} dB"
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/share/home/yuan/SYX/ssvep_10hz/data"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/share/home/yuan/SYX/eeg-mi-online/10Hz_SSVEP_Oz_PSD.png"),
    )
    args = parser.parse_args()

    mat_paths = sorted(args.data_dir.glob("S*.mat"))
    mat_paths = [p for p in mat_paths if p.stat().st_size > 50_000_000]
    if not mat_paths:
        raise FileNotFoundError(
            f"No complete S*.mat (>50 MB) in {args.data_dir}. "
            "Download Wang2016 from https://zenodo.org/records/14865172"
        )
    print(f"[data] using {[p.name for p in mat_paths]}")

    freq_table = load_freq_table(args.data_dir)
    arr_v, target_idx, oz_idx = load_10hz_epochs_array(mat_paths, freq_table)
    print(f"[data] stacked {arr_v.shape[0]} trials, Oz ch index={oz_idx}, target={target_idx}")
    epochs = make_epochs(arr_v)
    epochs = preprocess(epochs)
    freqs, psd_db = compute_oz_psd(epochs)
    plot_psd(freqs, psd_db, n_epochs=len(epochs), out_png=args.out)


if __name__ == "__main__":
    mne.set_log_level("ERROR")
    main()

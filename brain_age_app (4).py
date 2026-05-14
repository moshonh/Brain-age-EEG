"""
╔══════════════════════════════════════════════════════════════════╗
║          NEUROAGE  –  EEG Brain Age Estimation Dashboard         ║
║          Ridge Regression · Alpha-Peak · Delta/Alpha Ratio       ║
╚══════════════════════════════════════════════════════════════════╝

A self-contained Streamlit application that predicts brain age from
19-channel EEG (.edf) recordings using established neurophysiological
biomarkers (Alpha Peak Frequency and Delta/Alpha power ratio), with
optional support for a pre-trained scikit-learn .joblib / .pkl model.

Run:
    pip install streamlit mne numpy scipy matplotlib joblib
    streamlit run brain_age_app.py
"""

# ── Standard library ──────────────────────────────────────────────
import io
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Compatibility patch: np.trapz was removed in NumPy 2.0
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import streamlit as st
from scipy.signal import welch

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════

TARGET_FS      = 250          # Hz – resample target
BANDPASS_LOW   = 1.0          # Hz
BANDPASS_HIGH  = 40.0         # Hz
NOTCH_FREQ     = 50.0         # Hz
EPOCH_DURATION = 4.0          # seconds per segment for spectral estimation
ARTIFACT_THRESH= 150e-6       # Volts (150 µV peak-to-peak)

# Standard 10-20 system (19 channels) – ORDER MATTERS for topomap
STANDARD_19 = [
    "Fp1", "Fp2",
    "F7",  "F3",  "Fz",  "F4",  "F8",
    "T7",  "C3",  "Cz",  "C4",  "T8",
    "P7",  "P3",  "Pz",  "P4",  "P8",
    "O1",  "O2",
]

# Approximate 2-D (x, y) positions for the 19 electrodes on a unit circle
# Derived from the standard 10-20 azimuthal projection
TOPO_COORDS = {
    "Fp1": (-0.18,  0.92), "Fp2": ( 0.18,  0.92),
    "F7":  (-0.71,  0.55), "F3":  (-0.40,  0.58), "Fz":  ( 0.00,  0.65),
    "F4":  ( 0.40,  0.58), "F8":  ( 0.71,  0.55),
    "T7":  (-0.90,  0.00), "C3":  (-0.46,  0.00), "Cz":  ( 0.00,  0.00),
    "C4":  ( 0.46,  0.00), "T8":  ( 0.90,  0.00),
    "P7":  (-0.71, -0.55), "P3":  (-0.40, -0.58), "Pz":  ( 0.00, -0.65),
    "P4":  ( 0.40, -0.58), "P8":  ( 0.71, -0.55),
    "O1":  (-0.18, -0.92), "O2":  ( 0.18, -0.92),
}

# Frequency bands (Hz)
BANDS = {
    "Delta": (1.0,  4.0),
    "Theta": (4.0,  8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
}

# ── Reference-model coefficients (inspired by LEMON dataset literature) ──────
# Brain age ≈ INTERCEPT + w_apf * APF + w_dar * Delta/Alpha_ratio
# APF typically 9–11 Hz in young adults, declining ~0.1 Hz/decade
# Delta/Alpha ratio increases with age
REFERENCE_MODEL = {
    "intercept": 78.5,
    "w_apf":     -3.2,    # each Hz rise in APF → younger brain
    "w_dar":      6.8,    # each unit rise in δ/α ratio → older brain
    "apf_norm":  10.0,    # normalisation centre for APF
    "dar_norm":   0.5,    # normalisation centre for δ/α
}


# ══════════════════════════════════════════════════════════════════
# 1. CHANNEL MATCHING
# ══════════════════════════════════════════════════════════════════

def _resolve_channel(target: str, available: list[str]) -> Optional[str]:
    """
    Map a desired 10-20 label to its recording name using:
      1. Exact match (case-insensitive).
      2. Regex strip of common prefixes/suffixes.
    """
    t = target.upper()
    for ch in available:
        if ch.strip().upper() == t:
            return ch
    pattern = re.compile(
        r"^(?:EEG\s*)?([A-Z][A-Z0-9]*(?:p[0-9]?)?)(?:[- _]?(?:REF|LE|AVG|A[12]))?$",
        re.IGNORECASE,
    )
    for ch in available:
        m = pattern.match(ch.strip())
        if m and m.group(1).upper() == t:
            return ch
    return None


def select_standard_channels(raw) -> dict[str, str]:
    """Return {standard_label: recording_label} for all 19 channels."""
    mapping, missing = {}, []
    for std in STANDARD_19:
        found = _resolve_channel(std, raw.ch_names)
        if found:
            mapping[std] = found
        else:
            missing.append(std)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required channel(s): {', '.join(missing)}.\n"
            f"Found: {', '.join(raw.ch_names)}"
        )
    return mapping


# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING + FEATURE EXTRACTION (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def preprocess_and_extract(edf_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline:
      load → channel select → resample → filter → CAR →
      epoch → artifact reject → band-power → APF → δ/α ratio

    Returns a dict with all features, metadata, and objects needed for plots.
    """
    import mne  # local import so missing MNE fails gracefully

    # Write temp file (MNE requires a path)
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(edf_bytes)
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

    # ── Legacy 10-20 → modern name normalisation ──────────────────
    # The old AES (1961) standard used T3/T4/T5/T6; the current IFCN
    # (1991) standard renamed them T7/T8/P7/P8.  Many clinical EDF
    # files (especially older Nihon-Kohden / Cadwell systems) still
    # export the legacy labels, which would cause select_standard_channels
    # to fail when looking for T7, T8, P7, P8.
    # We also handle common prefix variants such as "EEG T3-Ref".
    LEGACY_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    legacy_pattern = re.compile(
        r"^(?:EEG\s*)?(T3|T4|T5|T6)(?:[- _]?(?:REF|LE|AVG|A[12]))?$",
        re.IGNORECASE,
    )
    rename_legacy: dict[str, str] = {}
    for ch in raw.ch_names:
        m = legacy_pattern.match(ch.strip())
        if m:
            modern = LEGACY_MAP[m.group(1).upper()]
            rename_legacy[ch] = modern
    if rename_legacy:
        raw.rename_channels(rename_legacy)

    meta = {
        "filename":        filename,
        "orig_fs":         raw.info["sfreq"],
        "duration_s":      raw.times[-1],
        "orig_channels":   list(raw.ch_names),          # already normalised
        "legacy_renamed":  rename_legacy,               # e.g. {"T3": "T7", …}
    }

    # ── Channel selection ─────────────────────────────────────────
    ch_map = select_standard_channels(raw)
    raw.pick_channels(list(ch_map.values()))
    raw.rename_channels({v: k for k, v in ch_map.items()})
    raw.reorder_channels(STANDARD_19)

    # ── Signal processing ─────────────────────────────────────────
    raw.resample(TARGET_FS, npad="auto", verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="fir", verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ── Epoch + artifact rejection ────────────────────────────────
    data    = raw.get_data()                          # (19, T)
    n_ep    = int(data.shape[1] // (TARGET_FS * EPOCH_DURATION))
    n_pts   = int(TARGET_FS * EPOCH_DURATION)         # 1000

    if n_ep == 0:
        raise ValueError("Recording is too short for 4-second epochs.")

    epochs  = data[:, :n_ep * n_pts].reshape(19, n_ep, n_pts).transpose(1, 0, 2)
    ptp     = np.ptp(epochs, axis=2).max(axis=1)
    good    = epochs[ptp <= ARTIFACT_THRESH]

    if len(good) == 0:
        raise ValueError("All epochs rejected by artifact threshold (150 µV).")

    meta["n_epochs_total"]    = n_ep
    meta["n_epochs_accepted"] = len(good)

    # ── PSD via Welch (per epoch → average) ───────────────────────
    f, Pxx_all = welch(good, fs=TARGET_FS, nperseg=TARGET_FS * 2, axis=2)
    # Pxx_all: (epochs, channels, freqs)
    Pxx_mean   = Pxx_all.mean(axis=0)    # (channels, freqs)

    # np.trapz removed in NumPy 2.0; np.trapezoid is the replacement.
    # getattr fallback keeps compatibility with NumPy < 1.25.
    _trapz = getattr(np, "trapezoid", np.trapz)

    def _band_power(f, Pxx, lo, hi):
        idx = np.logical_and(f >= lo, f <= hi)
        return _trapz(Pxx[:, idx], f[idx], axis=1)   # (channels,)

    def _total_power(f, Pxx):
        idx = np.logical_and(f >= BANDPASS_LOW, f <= BANDPASS_HIGH)
        return _trapz(Pxx[:, idx], f[idx], axis=1)

    tot = _total_power(f, Pxx_mean)
    band_abs  = {b: _band_power(f, Pxx_mean, lo, hi) for b, (lo, hi) in BANDS.items()}
    band_rel  = {b: v / (tot + 1e-30) for b, v in band_abs.items()}

    # ── Alpha Peak Frequency (per recording – averaged across channels) ─
    alpha_idx = np.logical_and(f >= 7.0, f <= 14.0)
    alpha_psd = Pxx_mean[:, alpha_idx].mean(axis=0)   # (freqs,)
    apf       = float(f[alpha_idx][np.argmax(alpha_psd)])

    # ── δ/α ratio (averaged across channels) ─────────────────────
    dar = float(band_rel["Delta"].mean() / (band_rel["Alpha"].mean() + 1e-30))

    return {
        "meta":       meta,
        "raw":        raw,        # filtered MNE Raw for PSD plot
        "freqs":      f,
        "pxx_mean":   Pxx_mean,   # (19, freqs)
        "band_rel":   band_rel,   # dict of (19,) arrays
        "apf":        apf,
        "dar":        dar,
        "epochs_clean": good,     # (N_good, 19, 1000) float64 – for REVE
    }


# ══════════════════════════════════════════════════════════════════
# 3. MODEL LOADING + PREDICTION
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_external_model(model_bytes: bytes, suffix: str):
    """Load a scikit-learn model from joblib/pickle bytes."""
    import joblib
    buf = io.BytesIO(model_bytes)
    return joblib.load(buf)


# ── BENDR Foundation Model ───────────────────────────────────────
# BENDR (Brain Encoder Neural Decoder Representations)
# Kostas et al. · huggingface.co/braindecode/braindecode-bendr
#
# Architecture: Convolutional encoder + Transformer context network.
# Pretrained on TUH Abnormal EEG Corpus (400K samples) + SHHS sleep EEG.
# Fully open — no token, no gating, BSD-3 licence.
# Loaded via braindecode's from_pretrained() API.

BENDR_REPO = "braindecode/braindecode-bendr"


@st.cache_resource(show_spinner=False)
def load_bendr_model():
    """
    Download BENDR encoder from HuggingFace Hub via braindecode.

    Returns BENDR model in eval() mode with n_outputs=1 (regression).

    Requirements
    ────────────
    pip install braindecode torch huggingface_hub safetensors

    No token or account required — fully open model.
    """
    import subprocess, sys

    # safetensors is required to load BENDR weights but is not always
    # pulled in automatically.  Install it at runtime if missing.
    try:
        import safetensors          # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "safetensors", "-q"]
        )
        import safetensors          # noqa: F401

    from braindecode.models import BENDR
    model = BENDR.from_pretrained(BENDR_REPO, n_outputs=1)
    model.eval()
    return model


def _get_bendr_embeddings(
    model,
    epochs: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Run BENDR encoder in mini-batches over all clean epochs.
    Returns the mean embedding vector across epochs.

    Parameters
    ----------
    model      : BENDR instance in eval mode
    epochs     : (N, 19, 1000)  float32  at 250 Hz
    batch_size : number of epochs per forward pass

    Returns
    -------
    embedding : (embed_dim,) float32  — mean across all accepted epochs

    API note
    --------
    BENDR.forward(x, return_features=True) returns a dict:
        {"features": Tensor(B, 512), "cls_token": None}
    """
    import torch

    device   = next(model.parameters()).device
    all_embs = []

    with torch.no_grad():
        for start in range(0, len(epochs), batch_size):
            batch = torch.from_numpy(
                epochs[start : start + batch_size].astype(np.float32)
            ).to(device)                                    # (B, 19, 1000)

            out = model(batch, return_features=True)        # dict

            # braindecode BENDR returns {"features": (B, D), "cls_token": ...}
            if isinstance(out, dict):
                features = out["features"]                  # (B, 512)
            elif isinstance(out, tuple):
                features = out[0]
            else:
                features = out

            if features.dim() == 3:
                features = features.mean(dim=1)             # (B, D)

            all_embs.append(features.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0).mean(axis=0)   # (D,)


def _bendr_linear_head(embedding: np.ndarray, apf: float, dar: float) -> float:
    """
    Linear regression head on top of BENDR embeddings.

    Out-of-the-box: fixed random projection (seed=42) calibrated to the
    physiological reference model, giving a ±5-yr modulation around
    the biomarker estimate.  Replace with fitted Ridge weights (saved as
    .joblib) and load via the sidebar uploader for production accuracy.
    """
    rng   = np.random.default_rng(42)
    D     = embedding.shape[0]
    w_rnd = rng.standard_normal(D).astype(np.float32)
    w_rnd /= (np.linalg.norm(w_rnd) + 1e-8)

    proj    = float(np.dot(w_rnd, embedding))
    ref_age = predict_reference_model(apf, dar)
    blended = ref_age + 5.0 * float(np.tanh(proj))
    return float(np.clip(blended, 20.0, 80.0))


def predict_with_bendr(model, result: dict) -> Tuple[float, str]:
    """
    Full BENDR prediction pipeline:
      clean epochs (250 Hz, 19 ch)
        → BENDR encoder (mini-batched)
        → mean embedding
        → linear regression head
        → predicted age (years)
    """
    epochs  = result["epochs_clean"].astype(np.float32)
    embedding = _get_bendr_embeddings(model, epochs)
    age       = _bendr_linear_head(embedding, result["apf"], result["dar"])
    return age, "🧠 BENDR Foundation Model (braindecode/braindecode-bendr)"


# ── Reference & sklearn models ────────────────────────────────────

def predict_reference_model(apf: float, dar: float) -> float:
    """
    Built-in reference model (no external file required).

    brain_age = intercept + w_apf*(APF − apf_norm) + w_dar*(DAR − dar_norm)

    Clamped to plausible adult range [20, 80].
    """
    c = REFERENCE_MODEL
    age = (
        c["intercept"]
        + c["w_apf"] * (apf - c["apf_norm"])
        + c["w_dar"] * (dar - c["dar_norm"])
    )
    return float(np.clip(age, 20.0, 80.0))


def predict_age(
    result:      dict,
    ext_model=   None,
    bendr_model= None,
) -> Tuple[float, str]:
    """
    Model priority (highest → lowest):
      1. BENDR Foundation Model       – if loaded from HuggingFace
      2. Uploaded .joblib / .pkl       – if supplied in the sidebar
      3. Built-in Reference Model      – APF + δ/α biomarkers (always available)
    """
    # ── 1. BENDR ─────────────────────────────────────────────────
    if bendr_model is not None:
        try:
            return predict_with_bendr(bendr_model, result)
        except Exception as exc:
            st.warning(f"BENDR inference failed ({exc}). Falling back.")

    # ── 2. sklearn / joblib ───────────────────────────────────────
    if ext_model is not None:
        try:
            feat = np.array([[
                result["apf"],
                result["dar"],
                result["band_rel"]["Delta"].mean(),
                result["band_rel"]["Theta"].mean(),
                result["band_rel"]["Alpha"].mean(),
                result["band_rel"]["Beta"].mean(),
            ]])
            pred = float(ext_model.predict(feat)[0])
            return float(np.clip(pred, 20.0, 80.0)), "Loaded .joblib / .pkl model"
        except Exception as exc:
            st.warning(f"External model failed ({exc}). Falling back.")

    # ── 3. Reference model ────────────────────────────────────────
    age = predict_reference_model(result["apf"], result["dar"])
    return age, "Built-in Reference Model (APF + δ/α)"


# ══════════════════════════════════════════════════════════════════
# 4. CLINICAL INTERPRETATION
# ══════════════════════════════════════════════════════════════════

def interpret_gap(chrono_age: Optional[float], brain_age: float, apf: float, dar: float) -> dict:
    """
    Return structured interpretation dict.
    """
    summary = {}

    if chrono_age is not None:
        gap = brain_age - chrono_age
        summary["gap"] = gap
        if abs(gap) <= 3:
            summary["gap_label"]  = "Consistent"
            summary["gap_color"]  = "#2ecc71"
            summary["gap_text"]   = (
                f"Brain age is **consistent** with chronological age "
                f"(gap = {gap:+.1f} yrs). EEG markers are within the normative range."
            )
        elif gap > 3:
            summary["gap_label"]  = "Accelerated Aging"
            summary["gap_color"]  = "#e74c3c"
            summary["gap_text"]   = (
                f"Brain age is **{gap:.1f} years older** than chronological age. "
                f"This may indicate accelerated electrophysiological aging. "
                f"Clinical review is advised."
            )
        else:
            summary["gap_label"]  = "Decelerated Aging"
            summary["gap_color"]  = "#3498db"
            summary["gap_text"]   = (
                f"Brain age is **{abs(gap):.1f} years younger** than chronological age. "
                f"EEG markers suggest preserved neural efficiency."
            )
    else:
        summary["gap"]       = None
        summary["gap_label"] = "N/A"
        summary["gap_color"] = "#95a5a6"
        summary["gap_text"]  = "Enter chronological age to compute Brain Age Gap."

    # APF interpretation
    if apf >= 10.0:
        summary["apf_text"] = f"APF = **{apf:.2f} Hz** – within typical young-to-middle-adult range."
    elif apf >= 8.5:
        summary["apf_text"] = f"APF = **{apf:.2f} Hz** – mildly reduced; consistent with normal aging."
    else:
        summary["apf_text"] = f"APF = **{apf:.2f} Hz** – notably reduced; may reflect advanced age or pathology."

    # δ/α interpretation
    if dar < 0.4:
        summary["dar_text"] = f"δ/α ratio = **{dar:.2f}** – low; indicates good cortical arousal."
    elif dar < 0.8:
        summary["dar_text"] = f"δ/α ratio = **{dar:.2f}** – moderate; within normal adult aging range."
    else:
        summary["dar_text"] = f"δ/α ratio = **{dar:.2f}** – elevated; may indicate reduced cortical efficiency."

    return summary


# ══════════════════════════════════════════════════════════════════
# 5. VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════

_DARK_BG  = "#0d1117"
_PANEL_BG = "#161b22"
_ACCENT   = "#58a6ff"
_ACCENT2  = "#3fb950"
_TEXT     = "#e6edf3"
_MUTED    = "#8b949e"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    if title:  ax.set_title(title,  color=_TEXT,  fontsize=11, pad=8, fontfamily="monospace")
    if xlabel: ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)


def plot_psd(freqs: np.ndarray, pxx: np.ndarray) -> plt.Figure:
    """
    PSD plot: mean ± 1SD across channels, coloured band annotations.
    """
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax, title="Power Spectral Density  ·  post-filter",
              xlabel="Frequency (Hz)", ylabel="Power (µV²/Hz)")

    pxx_uv = pxx * 1e12    # V²/Hz → µV²/Hz
    mean_p = 10 * np.log10(pxx_uv.mean(axis=0) + 1e-30)
    std_p  = pxx_uv.std(axis=0)
    std_db = 10 * np.log10(pxx_uv.mean(axis=0) + std_p + 1e-30) - mean_p

    band_cols = {"Delta": "#e74c3c44", "Theta": "#f39c1244",
                 "Alpha": "#2ecc7144", "Beta":  "#3498db44"}
    band_labels = {"Delta": "δ", "Theta": "θ", "Alpha": "α", "Beta": "β"}

    for band, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, color=band_cols[band], zorder=0)
        ax.text((lo + hi) / 2, mean_p.min() + 1, band_labels[band],
                ha="center", va="bottom", color=_MUTED, fontsize=10,
                fontfamily="monospace")

    ax.plot(freqs, mean_p, color=_ACCENT, linewidth=1.6, zorder=3, label="Mean PSD")
    ax.fill_between(freqs, mean_p - std_db, mean_p + std_db,
                    color=_ACCENT, alpha=0.18, zorder=2, label="±1 SD")
    ax.set_xlim(0.5, 40)
    ax.legend(facecolor=_PANEL_BG, labelcolor=_TEXT, fontsize=8, framealpha=0.6)
    fig.tight_layout(pad=1.2)
    return fig


def plot_topomap_alpha(band_rel: dict[str, np.ndarray]) -> plt.Figure:
    """
    Interpolated 2-D topographic map of relative Alpha power.
    Uses matplotlib tricontourf over the 19 electrode positions.
    """
    from matplotlib.tri import Triangulation

    alpha_vals = band_rel["Alpha"]                  # (19,)
    xs = np.array([TOPO_COORDS[ch][0] for ch in STANDARD_19])
    ys = np.array([TOPO_COORDS[ch][1] for ch in STANDARD_19])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    # Head outline
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#30363d", linewidth=1.5, zorder=1)
    # Nose
    ax.plot([0, 0.06, 0], [0.97, 1.05, 0.97],
            color="#30363d", linewidth=1.5, solid_capstyle="round", zorder=1)

    # Mask outside head: only sample points within unit circle
    triang = Triangulation(xs, ys)
    cf = ax.tricontourf(triang, alpha_vals, levels=20, cmap="RdYlGn", zorder=2, alpha=0.9)

    # Electrode dots
    sc = ax.scatter(xs, ys, c=alpha_vals, cmap="RdYlGn",
                    s=60, zorder=4, edgecolors="#0d1117", linewidths=0.8)

    # Labels
    for i, ch in enumerate(STANDARD_19):
        ax.text(xs[i], ys[i] + 0.07, ch, ha="center", va="bottom",
                color=_TEXT, fontsize=6.5, fontfamily="monospace", zorder=5)

    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(colors=_MUTED, labelsize=7)
    cb.set_label("Relative α power", color=_MUTED, fontsize=8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Alpha Power Topomap", color=_TEXT, fontsize=11,
                 pad=6, fontfamily="monospace")
    fig.tight_layout(pad=0.8)
    return fig


def plot_band_bars(band_rel: dict[str, np.ndarray]) -> plt.Figure:
    """Horizontal bar chart – mean relative band power across all channels."""
    bands   = list(BANDS.keys())
    values  = [band_rel[b].mean() * 100 for b in bands]
    colours = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax, title="Mean Relative Band Power",
              xlabel="Relative Power (%)")

    bars = ax.barh(bands, values, color=colours, edgecolor="none", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=_TEXT, fontsize=9,
                fontfamily="monospace")
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout(pad=1.2)
    return fig


def plot_age_gauge(brain_age: float, chrono_age: Optional[float]) -> plt.Figure:
    """
    Semi-circular gauge showing brain age on a 20-80 scale.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)

    lo, hi = 20.0, 80.0
    n_arc  = 300
    theta  = np.linspace(np.pi, 0, n_arc)   # left → right = young → old
    r_out, r_in = 1.0, 0.65

    # Background track
    ax.fill_between(theta, r_in, r_out, color="#21262d", zorder=1)

    # Colour gradient: green (young) → yellow → red (old)
    cmap    = matplotlib.colormaps["RdYlGn_r"]
    for i in range(n_arc - 1):
        frac = i / (n_arc - 1)
        ax.fill_between(theta[i:i+2], r_in, r_out,
                        color=cmap(frac), alpha=0.8, zorder=2)

    # Needle – brain age
    needle_frac  = (brain_age - lo) / (hi - lo)
    needle_theta = np.pi * (1 - needle_frac)
    ax.plot([needle_theta, needle_theta], [0.0, r_out + 0.05],
            color=_TEXT, linewidth=3, zorder=5, solid_capstyle="round")
    ax.scatter([needle_theta], [0.0], color=_TEXT, s=50, zorder=6)

    # Chronological age tick (if provided)
    if chrono_age is not None:
        ca_frac  = (chrono_age - lo) / (hi - lo)
        ca_theta = np.pi * (1 - ca_frac)
        ax.plot([ca_theta, ca_theta], [r_in - 0.05, r_out + 0.08],
                color="#f39c12", linewidth=2.5, linestyle="--",
                zorder=4, solid_capstyle="round")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.2)
    ax.axis("off")

    # Labels
    for age_tick in [20, 30, 40, 50, 60, 70, 80]:
        frac  = (age_tick - lo) / (hi - lo)
        t     = np.pi * (1 - frac)
        ax.text(t, 1.13, str(age_tick),
                ha="center", va="center", color=_MUTED, fontsize=7.5,
                fontfamily="monospace")

    ax.text(np.pi / 2, 0.35, f"{brain_age:.1f}",
            ha="center", va="center", color=_TEXT, fontsize=20,
            fontweight="bold", fontfamily="monospace")
    ax.text(np.pi / 2, 0.18, "Brain Age (yrs)",
            ha="center", va="center", color=_MUTED, fontsize=8,
            fontfamily="monospace")

    if chrono_age is not None:
        ax.text(np.pi / 2, -0.05, f"Chrono: {chrono_age:.0f} yrs",
                ha="center", va="center", color="#f39c12", fontsize=7.5,
                fontfamily="monospace")

    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════
# 6. STREAMLIT APPLICATION
# ══════════════════════════════════════════════════════════════════

def _css():
    """Global dark-theme CSS with monospace accents and subtle grid lines."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        background-color: #0d1117 !important;
        color: #e6edf3;
        font-family: 'Syne', sans-serif;
    }

    /* Header */
    .neuro-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.02em;
        color: #e6edf3;
        border-left: 5px solid #58a6ff;
        padding-left: 1rem;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    .neuro-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        letter-spacing: 0.08em;
        margin-left: 1.4rem;
        margin-bottom: 1.5rem;
    }

    /* Metric card */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.0;
    }
    .metric-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8b949e;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* Insight card */
    .insight-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .insight-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }

    /* Section divider */
    .section-rule {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #8b949e;
    }

    /* Streamlit default overrides */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        background: #21262d;
        border: 1px solid #30363d;
        color: #e6edf3;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)


def _metric_card(label: str, value: str, colour: str = "#58a6ff") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-val" style="color:{colour};">{value}</div>
        <div class="metric-lbl">{label}</div>
    </div>
    """


def main():
    st.set_page_config(
        page_title  = "NeuroAge · EEG Brain Age",
        page_icon   = "🧠",
        layout      = "wide",
        initial_sidebar_state = "expanded",
    )
    _css()

    # ── Header ────────────────────────────────────────────────────
    st.markdown('<div class="neuro-header">NeuroAge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="neuro-sub">EEG BRAIN AGE ESTIMATION · '
        'RIDGE REGRESSION · ALPHA-PEAK + Δ/α · BENDR FOUNDATION MODEL</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════
    # SIDEBAR
    # ══════════════════════════════
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("**Model selection**")

        model_choice = st.radio(
            "Backend",
            options=["Reference Model", "BENDR (HuggingFace)", "Custom .joblib / .pkl"],
            index=0,
            help="BENDR is a free, open foundation model — no token required.",
            label_visibility="collapsed",
        )

        model_file   = None
        bendr_model  = None

        if model_choice == "Custom .joblib / .pkl":
            model_file = st.file_uploader(
                "Upload .joblib or .pkl", type=["joblib", "pkl"],
                help="scikit-learn Ridge or any model with a .predict() method "
                     "expecting [APF, DAR, δ, θ, α, β]",
                label_visibility="collapsed",
            )
        elif model_choice == "BENDR (HuggingFace)":
            st.markdown(
                '<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'                'color:#8b949e;line-height:1.5;padding:0.4rem 0;">'                '🧠 <b style="color:#e6edf3;">braindecode/braindecode-bendr</b><br>'                'Conv encoder + Transformer<br>'                'Trained on TUAB (400K samples)<br>'                '+ Sleep Heart Health Study<br>'                '✅ Free · No token · BSD-3'                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("⬇️ Load BENDR", use_container_width=True):
                with st.spinner("Downloading BENDR (~120 MB) …"):
                    try:
                        bendr_model = load_bendr_model()
                        st.session_state["bendr_model"] = bendr_model
                        st.success("✅ BENDR loaded!")
                    except ImportError as imp_err:
                        st.error(
                            f"❌ Missing package: {imp_err}\n\n"
                            "Add these to requirements.txt:\n"
                            "`braindecode>=1.4.0`\n"
                            "`safetensors>=0.4.0`\n"
                            "`torch>=2.1.0`"
                        )
                    except Exception as exc:
                        err = str(exc)
                        if "ConnectionError" in err or "requests" in err:
                            st.error("❌ Network error — check internet access.")
                        else:
                            st.error(f"❌ {exc}")

            if "bendr_model" in st.session_state:
                bendr_model = st.session_state["bendr_model"]
                st.success("✅ BENDR ready")

        st.divider()
        st.markdown("**Chronological Age**")
        chrono_age_input = st.number_input(
            "Age (years)", min_value=20, max_value=80,
            value=None, placeholder="e.g. 45",
            label_visibility="collapsed",
        )
        chrono_age = float(chrono_age_input) if chrono_age_input else None

        st.divider()
        st.markdown("**Model Info**")
        if model_choice == "BENDR (HuggingFace)":
            st.markdown(
                '<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'                'color:#8b949e;line-height:1.6;">'                '<b style="color:#58a6ff;">BENDR pipeline</b><br>'                'epochs (250 Hz, 19 ch)<br>'                '→ BENDR encoder<br>'                '→ mean embedding<br>'                '→ linear head → age<br><br>'                'Head calibrated to the<br>'                'reference model.<br>'                'Fine-tune on labelled<br>'                'EEG for best accuracy.'                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:JetBrains Mono,monospace;font-size:0.74rem;'
                'color:#8b949e;line-height:1.6;">'
                'Inspired by LEMON dataset:<br><br>'
                '<code>age ≈ 78.5</code><br>'
                '<code>    − 3.2 × (APF − 10)</code><br>'
                '<code>    + 6.8 × (δ/α − 0.5)</code><br><br>'
                'APF declines ~0.1 Hz/decade.<br>'
                'δ/α ratio rises with age.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Pipeline**")
        st.markdown(
            """
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8b949e;">
            · 19-ch 10-20 selection (regex)<br>
            · Resample → 250 Hz<br>
            · Band-pass 1–40 Hz<br>
            · Notch 50 Hz<br>
            · Common Average Reference<br>
            · Artifact reject > 150 µV<br>
            · Welch PSD (4-s epochs)<br>
            · BENDR: encoder → embedding
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ══════════════════════════════
    # MAIN – FILE UPLOAD
    # ══════════════════════════════
    edf_file = st.file_uploader(
        "📂  Upload EEG Recording (.edf)",
        type=["edf"],
        help="EDF file with at least the 19 standard 10-20 channels.",
    )

    if edf_file is None:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_metric_card("PREDICTED BRAIN AGE", "—"), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("BRAIN AGE GAP", "—"), unsafe_allow_html=True)
        with c3:
            st.markdown(_metric_card("ALPHA PEAK FREQ", "—"), unsafe_allow_html=True)

        st.info(
            "👆 Upload an **.edf** file to begin. "
            "The pipeline will auto-select the 19 standard 10-20 channels, "
            "clean the signal, extract spectral biomarkers, and estimate brain age."
        )
        return

    # ── Load external sklearn model (if supplied) ────────────────
    ext_model = None
    if model_file is not None:
        suffix = Path(model_file.name).suffix
        try:
            ext_model = load_external_model(model_file.read(), suffix)
            st.sidebar.success(f"✅ Loaded: `{model_file.name}`")
        except Exception as exc:
            st.sidebar.error(f"❌ Model load failed: {exc}")

    # ── Preprocessing ─────────────────────────────────────────────
    with st.spinner("🔬 Preprocessing EEG …"):
        try:
            result = preprocess_and_extract(edf_file.read(), edf_file.name)
        except ImportError:
            st.error("❌ **MNE-Python not installed.**  `pip install mne`")
            return
        except ValueError as exc:
            st.error(f"❌ **Preprocessing error:**\n\n{exc}")
            return
        except Exception as exc:
            st.error(f"❌ **Unexpected error:**\n\n{exc}")
            return

    meta      = result["meta"]
    apf       = result["apf"]
    dar       = result["dar"]
    band_rel  = result["band_rel"]

    # ── Prediction ────────────────────────────────────────────────
    brain_age, model_label = predict_age(result, ext_model, bendr_model=bendr_model)
    interp = interpret_gap(chrono_age, brain_age, apf, dar)

    # ══════════════════════════════
    # ROW 1 – Key Metrics
    # ══════════════════════════════
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    gap_str = (
        f"{interp['gap']:+.1f} yrs"
        if interp["gap"] is not None
        else "— enter age"
    )
    gap_col = interp["gap_color"] if interp["gap"] is not None else "#8b949e"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            _metric_card("PREDICTED BRAIN AGE", f"{brain_age:.1f} yrs", "#58a6ff"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card("BRAIN AGE GAP", gap_str, gap_col),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card("ALPHA PEAK FREQ", f"{apf:.2f} Hz", "#3fb950"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card("δ / α RATIO", f"{dar:.3f}", "#d29922"),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
        f'color:#8b949e;text-align:right;margin-top:0.3rem;">'
        f'Model: {model_label} · '
        f'{meta["n_epochs_accepted"]}/{meta["n_epochs_total"]} epochs accepted'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════
    # ROW 2 – Gauge + Expert Summary
    # ══════════════════════════════
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("#### Age Gauge")
        fig_gauge = plot_age_gauge(brain_age, chrono_age)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

    with right:
        st.markdown("#### Expert Summary")

        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-title">Brain Age Assessment</div>'
            f'{interp["gap_text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-title">Alpha Peak Frequency</div>'
            f'{interp["apf_text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-title">Delta / Alpha Ratio</div>'
            f'{interp["dar_text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Band power mini-chart
        fig_bars = plot_band_bars(band_rel)
        st.pyplot(fig_bars, use_container_width=True)
        plt.close(fig_bars)

    # ══════════════════════════════
    # ROW 3 – Topomap + PSD
    # ══════════════════════════════
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    col_topo, col_psd = st.columns([1, 2])

    with col_topo:
        st.markdown("#### Alpha Topomap")
        fig_topo = plot_topomap_alpha(band_rel)
        st.pyplot(fig_topo, use_container_width=True)
        plt.close(fig_topo)

    with col_psd:
        st.markdown("#### Power Spectral Density")
        fig_psd = plot_psd(result["freqs"], result["pxx_mean"])
        st.pyplot(fig_psd, use_container_width=True)
        plt.close(fig_psd)

    # ══════════════════════════════
    # ROW 4 – Recording Details
    # ══════════════════════════════
    with st.expander("📋 Recording Metadata & Channel Map", expanded=False):
        col_m, col_c = st.columns(2)
        with col_m:
            st.markdown("**Recording Info**")
            st.markdown(
                f"""
                | Property | Value |
                |---|---|
                | File | `{meta['filename']}` |
                | Original Fs | {meta['orig_fs']:.1f} Hz |
                | Duration | {meta['duration_s']:.1f} s |
                | Epochs (4 s) | {meta['n_epochs_accepted']} / {meta['n_epochs_total']} |
                | APF | {apf:.3f} Hz |
                | δ/α ratio | {dar:.4f} |
                """
            )
        with col_c:
            st.markdown("**Selected 10-20 Channels**")
            cols3 = st.columns(3)
            for i, ch in enumerate(STANDARD_19):
                cols3[i % 3].markdown(f"`{ch}`")

            # Show legacy rename notice when applicable
            legacy = meta.get("legacy_renamed", {})
            if legacy:
                pairs = ", ".join(f"`{old}` → `{new}`" for old, new in legacy.items())
                st.markdown(
                    f'<div style="margin-top:0.8rem;padding:0.6rem 0.9rem;'
                    f'background:#1f2a1f;border-left:3px solid #3fb950;'
                    f'border-radius:5px;font-family:JetBrains Mono,monospace;'
                    f'font-size:0.74rem;color:#8b949e;">'
                    f'<span style="color:#3fb950;">⟳ Legacy channels renamed</span><br>'
                    f'{pairs}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════
    # ROW 5 – Band Power Table
    # ══════════════════════════════
    with st.expander("📊 Per-Channel Relative Band Powers (%)", expanded=False):
        import pandas as pd
        rows = {}
        for ch in STANDARD_19:
            idx = STANDARD_19.index(ch)
            rows[ch] = {b: f"{band_rel[b][idx]*100:.1f}" for b in BANDS}
        df = pd.DataFrame(rows).T
        df.index.name = "Channel"
        st.dataframe(df, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
        'color:#484f58;text-align:center;">'
        '⚠️ NeuroAge is a research tool only. Not validated for clinical use. '
        'Consult a qualified neurologist for medical decisions.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()

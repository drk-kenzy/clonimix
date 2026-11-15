"""Profil de prosodie : repère une voix trop plate / robotique.

Une voix robotique varie peu en hauteur, en énergie et en rythme. On mesure :
- f0_std : variation de hauteur, en demi-tons (plus haut = plus expressif)
- f0_range : étendue de hauteur (percentile 90 - 10), en demi-tons
- db_std : variation d'énergie de la voix, en dB
- pause_pct : part du temps passée en silence
Ce sont des indicateurs, pas un verdict : l'oreille reste le juge.
"""

from __future__ import annotations

import numpy as np

SR = 16000
HOP = 256


def profile(wav: np.ndarray, sr: int = SR) -> dict:
    import librosa

    if sr != SR:
        wav = librosa.resample(np.asarray(wav, dtype="float32"), orig_sr=sr, target_sr=SR)
    wav = np.asarray(wav, dtype="float32")
    peak = float(np.max(np.abs(wav))) or 1.0
    wav = wav / peak * 0.95

    f0, voiced, _ = librosa.pyin(
        wav, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=HOP
    )
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]

    db = 20 * np.log10(np.maximum(rms, 1e-5))
    speech = db > (db.max() - 35)  # au-dessus du plancher de bruit
    sel = voiced & speech & np.isfinite(f0)
    st = 12 * np.log2(f0[sel] / np.median(f0[sel])) if sel.sum() > 20 else np.array([0.0])

    return {
        "f0_std": round(float(np.std(st)), 2),
        "f0_range": round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
        "db_std": round(float(np.std(db[sel])) if sel.sum() > 20 else 0.0, 2),
        "pause_pct": round(float(100 * (1 - speech.mean())), 1),
        "f0_median_hz": round(float(np.median(f0[sel])) if sel.sum() > 20 else 0.0),
    }

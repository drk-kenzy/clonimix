"""Mesure de ressemblance vocale.

On compare l'empreinte vocale (VoiceEncoder de Chatterbox) d'un audio à celle
d'une référence. Score entre -1 et 1 (cosinus) : plus c'est haut, plus la
voix ressemble. C'est un indicateur objectif du *timbre*, pas de la prosodie
ni du naturel : l'oreille reste le juge final.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000


def load_ve(device: str = "cpu"):
    """Charge uniquement l'encodeur de voix (léger, sans le modèle complet)."""
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from huggingface_hub import snapshot_download

    kwargs = dict(repo_id="ResembleAI/chatterbox", allow_patterns=["ve.pt"])
    try:
        ckpt = Path(snapshot_download(local_files_only=True, **kwargs))
    except Exception:  # noqa: BLE001 - pas encore en cache
        ckpt = Path(snapshot_download(**kwargs))
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt / "ve.pt", map_location="cpu", weights_only=True))
    return ve.to(device).eval()


def _normalize(wav: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) or 1.0
    return (wav / peak * 0.95).astype("float32")


def load_wav(path: str | Path, sr: int = SR) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return _normalize(wav)


def embed(ve, wavs, sr: int = SR) -> np.ndarray:
    """Empreinte vocale unique (norme 1) d'un ou plusieurs audios."""
    import librosa

    if isinstance(wavs, np.ndarray):
        wavs = [wavs]
    wavs = [np.asarray(w, dtype="float32") for w in wavs]
    if sr != SR:  # on rééchantillonne ici (resampy, requis par l'encodeur, n'est pas installé)
        wavs = [librosa.resample(w, orig_sr=sr, target_sr=SR) for w in wavs]
    wavs = [_normalize(w) for w in wavs]
    return ve.embeds_from_wavs(wavs, sample_rate=SR, as_spk=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

"""Gestion des voix enregistrées (stockage 100 % local)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

TARGET_SR = 24000
MIN_SECONDS = 5.0
MAX_SECONDS = 30.0


def home() -> Path:
    """Dossier de données. Surchargeable avec CLONIMIX_HOME."""
    base = Path(os.environ.get("CLONIMIX_HOME", Path.home() / ".clonimix"))
    (base / "voices").mkdir(parents=True, exist_ok=True)
    return base


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError(
            "Nom de voix invalide : lettres, chiffres, _ et - uniquement (40 max)."
        )
    return name


def voice_dir(name: str) -> Path:
    return home() / "voices" / _safe_name(name)


def reference_path(name: str) -> Path:
    path = voice_dir(name) / "reference.wav"
    if not path.exists():
        raise FileNotFoundError(
            f"Voix '{name}' introuvable. Lance d'abord : clonimix enroll <audio> --name {name}"
        )
    return path


def load_params(name: str) -> dict:
    """Réglages optimisés enregistrés pour cette voix (vide si aucun)."""
    meta = voice_dir(name) / "meta.json"
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8")).get("params", {})


def list_voices() -> list[dict]:
    voices = []
    for meta in sorted((home() / "voices").glob("*/meta.json")):
        voices.append(json.loads(meta.read_text(encoding="utf-8")))
    return voices


def remove_voice(name: str) -> None:
    import shutil

    folder = voice_dir(name)
    if not folder.exists():
        raise FileNotFoundError(f"Voix '{name}' introuvable.")
    shutil.rmtree(folder)


def enroll(
    source: Path, name: str, own_voice_confirmed: bool, denoise: bool = False
) -> dict:
    """Prépare un échantillon de référence propre à partir d'un fichier audio."""
    if not own_voice_confirmed:
        raise PermissionError(
            "Consentement requis : tu dois confirmer que c'est TA voix "
            "(ou celle d'une personne qui a donné son accord explicite)."
        )

    import librosa
    import numpy as np
    import soundfile as sf

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Fichier introuvable : {source}")

    audio, _ = librosa.load(str(source), sr=TARGET_SR, mono=True)
    if denoise:
        # Option : à n'activer que si l'enregistrement est vraiment bruité.
        # Testé : sur un enregistrement propre, le débruitage abîme le timbre
        # et donne un clone moins fidèle. Le mieux reste un bon enregistrement.
        import noisereduce as nr

        audio = nr.reduce_noise(y=audio, sr=TARGET_SR, stationary=False, prop_decrease=0.85)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    duration = len(audio) / TARGET_SR
    if duration < MIN_SECONDS:
        raise ValueError(
            f"Échantillon trop court ({duration:.1f}s). Il faut au moins {MIN_SECONDS:.0f}s de parole claire."
        )
    if duration > MAX_SECONDS:
        audio = audio[: int(MAX_SECONDS * TARGET_SR)]
        duration = MAX_SECONDS

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.95

    folder = voice_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "reference.wav"), audio, TARGET_SR)

    meta = {
        "name": name,
        "duration_s": round(duration, 1),
        "source_file": source.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "own_voice_confirmed": True,
        "denoised": denoise,
    }
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta

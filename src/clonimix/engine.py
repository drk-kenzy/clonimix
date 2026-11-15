"""Moteur de synthèse. Par défaut : Chatterbox multilingue (MIT, tourne en local).

Chatterbox ajoute automatiquement un filigrane audio inaudible (Perth) aux voix
générées, ce qui permet d'identifier un audio synthétique.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_CHARS = 250


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Découpe un long texte en morceaux courts, aux frontières de phrases."""
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _lazy_chinese_segmenter() -> None:
    """Évite un téléchargement inutile au démarrage.

    Chatterbox charge à chaque lancement un découpeur de chinois (pkuseg) qui
    télécharge un fichier depuis GitHub (délai de 5 s : échoue sur une connexion
    lente) même pour du français. On ne le charge que si le texte contient du
    chinois ; sans réseau, le chinois est alors traité caractère par caractère.
    """
    from chatterbox.models.tokenizers import tokenizer as tk

    cls = tk.ChineseCangjieConverter
    if getattr(cls, "_clonimix_lazy", False):
        return
    original_init, original_call = cls._init_segmenter, cls.__call__

    def deferred_init(self):
        self.segmenter = None
        self._segmenter_pending = True

    def call(self, text):
        if getattr(self, "_segmenter_pending", False) and any(
            "一" <= ch <= "鿿" for ch in text
        ):
            self._segmenter_pending = False
            try:
                original_init(self)
            except Exception:  # noqa: BLE001 - pas de réseau : repli caractère par caractère
                self.segmenter = None
        return original_call(self, text)

    cls._init_segmenter, cls.__call__, cls._clonimix_lazy = deferred_init, call, True


class Engine:
    def __init__(self, device: str | None = None):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            self.device = pick_device(self.device)
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            _lazy_chinese_segmenter()
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        return self._model

    def synthesize(
        self,
        text: str,
        reference: Path,
        lang: str = "fr",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        pause_s: float = 0.25,
        best_of: int = 1,
        log=print,
    ):
        """Retourne (signal numpy mono, fréquence d'échantillonnage).

        best_of > 1 : pour chaque phrase, génère plusieurs candidats et garde
        celui dont l'empreinte vocale ressemble le plus à la référence.
        """
        import numpy as np

        model = self._load()
        model.prepare_conditionals(str(reference), exaggeration=exaggeration)

        ref_emb = None
        if best_of > 1:
            from . import similarity as sim

            ref_emb = sim.embed(model.ve, sim.load_wav(reference))

        pieces = []
        gap = np.zeros(int(pause_s * model.sr), dtype="float32")
        chunks = split_text(text)
        for i, chunk in enumerate(chunks, 1):
            best, best_score = None, -2.0
            for _ in range(max(1, best_of)):
                wav = model.generate(
                    chunk,
                    language_id=lang,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                sig = wav.squeeze().detach().cpu().numpy().astype("float32")
                if ref_emb is None:
                    best = sig
                    break
                score = sim.cosine(sim.embed(model.ve, sig, model.sr), ref_emb)
                if score > best_score:
                    best, best_score = sig, score
            if ref_emb is not None:
                log(f"  phrase {i}/{len(chunks)} : meilleur score {best_score:.3f}")
            pieces.append(best)
            pieces.append(gap)
        return np.concatenate(pieces[:-1]), model.sr

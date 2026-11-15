"""Optimisation d'une voix : meilleur extrait de référence + meilleurs réglages.

Chatterbox n'écoute que les ~10 premières secondes de la référence (6 s pour le
rythme, 10 s pour le timbre). On teste donc plusieurs fenêtres de 10 s, on
mesure le bruit entre deux tirages (la synthèse est aléatoire), puis on essaie
quelques réglages et on garde ceux qui font un vrai gain.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TARGET_SR = 24000
WIN_S = 10.0
HOP_S = 3.0
MIN_GAIN = 0.01  # gain minimal (score) pour adopter un réglage


def _prepare_source(path: Path) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    peak = float(np.max(np.abs(wav))) or 1.0
    return (wav / peak * 0.95).astype("float32")


def _window_starts(n: int) -> list[int]:
    win, hop = int(WIN_S * TARGET_SR), int(HOP_S * TARGET_SR)
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if n - win - starts[-1] > TARGET_SR:  # évite deux fenêtres quasi identiques
        starts.append(n - win)
    return starts


def run(
    source: Path,
    holdout: list[str],
    text: str,
    save_as: str,
    own_voice_confirmed: bool,
    lang: str = "fr",
) -> dict:
    if not own_voice_confirmed:
        raise PermissionError("Consentement requis : confirme que c'est ta voix.")

    import soundfile as sf
    import torch

    from . import similarity as sim
    from . import voices
    from .engine import Engine

    outdir = Path("outputs") / "tune"
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = outdir / "log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    full = _prepare_source(Path(source))
    log(f"Source : {len(full) / TARGET_SR:.1f}s utiles. Chargement du modèle...")
    model = Engine()._load()
    ve = model.ve

    ref_emb = sim.embed(ve, full, TARGET_SR)
    hold_emb = sim.embed(ve, [sim.load_wav(p) for p in holdout]) if holdout else None

    results: list[dict] = []

    def trial(name, ref_path, ex=0.5, cfg=0.5, temp=0.8, seed=1) -> dict:
        t0 = time.time()
        torch.manual_seed(seed)
        model.prepare_conditionals(str(ref_path), exaggeration=ex)
        wav = model.generate(
            text, language_id=lang, exaggeration=ex, cfg_weight=cfg, temperature=temp
        )
        sig = wav.squeeze().detach().cpu().numpy().astype("float32")
        sf.write(str(outdir / f"{name}.wav"), sig, model.sr)
        emb = sim.embed(ve, sig, model.sr)
        rec = {
            "name": name,
            "ref": round(sim.cosine(emb, ref_emb), 3),
            "held": None if hold_emb is None else round(sim.cosine(emb, hold_emb), 3),
            "ex": ex, "cfg": cfg, "temp": temp, "seed": seed,
            "sec": round(time.time() - t0),
        }
        results.append(rec)
        log(json.dumps(rec, ensure_ascii=False))
        return rec

    # 1) Quelle fenêtre de 10 s sert le mieux de référence ?
    win = int(WIN_S * TARGET_SR)
    windows = []
    for i, start in enumerate(_window_starts(len(full))):
        path = outdir / f"ref_{i}.wav"
        sf.write(str(path), full[start : start + win], TARGET_SR)
        rec = trial(f"fenetre{i}_{start / TARGET_SR:.0f}s", path)
        windows.append((rec, path))
    best_rec, best_path = max(windows, key=lambda x: x[0]["ref"])
    log(f"Meilleure fenêtre : {best_rec['name']} (score {best_rec['ref']})")

    # 2) Bruit entre tirages (même réglages, autre graine)
    noise = [best_rec["ref"]] + [
        trial(f"bruit_graine{s}", best_path, seed=s)["ref"] for s in (2, 3)
    ]
    base_mean = float(np.mean(noise))
    log(f"Bruit entre tirages : min {min(noise)}, max {max(noise)}, moyenne {base_mean:.3f}")

    # 3) Réglages
    variants = {
        "exag0.3": dict(ex=0.3), "exag0.7": dict(ex=0.7), "cfg0.25": dict(cfg=0.25),
        "temp0.6": dict(temp=0.6), "temp1.0": dict(temp=1.0),
    }
    scored = {n: trial(f"reglage_{n}", best_path, **kw) for n, kw in variants.items()}
    best_name = max(scored, key=lambda n: scored[n]["ref"])
    params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
    # Un réglage n'est adopté que s'il bat le MEILLEUR des tirages identiques :
    # sinon le « gain » est du hasard (le bruit entre tirages est de l'ordre de 0.05).
    if scored[best_name]["ref"] > max(noise) + MIN_GAIN:
        kw = variants[best_name]
        params = {
            "exaggeration": kw.get("ex", 0.5),
            "cfg_weight": kw.get("cfg", 0.5),
            "temperature": kw.get("temp", 0.8),
        }
        log(f"Réglage adopté : {best_name} -> {params}")
    else:
        log("Aucun réglage ne dépasse le bruit : réglages par défaut conservés.")

    # 4) Les scores d'ancrage prédisent-ils le score hors échantillon ?
    corr = None
    pairs = [(r["ref"], r["held"]) for r in results if r["held"] is not None]
    if len(pairs) >= 4:
        corr = round(float(np.corrcoef(*zip(*pairs))[0, 1]), 2)
        log(f"Corrélation score-référence / score-hors-échantillon : {corr}")

    # 5) Sauvegarde de la meilleure voix
    folder = voices.voice_dir(save_as)
    folder.mkdir(parents=True, exist_ok=True)
    seg = sf.read(str(best_path))[0]
    sf.write(str(folder / "reference.wav"), seg, TARGET_SR)
    meta = {
        "name": save_as,
        "duration_s": round(len(seg) / TARGET_SR, 1),
        "source_file": Path(source).name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "own_voice_confirmed": True,
        "denoised": False,
        "tuned": True,
        "params": params,
    }
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {"results": results, "params": params, "base_mean": base_mean, "corr": corr}
    (outdir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Voix '{save_as}' sauvegardée. Résumé : {outdir / 'results.json'}")
    return summary

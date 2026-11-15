"""Réglage « naturel » : moins robotique, sans perdre la ressemblance.

Le clone hérite de l'expressivité de la référence, et une voix robotique varie
peu en hauteur. On garde la ressemblance comme contrainte (score proche du
meilleur) et on choisit, parmi les candidats, celui dont la variation de
hauteur est la plus proche de ta vraie voix.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIGS = [
    ("base", dict(ex=0.5, cfg=0.5, temp=0.8)),
    ("exp0.7_cfg0.3", dict(ex=0.7, cfg=0.3, temp=0.8)),
    ("exp0.9_cfg0.3", dict(ex=0.9, cfg=0.3, temp=0.8)),
    ("exp0.7_cfg0.2_t1.0", dict(ex=0.7, cfg=0.2, temp=1.0)),
    ("exp0.9_cfg0.5", dict(ex=0.9, cfg=0.5, temp=0.8)),
]
# Le bruit entre deux tirages identiques est mesuré à ~0.05 : en dessous, une
# différence de ressemblance n'est pas distinguable du hasard.
SIM_TOLERANCE = 0.05
PAUSE_S = 0.5  # pause entre phrases (les clones respirent trop peu)


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

    from . import prosody, voices
    from . import similarity as sim
    from .engine import Engine
    from .tune import TARGET_SR, _prepare_source

    tdir = Path("outputs") / "tune"
    ndir = Path("outputs") / "natural"
    ndir.mkdir(parents=True, exist_ok=True)
    logfile = ndir / "log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    # Prosodie de ta vraie voix (cible)
    full = _prepare_source(Path(source))
    target = prosody.profile(full, TARGET_SR)
    log(f"Cible (ta vraie voix) : {target}")

    # 1) Fenêtre de référence : parmi celles qui ressemblent autant, la plus expressive
    prev = json.loads((tdir / "results.json").read_text(encoding="utf-8"))["results"]
    wins = [r for r in prev if r["name"].startswith("fenetre")]
    for r in wins:
        r["f0_std"] = prosody.profile(sim.load_wav(tdir / f"{r['name']}.wav"))["f0_std"]
        log(f"  {r['name']}: ressemblance {r['ref']}, variation de hauteur {r['f0_std']}")
    top = max(r["ref"] for r in wins)
    near = [r for r in wins if r["ref"] >= top - SIM_TOLERANCE]
    pick = min(near, key=lambda r: abs(r["f0_std"] - target["f0_std"]))
    idx = int(pick["name"][len("fenetre"):].split("_")[0])
    ref_path = tdir / f"ref_{idx}.wav"
    log(f"Fenêtre retenue : {pick['name']}")

    # 2) Réglages d'expressivité
    model = Engine()._load()
    ve = model.ve
    ref_emb = sim.embed(ve, full, TARGET_SR)
    hold_emb = sim.embed(ve, [sim.load_wav(p) for p in holdout]) if holdout else None

    results = []
    for name, kw in CONFIGS:
        t0 = time.time()
        torch.manual_seed(1)
        model.prepare_conditionals(str(ref_path), exaggeration=kw["ex"])
        wav = model.generate(
            text, language_id=lang, exaggeration=kw["ex"],
            cfg_weight=kw["cfg"], temperature=kw["temp"],
        )
        sig = wav.squeeze().detach().cpu().numpy().astype("float32")
        sf.write(str(ndir / f"{name}.wav"), sig, model.sr)
        emb = sim.embed(ve, sig, model.sr)
        pro = prosody.profile(sig, model.sr)
        rec = {
            "name": name, **kw,
            "ref": round(sim.cosine(emb, ref_emb), 3),
            "held": None if hold_emb is None else round(sim.cosine(emb, hold_emb), 3),
            **pro, "sec": round(time.time() - t0),
        }
        results.append(rec)
        log(json.dumps(rec, ensure_ascii=False))

    # 3) Choix : ressemblance proche du meilleur, hauteur la plus proche de ta voix
    best_ref = max(r["ref"] for r in results)
    ok = [r for r in results if r["ref"] >= best_ref - SIM_TOLERANCE]
    chosen = min(ok, key=lambda r: abs(r["f0_std"] - target["f0_std"]))
    params = {
        "exaggeration": chosen["ex"],
        "cfg_weight": chosen["cfg"],
        "temperature": chosen["temp"],
        "pause_s": PAUSE_S,
    }
    log(f"Réglage retenu : {chosen['name']} -> {params}")

    # 4) Sauvegarde
    folder = voices.voice_dir(save_as)
    folder.mkdir(parents=True, exist_ok=True)
    seg = sf.read(str(ref_path))[0]
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
    summary = {"target": target, "results": results, "chosen": chosen["name"], "params": params}
    (ndir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Voix '{save_as}' sauvegardée.")
    return summary

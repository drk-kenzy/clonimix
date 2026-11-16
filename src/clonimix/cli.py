"""CLI clonimix : doctor, enroll, say, list, remove."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from . import __version__, voices


def cmd_doctor(_: argparse.Namespace) -> int:
    ok = True
    print(f"clonimix {__version__} - Python {sys.version.split()[0]}")
    for module in ("torch", "chatterbox", "librosa", "soundfile", "numpy"):
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [KO]   {module} : {exc}")
    if ok:
        from .engine import pick_device

        device = pick_device()
        print(f"  Appareil de calcul : {device}")
        if device == "cpu":
            print("  (CPU : ça marche, mais compte quelques secondes par phrase)")
    print(f"  Dossier des voix : {voices.home() / 'voices'}")
    return 0 if ok else 1


def cmd_enroll(args: argparse.Namespace) -> int:
    confirmed = args.confirm_own_voice
    if not confirmed:
        answer = input(
            "Confirmes-tu que c'est TA voix (ou que la personne a donné son accord) ? [o/N] "
        )
        confirmed = answer.strip().lower() in {"o", "oui", "y", "yes"}
    meta = voices.enroll(
        Path(args.audio), args.name, confirmed, denoise=args.denoise
    )
    print(f"Voix '{meta['name']}' enregistrée ({meta['duration_s']}s de référence).")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    from .engine import Engine
    import soundfile as sf

    reference = voices.reference_path(args.voice)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if text and getattr(args, "translate_from", None):
        from .translate import translate

        text = translate(text, args.translate_from, args.lang)
        print(f"Texte traduit : {text}")
    if not text or not text.strip():
        print("Rien à dire : donne un texte ou --file.", file=sys.stderr)
        return 2

    out = Path(args.output) if args.output else (
        Path("outputs") / f"{args.voice}_{datetime.now():%Y%m%d_%H%M%S}.wav"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = Engine(device=args.device)
    print("Chargement du modèle (le premier lancement télécharge les poids)...")
    # Priorité : options de la commande > réglages optimisés de la voix > défauts
    params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "pause_s": 0.25}
    params.update(voices.load_params(args.voice))
    for key in params:
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    signal, sr = engine.synthesize(
        text, reference, lang=args.lang, best_of=args.best_of, **params
    )
    sf.write(str(out), signal, sr)
    print(f"Audio généré : {out}")
    return 0


def _params_for(voice: str, overrides: dict) -> dict:
    """Options de la commande > réglages optimisés de la voix > défauts."""
    params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "pause_s": 0.25}
    params.update(voices.load_params(voice))
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def cmd_session(args: argparse.Namespace) -> int:
    """Charge le modèle UNE seule fois, puis génère autant de textes que voulu."""
    import itertools

    import soundfile as sf

    from .engine import Engine

    voice, best_of = args.voice, args.best_of
    voices.reference_path(voice)  # échoue tout de suite si la voix n'existe pas
    overrides = {k: getattr(args, k) for k in ("exaggeration", "cfg_weight", "temperature", "pause_s")}
    counter = itertools.count(1)

    engine = Engine(device=args.device)
    print("Chargement du modèle (une seule fois)...", flush=True)
    engine._load()

    def generate(text: str) -> None:
        out = Path(args.output_dir) / (
            f"{voice}_{datetime.now():%Y%m%d_%H%M%S}_{next(counter):02d}.wav"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        signal, sr = engine.synthesize(
            text,
            voices.reference_path(voice),
            lang=args.lang,
            best_of=best_of,
            **_params_for(voice, overrides),
        )
        sf.write(str(out), signal, sr)
        print(f"Audio généré : {out}", flush=True)

    if args.batch:  # un texte par ligne, sans interaction
        lines = [
            ln.strip()
            for ln in Path(args.batch).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for i, line in enumerate(lines, 1):
            print(f"[{i}/{len(lines)}] {line[:60]}", flush=True)
            generate(line)
        return 0

    print("Écris un texte puis Entrée. Commandes : :voix NOM  :best N  :quit", flush=True)
    while True:
        try:
            line = input("texte> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line.startswith(":voix "):
            name = line.split(None, 1)[1]
            try:
                voices.reference_path(name)
                voice = name
                print(f"Voix : {voice}", flush=True)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Erreur : {exc}", flush=True)
            continue
        if line.startswith(":best "):
            try:
                best_of = max(1, int(line.split()[1]))
                print(f"best-of : {best_of}", flush=True)
            except (ValueError, IndexError):
                print("Usage : :best 3", flush=True)
            continue
        try:
            generate(line)
        except Exception as exc:  # noqa: BLE001 - on reste dans la session
            print(f"Erreur : {exc}", flush=True)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import similarity as sim

    ve = sim.load_ve()
    refs = {Path(p).name: sim.embed(ve, sim.load_wav(p)) for p in args.against}
    print(f"{'audio':38s}" + "".join(f"{n[:22]:>24s}" for n in refs))
    for audio in args.audio:
        emb = sim.embed(ve, sim.load_wav(audio))
        row = "".join(f"{sim.cosine(emb, r):>24.3f}" for r in refs.values())
        print(f"{Path(audio).name[:38]:38s}{row}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from . import tune

    confirmed = args.confirm_own_voice
    if not confirmed:
        answer = input("Confirmes-tu que c'est TA voix ? [o/N] ")
        confirmed = answer.strip().lower() in {"o", "oui", "y", "yes"}
    tune.run(
        Path(args.audio), args.holdout or [], args.text, args.save_as, confirmed, args.lang
    )
    return 0


def cmd_natural(args: argparse.Namespace) -> int:
    from . import natural

    confirmed = args.confirm_own_voice
    if not confirmed:
        answer = input("Confirmes-tu que c'est TA voix ? [o/N] ")
        confirmed = answer.strip().lower() in {"o", "oui", "y", "yes"}
    natural.run(
        Path(args.audio), args.holdout or [], args.text, args.save_as, confirmed, args.lang
    )
    return 0


def cmd_prosody(args: argparse.Namespace) -> int:
    from . import prosody, similarity as sim

    print(f"{'audio':34s}{'f0_std':>8s}{'f0_range':>10s}{'db_std':>8s}{'pause%':>8s}{'f0_Hz':>7s}")
    for audio in args.audio:
        p = prosody.profile(sim.load_wav(audio))
        print(
            f"{Path(audio).name[:34]:34s}{p['f0_std']:>8}{p['f0_range']:>10}"
            f"{p['db_std']:>8}{p['pause_pct']:>8}{p['f0_median_hz']:>7}"
        )
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    items = voices.list_voices()
    if not items:
        print("Aucune voix. Crée-en une : clonimix enroll mon_audio.wav --name moi")
        return 0
    for meta in items:
        print(f"- {meta['name']}  ({meta['duration_s']}s, créée le {meta['created_at'][:10]})")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    voices.remove_voice(args.name)
    print(f"Voix '{args.name}' supprimée.")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .web import serve

    serve(port=args.port, device=args.device, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clonimix", description="Clone ta propre voix, en local."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Vérifie l'installation").set_defaults(func=cmd_doctor)

    p = sub.add_parser("enroll", help="Enregistre ta voix depuis un fichier audio")
    p.add_argument("audio", help="Fichier audio (wav, mp3, m4a, flac...) de 5 à 30 s")
    p.add_argument("--name", required=True, help="Nom de la voix (ex: moi)")
    p.add_argument(
        "--confirm-own-voice",
        action="store_true",
        help="Confirme que c'est ta voix (ou accord explicite de la personne)",
    )
    p.add_argument(
        "--denoise",
        action="store_true",
        help="Réduit le bruit de fond (désactivé par défaut : peut abîmer le timbre)",
    )
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("web", help="Interface web locale (enregistrer, gérer, générer)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", help="cpu, cuda ou mps (auto par défaut)")
    p.add_argument("--no-browser", action="store_true", help="N'ouvre pas le navigateur")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("say", help="Génère un audio avec une voix enregistrée")
    p.add_argument("text", nargs="?", help="Texte à dire")
    p.add_argument("--file", help="Fichier texte à lire à la place")
    p.add_argument("--voice", required=True, help="Nom de la voix")
    p.add_argument("--lang", default="fr", help="Langue (fr, en, es, de, it...)")
    p.add_argument("--translate-from", help="Langue du texte à traduire avant lecture (ex: fr)")
    p.add_argument("--exaggeration", type=float, help="Expressivité 0.25-1.0 (défaut : 0.5)")
    p.add_argument("--cfg-weight", dest="cfg_weight", type=float, help="Guidage 0-1 (défaut : 0.5)")
    p.add_argument("--temperature", type=float, help="Variabilité (défaut : 0.8)")
    p.add_argument(
        "--pause", dest="pause_s", type=float, help="Pause entre phrases en s (défaut : 0.25)"
    )
    p.add_argument(
        "--best-of", dest="best_of", type=int, default=1,
        help="Candidats par phrase, garde le plus ressemblant (plus lent, plus fidèle)",
    )
    p.add_argument("--device", help="cpu, cuda ou mps (auto par défaut)")
    p.add_argument("-o", "--output", help="Fichier de sortie .wav")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser(
        "session",
        help="Charge le modèle une fois et génère plusieurs textes (interactif ou --batch)",
    )
    p.add_argument("--voice", required=True, help="Nom de la voix")
    p.add_argument("--lang", default="fr")
    p.add_argument("--best-of", dest="best_of", type=int, default=1, help="Candidats par phrase")
    p.add_argument("--batch", help="Fichier texte : un texte par ligne, sans interaction")
    p.add_argument("--output-dir", default="outputs", help="Dossier des audios (défaut : outputs)")
    p.add_argument("--exaggeration", type=float)
    p.add_argument("--cfg-weight", dest="cfg_weight", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--pause", dest="pause_s", type=float)
    p.add_argument("--device", help="cpu, cuda ou mps (auto par défaut)")
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("tune", help="Optimise une voix (meilleur extrait + réglages)")
    p.add_argument("audio", help="Enregistrement source (20 à 30 s idéalement)")
    p.add_argument("--save-as", required=True, help="Nom de la voix optimisée")
    p.add_argument(
        "--holdout", nargs="*", help="Autres vraies voix, pour vérifier sans biais (facultatif)"
    )
    p.add_argument(
        "--text",
        default="Ce qui me plaît, c'est que tout reste chez moi, sur mon ordinateur.",
        help="Phrase de test (courte, pour aller vite)",
    )
    p.add_argument("--lang", default="fr")
    p.add_argument("--confirm-own-voice", action="store_true")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("score", help="Mesure la ressemblance d'audios avec des références")
    p.add_argument("audio", nargs="+", help="Audios à évaluer")
    p.add_argument("--against", nargs="+", required=True, help="Vraies voix de référence")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("natural", help="Rend la voix moins robotique (après 'tune')")
    p.add_argument("audio", help="Même enregistrement source que pour 'tune'")
    p.add_argument("--save-as", required=True, help="Nom de la voix résultante")
    p.add_argument("--holdout", nargs="*", help="Autres vraies voix, pour vérifier (facultatif)")
    p.add_argument(
        "--text",
        default="Ce qui me plaît, c'est que tout reste chez moi, sur mon ordinateur.",
    )
    p.add_argument("--lang", default="fr")
    p.add_argument("--confirm-own-voice", action="store_true")
    p.set_defaults(func=cmd_natural)

    p = sub.add_parser("prosody", help="Mesure la variation de hauteur/énergie (voix plate ?)")
    p.add_argument("audio", nargs="+")
    p.set_defaults(func=cmd_prosody)

    sub.add_parser("list", help="Liste les voix").set_defaults(func=cmd_list)

    p = sub.add_parser("remove", help="Supprime une voix")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (PermissionError, FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

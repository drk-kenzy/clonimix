"""Traduction locale facultative (Argos Translate), hors ligne après le premier téléchargement.

Chaque langue demande un petit modèle (environ 100 Mo) téléchargé une seule fois.
Les paires sans modèle direct passent par l'anglais.
"""

from __future__ import annotations

# Langue de Chatterbox -> code Argos quand ils diffèrent
ARGOS_CODE = {"no": "nb"}


def available() -> bool:
    try:
        import argostranslate.translate  # noqa: F401
    except ImportError:
        return False
    return True


def _code(lang: str) -> str:
    return ARGOS_CODE.get(lang, lang)


def _installed_pairs() -> set[tuple[str, str]]:
    import argostranslate.package as pkg

    return {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}


def _refresh() -> None:
    """Argos garde la liste des modèles en mémoire : on la vide après chaque installation."""
    import importlib

    import argostranslate.translate as tr

    tr.get_installed_languages.cache_clear()
    importlib.reload(tr)


def _ensure_step(src: str, dst: str, log) -> None:
    import argostranslate.package as pkg

    if (src, dst) in _installed_pairs():
        return
    log(f"Téléchargement du modèle de traduction {src} vers {dst} (une seule fois)…")
    pkg.update_package_index()
    for candidate in pkg.get_available_packages():
        if candidate.from_code == src and candidate.to_code == dst:
            pkg.install_from_path(candidate.download())
            _refresh()
            return
    raise RuntimeError(f"Aucun modèle de traduction {src} vers {dst} disponible.")


def translate(text: str, src: str, dst: str, log=print) -> str:
    """Traduit `text` de `src` vers `dst` (codes Chatterbox). Renvoie le texte tel quel si src == dst."""
    if not available():
        raise RuntimeError(
            "Traduction non installée. Lance : pip install \"clonimix[translate]\" "
            "(ou : pip install argostranslate)."
        )
    if src == dst:
        return text
    s, d = _code(src), _code(dst)
    try:
        if s != "en":
            _ensure_step(s, "en", log)
        if d != "en":
            _ensure_step("en", d, log)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Impossible de préparer la traduction {src} vers {dst} "
            f"(premier téléchargement : connexion nécessaire). Détail : {exc}"
        ) from exc
    # Le paquet garde en mémoire la liste des langues : on la relit après un téléchargement.
    import argostranslate.translate as tr

    langs = tr.get_installed_languages()
    from_lang = next((x for x in langs if x.code == s), None)
    to_lang = next((x for x in langs if x.code == d), None)
    engine = from_lang.get_translation(to_lang) if from_lang and to_lang else None
    if engine is None:
        raise RuntimeError(f"Modèle de traduction {src} vers {dst} introuvable après installation.")
    result = engine.translate(text)
    if not result.strip():
        raise RuntimeError("La traduction est vide.")
    return result

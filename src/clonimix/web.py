"""Interface web locale : enregistrer sa voix, gérer ses voix, générer de l'audio.

Serveur HTTP standard (aucune dépendance en plus), lié à 127.0.0.1 uniquement.
"""

from __future__ import annotations

import json
import queue
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import voices

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 60 * 1024 * 1024
LANGS = ["fr", "en", "es", "de", "it", "pt", "nl", "pl", "ru", "tr", "ar", "zh", "ja", "ko", "hi"]

JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[tuple[str, dict]]" = queue.Queue()
STATE = {"engine": None, "device": None}


def outputs_dir() -> Path:
    path = voices.home() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _num(data: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(hi, max(lo, value))


def _postprocess(signal, sr: int, denoise: bool, speed: float):
    import numpy as np

    if denoise:
        import noisereduce as nr

        signal = nr.reduce_noise(y=signal, sr=sr, stationary=False, prop_decrease=0.6)
    if abs(speed - 1.0) > 0.01:
        import librosa

        signal = librosa.effects.time_stretch(signal, rate=speed)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.95).astype("float32")


def _run_job(job_id: str, req: dict) -> None:
    import soundfile as sf

    from .cli import _params_for
    from .engine import Engine

    job = JOBS[job_id]

    def log(message: str) -> None:
        job["log"].append(str(message))

    try:
        if STATE["engine"] is None:
            job["status"] = "loading"
            log("Chargement du modèle (une seule fois, la première génération est plus longue)…")
            STATE["engine"] = Engine(device=STATE["device"])
            STATE["engine"]._load()
        job["status"] = "running"
        voice = req["voice"]
        params = _params_for(
            voice,
            {k: req[k] for k in ("exaggeration", "cfg_weight", "temperature", "pause_s")},
        )
        signal, sr = STATE["engine"].synthesize(
            req["text"],
            voices.reference_path(voice),
            lang=req["lang"],
            best_of=req["best_of"],
            log=log,
            **params,
        )
        signal = _postprocess(signal, sr, req["denoise"], req["speed"])
        name = f"{voice}_{datetime.now():%Y%m%d_%H%M%S}_{job_id[:4]}.wav"
        sf.write(str(outputs_dir() / name), signal, sr)
        job.update(status="done", file=name, seconds=round(len(signal) / sr, 1))
    except Exception as exc:  # noqa: BLE001 - on remonte l'erreur à l'interface
        job.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        job["finished"] = time.time()


def _worker() -> None:
    while True:
        job_id, req = QUEUE.get()
        _run_job(job_id, req)


class Handler(BaseHTTPRequestHandler):
    server_version = "Clonimix"

    def log_message(self, fmt, *args):  # silencieux
        pass

    # --- utilitaires ---
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            return self._json({"error": "Fichier introuvable."}, 404)
        self._send(200, path.read_bytes(), ctype)

    def _guard(self) -> bool:
        """Refuse les requêtes qui ne viennent pas de cette machine (anti-rebinding / CSRF)."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            self._json({"error": "Hôte refusé."}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
            self._json({"error": "Origine refusée."}, 403)
            return False
        return True

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            raise ValueError("Fichier trop volumineux.")
        return self.rfile.read(length)

    # --- GET ---
    def do_GET(self):
        if not self._guard():
            return
        url = urlparse(self.path)
        path = unquote(url.path)
        if path == "/":
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if path == "/api/voices":
            items = []
            for meta in voices.list_voices():
                meta = dict(meta)
                meta["params"] = voices.load_params(meta["name"])
                items.append(meta)
            return self._json({"voices": items, "langs": LANGS})
        m = re.fullmatch(r"/api/voice/([A-Za-z0-9_-]{1,40})/reference\.wav", path)
        if m:
            return self._file(voices.voice_dir(m.group(1)) / "reference.wav", "audio/wav")
        m = re.fullmatch(r"/api/job/([0-9a-f]{12})", path)
        if m:
            job = JOBS.get(m.group(1))
            return self._json(job or {"error": "Tâche inconnue."}, 200 if job else 404)
        m = re.fullmatch(r"/api/out/([A-Za-z0-9_.-]{1,120}\.wav)", path)
        if m:
            return self._file(outputs_dir() / m.group(1), "audio/wav")
        if path == "/api/outputs":
            files = sorted(outputs_dir().glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            return self._json({"files": [p.name for p in files[:30]]})
        self._json({"error": "Introuvable."}, 404)

    # --- POST ---
    def do_POST(self):
        if not self._guard():
            return
        url = urlparse(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        try:
            if path == "/api/enroll":
                return self._enroll(query)
            if path == "/api/say":
                return self._say()
            m = re.fullmatch(r"/api/voices/([A-Za-z0-9_-]{1,40})/delete", path)
            if m:
                voices.remove_voice(m.group(1))
                return self._json({"ok": True})
            m = re.fullmatch(r"/api/out/([A-Za-z0-9_.-]{1,120}\.wav)/delete", path)
            if m:
                (outputs_dir() / m.group(1)).unlink(missing_ok=True)
                return self._json({"ok": True})
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        self._json({"error": "Introuvable."}, 404)

    def _enroll(self, query: dict) -> None:
        name = (query.get("name") or [""])[0]
        confirmed = (query.get("confirm") or ["0"])[0] == "1"
        denoise = (query.get("denoise") or ["0"])[0] == "1"
        ext = re.sub(r"[^a-z0-9]", "", (query.get("ext") or ["wav"])[0].lower())[:5] or "wav"
        data = self._body()
        if not data:
            raise ValueError("Aucun audio reçu.")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / f"envoi.{ext}"
            source.write_bytes(data)
            meta = voices.enroll(source, name, confirmed, denoise=denoise)
        self._json({"voice": meta})

    def _say(self) -> None:
        data = json.loads(self._body() or b"{}")
        text = str(data.get("text", "")).strip()
        voice = str(data.get("voice", ""))
        if not text:
            raise ValueError("Écris un texte.")
        if len(text) > 5000:
            raise ValueError("Texte trop long (5000 caractères max par génération).")
        voices.reference_path(voice)  # vérifie que la voix existe
        lang = data.get("lang", "fr")
        if lang not in LANGS:
            raise ValueError("Langue non gérée.")
        req = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "exaggeration": _num(data, "exaggeration", 0.5, 0.0, 2.0),
            "cfg_weight": _num(data, "cfg_weight", 0.5, 0.0, 1.0),
            "temperature": _num(data, "temperature", 0.8, 0.05, 2.0),
            "pause_s": _num(data, "pause_s", 0.25, 0.0, 2.0),
            "best_of": int(_num(data, "best_of", 1, 1, 8)),
            "speed": _num(data, "speed", 1.0, 0.7, 1.3),
            "denoise": bool(data.get("denoise")),
        }
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "log": [], "created": time.time()}
        QUEUE.put((job_id, req))
        self._json({"job": job_id})


def serve(port: int = 8765, device: str | None = None, open_browser: bool = True) -> None:
    STATE["device"] = device
    threading.Thread(target=_worker, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Clonimix : interface sur {url}  (Ctrl+C pour arrêter)", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        server.server_close()

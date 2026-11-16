# Clonimix

**Clone your own voice, locally.** Open source (MIT), no cloud, no subscription, no API key.
Your voice and your audio never leave your machine.

[Version française](README.md)

Clonimix is built on [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT), a multilingual
text-to-speech model that clones a voice from a few seconds of audio. On top of it, Clonimix adds:

- a simple **CLI** (`enroll`, `say`, `session`…) and a **local web interface** (`clonimix web`);
- optional **local translation** of your text before it is spoken;
- tools to **improve the clone** (best reference window, expressiveness tuning, best-of-N selection);
- a **Claude skill**: tell Claude "clone my voice from this file" and it runs the right commands.

## What it does, and what it doesn't

| Does | Doesn't |
|---|---|
| Clone your voice from a 10 to 30 s recording | Clone someone's voice without their consent (see [Ethics](#ethics-and-consent)) |
| Read any text in your voice, in 23 languages | Guarantee an indistinguishable result: the clone is close, not perfect |
| Run 100% offline after the first model download | Run fast on an old CPU (see [Performance](#performance)) |

## Installation

Requirements: Python 3.10 or newer, about 5 GB of disk (model and dependencies). An NVIDIA GPU or Apple Silicon speeds things up a lot; otherwise it runs on CPU.

```bash
git clone https://github.com/drk-kenzy/clonimix.git
cd clonimix
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS / Linux: source .venv/bin/activate
pip install -e .
clonimix doctor
```

The first run downloads the model (a few GB), once.

## Usage

```bash
# 1. Enroll your voice (10 to 30 s of clear speech, no music)
clonimix enroll my_audio.wav --name me

# 2. Make it say whatever you want
clonimix say "Hello, this is my cloned voice." --voice me --lang en -o hello.wav

# 3. Manage your voices
clonimix list
clonimix remove me
```

Long text: `clonimix say --file text.txt --voice me`.

### Other languages and translation

23 languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh. `--lang en` sets the pronunciation: **the text must already be written in that language**.

To write in your own language and have the text spoken in another one, enable local translation (Argos Translate, offline):

```bash
pip install -e ".[translate]"
clonimix say "Bonjour, ceci est ma voix." --voice me --lang en --translate-from fr -o hello.wav
```

A small model (about 100 MB) is downloaded the first time for each language, then everything works offline. Translation goes through English as a pivot and its quality varies by language (good for European languages, rougher for others such as Swahili). Your voice keeps its timbre in another language but may keep an accent.

### Web interface (record from the microphone, manage, generate)

```bash
clonimix web
```

Opens `http://127.0.0.1:8765` in your browser: record your voice straight from the microphone (or import a file), manage your voices, type a text (with automatic translation if you want), tune expressiveness, speed and audio cleanup, then listen and download the result. The server only listens on your machine (`127.0.0.1`).

### Session mode (several texts without reloading the model)

Loading the model takes time. Session mode loads it once:

```bash
clonimix session --voice me                    # interactive: type a text, get an audio file
clonimix session --voice me --batch texts.txt  # one text per line -> one file per line
```

Interactive commands: `:voix NAME` switches voice, `:best N` sets the number of draws, `:quit` exits.

## For the best quality

The base clone is decent. These steps make it more faithful and more natural. Run them from the same folder (`natural` reads the results of `tune`):

```bash
clonimix tune my_audio.wav --save-as me_opt --confirm-own-voice      # best reference window
clonimix natural my_audio.wav --save-as me_nat --confirm-own-voice   # expressiveness and pauses
clonimix say "Your text" --voice me_nat --best-of 3 -o out.wav       # keeps the best of 3 draws
```

- `tune` and `natural` generate about twenty test sentences: expect 30 to 40 minutes on a small CPU, a few minutes on a GPU.
- `--best-of N` generates N candidates per sentence and keeps the one whose timbre is closest to your voice (N times slower).
- `--holdout other_recording.m4a` checks the result against another real take of your voice.
- `--denoise` at enrollment reduces background noise. Off by default because it can damage the timbre: use it only for really noisy recordings.

### Measuring

- `clonimix score <audios> --against <real voices>` compares timbre (cosine similarity; a realistic ceiling between two real takes of the same person is 0.84 to 0.89).
- `clonimix prosody <audios>` flags a voice that is too flat (pitch variation).

### Honest limits

- Synthesis is random: the score varies by about 0.05 between identical draws. A setting only matters if it beats that noise. Settings from `natural` rest on a few draws: indicative only.
- The measures capture timbre and pitch, **not naturalness**: your ear is the judge.
- Chatterbox only uses the first seconds of the reference (about 6 s for the text model, 10 s for the decoder): pick a representative passage, which is what `tune` does.

## Recording tips

20 to 30 s of natural speech, microphone 10-15 cm away, quiet room, normal voice (not whispered or shouted), a single speaker, no music. A reference shorter than 10 s gives a mediocre clone. Read several varied sentences (statement, question, a bit of emotion) rather than a monotone reading.

## Performance

Measured on an i7-6600U laptop (2 cores, no GPU): about 100 s for a short sentence, hence session mode and `--best-of 1` for drafts. On a GPU it is on the order of a second. If you have a GPU, Clonimix detects it (`clonimix doctor` tells you), or force it with `--device cuda` / `--device mps`.

## With Claude (built-in skill)

Copy `skills/clonimix` into `~/.claude/skills/` (or your project's `.claude/skills/`). Then tell Claude: "clone my voice from this file" or "narrate this text in my voice". It runs the right commands and asks you to confirm the voice is yours.

## Data and privacy

Your voices are stored in `~/.clonimix/voices` (change it with the `CLONIMIX_HOME` variable). Never commit them: `.gitignore` already excludes `voices/`, `outputs/` and audio files.

## Ethics and consent

- Clone **only your own voice**, or that of a person who gave explicit consent. The tool asks you to confirm when enrolling a voice.
- Generated audio carries an inaudible watermark (Perth) that identifies it as synthetic. Do not remove it.
- Impersonation, fraud and disinformation are prohibited and illegal in most countries. You are responsible for how you use the generated audio.

## Troubleshooting

- **Paths too long on Windows**: create the venv in a short folder (`C:\cv`).
- **Slow network, failing downloads**: `pip install --retries 15 --timeout 60`, and `HF_HUB_DISABLE_XET=1` if the model download crashes. Once the model is cached, `HF_HUB_OFFLINE=1` avoids any network access.
- **Perth watermarker unavailable (`None`)**: `pip install "setuptools<81"` (already required by the project).
- **`pkuseg` download error**: Clonimix only loads the Chinese segmenter for CJK text; update the repository if you still see this error.

## Layout

```
src/clonimix/        engine, voice management, CLI, tuning and measuring tools, web interface
skills/clonimix/     Claude skill
```

## License and credits

MIT, see [LICENSE](LICENSE). Built on [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI (MIT).

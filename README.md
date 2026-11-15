# Clonimix

**Clone ta propre voix, en local.** Open source (MIT), sans cloud, sans abonnement, sans clé API.
Ta voix et tes audios ne quittent jamais ta machine.

Clonimix s'appuie sur [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT), un modèle
de synthèse vocale multilingue (français inclus) qui clone une voix à partir de quelques secondes d'audio.
Il fournit en plus :

- une **CLI** simple (`enroll`, `say`, `session`…) et une **interface web locale** (`clonimix web`) ;
- des outils pour **améliorer le clone** (choix de la meilleure référence, réglage de l'expressivité, sélection du meilleur tirage) ;
- une **compétence Claude** : tu dis « clone ma voix avec ce fichier » et Claude fait le reste.

## Ce que Clonimix fait, et ne fait pas

| Fait | Ne fait pas |
|---|---|
| Cloner ta voix depuis un enregistrement de 10 à 30 s | Cloner la voix de quelqu'un sans son accord (voir [Éthique](#éthique-et-consentement)) |
| Lire n'importe quel texte avec ta voix, en français et dans d'autres langues | Garantir un résultat indiscernable : le clone est proche, pas parfait |
| Tourner 100 % hors ligne après le premier téléchargement du modèle | Fonctionner vite sur un vieux CPU (voir [Performances](#performances)) |

## Installation

Prérequis : Python 3.10 ou plus, environ 5 Go de disque (modèle et dépendances). Un GPU NVIDIA ou Apple Silicon accélère beaucoup ; sinon ça tourne sur CPU.

```bash
git clone https://github.com/drk-kenzy/clonimix.git
cd clonimix
python -m venv .venv
# Windows : .venv\Scripts\activate    |    macOS / Linux : source .venv/bin/activate
pip install -e .
clonimix doctor
```

Le premier lancement télécharge le modèle (quelques Go), une seule fois.

## Utilisation

```bash
# 1. Enregistre ta voix (10 à 30 s de parole claire, sans musique)
clonimix enroll mon_audio.wav --name moi

# 2. Fais-lui dire ce que tu veux
clonimix say "Bonjour, ceci est ma voix clonée." --voice moi -o bonjour.wav

# 3. Gère tes voix
clonimix list
clonimix remove moi
```

Texte long : `clonimix say --file texte.txt --voice moi`. Autre langue : `--lang en`, `--lang es`…

### Interface web (enregistrer au micro, gérer, générer)

```bash
clonimix web
```

Ouvre `http://127.0.0.1:8765` dans ton navigateur : enregistre ta voix directement au micro (ou importe un fichier), gère tes voix, écris un texte, règle l'expressivité, la vitesse, le nettoyage du son, puis écoute et télécharge le résultat. Le serveur n'écoute que sur ta machine (`127.0.0.1`).

### Mode session (plusieurs textes sans recharger le modèle)

Charger le modèle prend du temps. Le mode session le charge une seule fois :

```bash
clonimix session --voice moi                    # interactif : tape un texte, obtiens un audio
clonimix session --voice moi --batch textes.txt # un texte par ligne -> un fichier par ligne
```

En interactif : `:voix NOM` change de voix, `:best N` règle le nombre de tirages, `:quit` quitte.

## Pour une qualité maximale

Le clone de base est correct. Ces étapes le rendent plus fidèle et plus naturel. Lance-les depuis le même dossier (`natural` relit les résultats de `tune`) :

```bash
clonimix tune mon_audio.wav --save-as moi_opt --confirm-own-voice      # meilleure fenêtre de référence
clonimix natural mon_audio.wav --save-as moi_nat --confirm-own-voice   # expressivité et pauses
clonimix say "Ton texte" --voice moi_nat --best-of 3 -o sortie.wav     # garde le meilleur de 3 tirages
```

- `tune` et `natural` génèrent une vingtaine de phrases de test : compte 30 à 40 minutes sur un petit CPU, quelques minutes sur GPU.
- `--best-of N` génère N candidats par phrase et garde celui dont le timbre ressemble le plus à ta voix (N fois plus lent).
- `--holdout autre_enregistrement.m4a` vérifie le résultat avec une autre vraie prise de ta voix.
- `--denoise` à l'enrôlement réduit le bruit de fond. Désactivé par défaut car il peut abîmer le timbre : à réserver aux enregistrements vraiment bruités.

### Mesurer

- `clonimix score <audios> --against <vraies voix>` compare le timbre (similarité cosinus, plafond réaliste entre deux vraies prises de la même personne : 0,84 à 0,89).
- `clonimix prosody <audios>` repère une voix trop plate (variation de hauteur).

### Limites honnêtes

- La synthèse est aléatoire : le score varie d'environ 0,05 entre deux tirages identiques. Un réglage ne compte que s'il dépasse ce bruit. Les réglages issus de `natural` reposent sur quelques tirages : indicatifs.
- Les mesures captent le timbre et la hauteur, **pas le naturel** : c'est ton oreille qui juge.
- Chatterbox n'utilise que les premières secondes de la référence (environ 6 s pour le modèle de texte, 10 s pour le décodeur) : choisis un passage représentatif, c'est ce que fait `tune`.

## Conseils d'enregistrement

20 à 30 s de parole naturelle, micro à 10-15 cm, pièce calme, voix normale (ni chuchotée ni criée), un seul locuteur, aucune musique. Une référence de moins de 10 s donne un clone médiocre. Lis plusieurs phrases variées (affirmation, question, un peu d'émotion) plutôt qu'une lecture monotone.

## Performances

Mesuré sur un portable i7-6600U (2 cœurs, sans GPU) : environ 100 s pour une phrase courte, d'où le mode `session` et le `--best-of 1` pour les brouillons. Sur GPU, c'est de l'ordre de la seconde. Si tu as un GPU, Clonimix le détecte (`clonimix doctor` l'indique), ou force-le avec `--device cuda` / `--device mps`.

## Avec Claude (compétence intégrée)

Copie `skills/clonimix` dans `~/.claude/skills/` (ou dans `.claude/skills/` de ton projet). Ensuite dis à Claude : « clone ma voix avec ce fichier » ou « fais-moi une narration de ce texte avec ma voix ». Il lance les bonnes commandes, en demandant ta confirmation que la voix est la tienne.

## Données et confidentialité

Tes voix sont stockées dans `~/.clonimix/voices` (changeable avec la variable `CLONIMIX_HOME`). Ne les commit jamais : le `.gitignore` exclut déjà `voices/`, `outputs/` et les fichiers audio.

## Éthique et consentement

- Clone **uniquement ta voix**, ou celle d'une personne qui a donné son accord explicite. L'outil demande une confirmation à l'enregistrement.
- Les audios générés contiennent un filigrane inaudible (Perth) qui les identifie comme synthétiques. Ne le retire pas.
- L'usurpation d'identité, l'arnaque et la désinformation sont interdites et illégales dans la plupart des pays. Tu es responsable de l'usage que tu fais des audios générés.

## Dépannage

- **Chemins trop longs sous Windows** : crée le venv dans un dossier court (`C:\cv`).
- **Réseau lent, téléchargement qui échoue** : `pip install --retries 15 --timeout 60`, et `HF_HUB_DISABLE_XET=1` si le téléchargement du modèle plante. Une fois le modèle en cache, `HF_HUB_OFFLINE=1` évite tout accès réseau.
- **Filigrane Perth indisponible (`None`)** : `pip install "setuptools<81"` (déjà imposé par le projet).
- **Erreur de téléchargement `pkuseg`** : Clonimix ne charge le segmenteur chinois que pour du texte CJK ; mets à jour le dépôt si tu vois encore cette erreur.

## Structure

```
src/clonimix/        moteur, gestion des voix, CLI, outils de réglage et de mesure
skills/clonimix/     compétence Claude
```

## Licence et crédits

MIT, voir [LICENSE](LICENSE). Basé sur [Chatterbox](https://github.com/resemble-ai/chatterbox) de Resemble AI (MIT).

---
name: clonimix
description: Clone la voix de l'utilisateur en local et génère de l'audio avec. À utiliser quand l'utilisateur veut cloner sa voix, enregistrer un échantillon de sa voix, faire dire un texte avec sa voix, générer une narration ou un doublage avec sa propre voix.
---

# clonimix : cloner sa propre voix, en local

Tout tourne sur la machine de l'utilisateur. Aucune donnée audio ne part vers un service externe
(seuls les poids du modèle sont téléchargés une fois).

## Règle de consentement (non négociable)

Ne clone **que la voix de l'utilisateur lui-même**, ou celle d'une personne qui a donné son accord
explicite. Si on te demande de cloner une célébrité, un proche non consentant, ou d'imiter quelqu'un
pour tromper (arnaque, usurpation, fausse déclaration), refuse et explique pourquoi.
Ne passe jamais `--confirm-own-voice` à la place de l'utilisateur : demande-lui d'abord de confirmer
dans la conversation que c'est bien sa voix.

## Workflow

1. **Vérifier l'installation** : `clonimix doctor`
   - Si la commande est absente : depuis la racine du dépôt, `pip install -e .` (dans un venv).
2. **Enregistrer la voix** (une seule fois) :
   `clonimix enroll <fichier_audio> --name moi --confirm-own-voice`
   - Conseille un échantillon de 20 à 30 s, une seule voix, sans musique ni écho, micro à 10-15 cm,
     voix normale. En dessous de 10 s ou avec un enregistrement très faible, le clone est médiocre.
   - `--denoise` est désactivé par défaut : testé, il abîme le timbre sur un enregistrement propre.
     Ne l'ajoute que si l'enregistrement est vraiment bruité.
   - Si c'est trop court (< 5 s) ou bruité, demande un nouvel enregistrement.
3. **Générer de l'audio** :
   `clonimix say "Texte à dire" --voice moi --lang fr -o sortie.wav`
   - Texte long : `--file texte.txt`.
   - `--exaggeration` (0.25 à 1.0) règle l'expressivité ; 0.5 par défaut.
   - **Qualité maximale** (sur demande, c'est long) : depuis le même dossier,
     `clonimix tune <audio> --save-as moi_opt --confirm-own-voice`, puis
     `clonimix natural <audio> --save-as moi_nat --confirm-own-voice`, puis
     `clonimix say "..." --voice moi_nat --best-of 3`.
     Prévenir l'utilisateur : 30 à 40 min sur CPU pour tune + natural, et `--best-of 3` triple le temps.
4. **Mesurer** (facultatif) : `clonimix score <audios> --against <vraies voix>` compare le timbre,
   `clonimix prosody <audios>` repère une voix trop plate. Le bruit entre deux tirages est d'environ
   0.05 : ne conclus rien d'une différence plus petite, et rappelle que l'oreille de l'utilisateur tranche.
5. **Gérer les voix** : `clonimix list`, `clonimix remove <nom>`.

## Notes pratiques

- Le premier `say` télécharge le modèle (plusieurs Go) : prévenir l'utilisateur.
- Sans GPU, la génération marche sur CPU mais prend quelques secondes à dizaines de secondes par phrase.
- Les audios générés portent un filigrane inaudible qui permet de les identifier comme synthétiques ;
  ne cherche pas à le retirer.
- Les voix sont stockées dans `~/.clonimix/voices` (ou `$CLONIMIX_HOME`). Ne les commit jamais.
- Langues : fr, en, es, de, it, pt, nl, pl, ru, tr, ja, ko, zh, ar, hi, et d'autres.

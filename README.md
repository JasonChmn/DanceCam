# WebcamDance

Jeu de danse à la webcam, de 1 à 4 joueurs, sans manette ni capteur : on danse face à l'écran en imitant
le ou les danseurs d'une vidéo, et le jeu note chaque mouvement.

N'importe quelle vidéo de chorégraphie peut devenir une chanson jouable : dance practice K-pop, clip,
tutoriel, cours de danse… Il suffit d'un lien (vidéo ou playlist YouTube) :
- **Import** : téléchargement, détection automatique du début et de la fin de la danse, repérage des
  danseurs « coachs » (1 à 4) et extraction hors ligne de leur squelette image par image avec YOLO-pose,
  recadrage de la vidéo sur les coachs.
- **En jeu** : la webcam est analysée en temps réel (YOLO-pose sur GPU), chaque joueur est suivi et associé
  à un coach, puis comparé à lui par l'orientation des membres, avec une tolérance de décalage et un poids
  plus fort sur les membres en mouvement. Chaque mouvement est jugé de PERFECT à MISS.

Aucune vidéo n'est fournie avec le projet : chacun importe les siennes.

```
cd ~/Documents/perso/danceCam/webcamdance

.venv/bin/python -m webcamdance                                        # lancer le jeu (--fullscreen)
.venv/bin/python -m webcamdance.import_list liste_chansons_link.txt     # importer la liste (3 en parallèle, 1 si GPU < 8 Go ; --jobs N)
.venv/bin/python -m webcamdance.import_list --retry-failed             # retenter les échecs
.venv/bin/python -m webcamdance.import_song "https://www.youtube.com/watch?v=..."   # importer une vidéo
```
Windows : `.venv\Scripts\python` au lieu de `.venv/bin/python`.

## Installation (Python 3.11)
```
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

## Windows
1. Pilote NVIDIA à jour.
2. Copier le dossier du projet (sans `.venv/`), avec `songs/` pour garder les chansons importées
   (clé USB exFAT ou Syncthing ; `songs/_cache` inutile).
3. Double-cliquer `installer.bat` : installe uv, Python 3.11, PyTorch CUDA et les dépendances, vérifie le
   GPU, crée le raccourci WebcamDance sur le Bureau, propose les diagnostics webcam / GPU.
4. Jouer : `jouer.bat` ou le raccourci.

Sur 6 Go de VRAM, `import_list` fait 1 import à la fois par défaut. TensorRT : exporter le `.engine` sur
la machine qui joue (lié au GPU), après `pip install tensorrt`.

---
---

## Plus de détails

Jeu de danse webcam, 1–4 joueurs. Cible : Windows + RTX 3060 6 Go (développé et testé sous Linux).
ffmpeg est fourni par `imageio-ffmpeg` (rien à installer). Torch : `cu124` suffit hors GPU RTX 50xx.

### Jouer
```
python -m webcamdance --fullscreen         # ou F11 en jeu ; --window 1920x1080, --fps 60
```
Menu : ↑/↓ choisir · Entrée jouer · 1-4 joueurs · A ajouter (coller un ou plusieurs liens / une
playlist, Ctrl+V) · Suppr supprimer · Échap quitter.
En jeu : Espace pause · S squelettes coach · ←/→ changer de coach (calibration) · Échap retour.

### Importer
`import_list` : déjà importées sautées, liste consommée au fur et à mesure, journal dans `logs/`,
échecs dans `logs/import_echecs.txt`. Options : `--jobs N` (imports simultanés, 3 par défaut),
`--max-downloads N` (téléchargements YouTube simultanés, 2 par défaut), `--keep-list`,
`--cookies-from-browser firefox` (si YouTube bloque, anti-robot).
```
python -m webcamdance.import_song URL1 URL2 ...                                   # lot (déjà importées sautées)
python -m webcamdance.import_song --links-file liens.txt                          # un lien par ligne
python -m webcamdance.import_song --playlist "https://www.youtube.com/playlist?list=..."
```
Options `import_song` : `--start/--end` (bornes de la danse), `--no-trim`, `--coaches N`, `--keep-source`.
Stockage : vidéo recadrée sur le bandeau des coachs, 720p H.264 (~30–40 Mo par chanson).

### Outils
- `tools/set_band.py songs/X --band Y0 Y1 --preview` : resserrer le bandeau affiché.
- `tools/debug_overlay.py songs/X` : vidéo de contrôle avec squelettes coachs.
- `tools/diag_webcam.py`, `tools/diag_inference.py`, `tools/diag_av.py` : diagnostics par brique.
- `tools/bot_play.py songs/X [--players N --isolate --still --cam-song songs/Y]` : partie jouée par un
  bot (la « webcam » rejoue la vidéo), pour tester sans danser.
- `tools/analyze_log.py logs/<partie>.jsonl` : fps, latence, scores et décalages par move.

### Réglages (webcamdance/config.py)
- `cam_latency` / `--latency-ms` : si les jugements semblent en retard/avance.
- `judgements` : seuils calés sur le bot (imitation parfaite ≈ 94 % PERFECT, danse fausse ≈ 74 % MISS).
- TensorRT : `yolo export model=yolo26s-pose.pt format=engine half=True imgsz=640` puis `--model yolo26s-pose.engine`.

### Tests
```
python -m pytest tests
```

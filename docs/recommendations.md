# VisoMaster — Préconisations et améliorations (sept. 2025)

Ce document synthétise des améliorations inspirées par FaceFusion et l'ONNX Model Zoo, adaptées à l’architecture actuelle de VisoMaster.

## Objectifs
- Gagner en performances temps réel (latence, throughput)
- Stabiliser la qualité visuelle (détection/alignement, blends, restaurations)
- Simplifier l’UX (install, presets, gestion modèles)
- Durcir la maintenance (tests, bench, logs, reproductibilité)

---

## Quick wins (faible risque, impact rapide)

1) ORT I/O Binding systématique et sorties nommées
- Constat: les swappers/détecteurs utilisent déjà `io_binding` (bien ✅), mais quelques fichiers sortent encore les outputs par index.
- Action: uniformiser des noms d’outputs pour éviter la fragilité par index (ex. `ghostfaceswap_model` avec "1165", "1549"). Ajouter une petite map nom-symbolique → nom réel.
- Bénéfice: robustesse lors de changements de graphes / versions.

2) Providers et options par modèle
- Constat: `ModelsProcessor` met un provider global. Certains modèles (détection vs upscalers) peuvent mieux tourner avec CUDA EP quand TensorRT n’apporte pas de gain.
- Action: exposer dans `models_data` un champ provider_hint (TensorRT|CUDA|CPU) et permettre `load_model(name, provider_override=None)`.
- Bénéfice: meilleurs temps de chargement, cache TRT plus pertinent.

3) Cache & Warmup
- Constat: TRT cache déjà activé (`trt_engine_cache_*`).
- Actions: après `load_model(...)`, exécuter un warmup minimal (1 run dummy) pour amortir la 1ère latence; pour ORT, activer `intra_op_num_threads` et `execution_mode=ORT_SEQUENTIAL` côté CPU si jamais fallback.

4) Détection: filtrage rotation/pose plus strict
- Constat: la détection RetinaFace fait déjà une logique rotationnelle, mais le seuil d’angle est fixe.
- Action: exposer le seuil dans l’UI (ex. ±40°/±50°) et permettre "auto" vs "manuel".

5) Logs structurés et NVTX

6) NMS réglable côté UI (nouveau)
- Ajout d'un slider « NMS Threshold (%) » dans l’onglet Detectors.
- Impact: le seuil de Non-Maximum Suppression utilisé par tous les détecteurs (RetinaFace, SCRFD, Yolov8, Yunet) est désormais ajustable depuis l’UI.
- Conseils:
  - 30–40%: plus strict, réduit les doublons mais peut supprimer des visages proches/chevauchants.
  - 45–60%: plus permissif, utile pour scènes denses mais peut garder des doublons si trop élevé.
  - Valeur par défaut: 40%.
- Constat: NVTX est déjà utilisé côté TensorRT predictor ✅.
- Action: ajouter un flag global `--profile` qui active NVTX dans les chemins ORT critiques (pré/post, memcpy éventuelles) et temps par étape; écrire un CSV léger par job.

---

## Changements déjà intégrés dans cette branche (résumé)

- I/O binding “par session” et fallback CPU lorsque nécessaire, évitant les erreurs de copie cross-device:
  - Swappers: ArcFace (Inswapper128ArcFace, SimSwapArcFace, CSCS*), InSwapper128, InStyleSwapper256, SimSwap512, GhostFace v1/v2/v3.
  - Landmarks 478 + FaceBlendShapes: choix du device par session, binding sûr, copy-back si l’output est sur un device différent de l’appelant.
  - Masques (occluder, DFL XSeg, faceparser) et Restorers: même logique device-safe + copie de sortie si besoin.
- Environnement (portable) aligné dans les deux workspaces: `scripts/setenv.bat` met `dependencies\CUDA\bin` et `dependencies\TensorRT\lib` en tête de PATH et vide `CUDA_PATH` global pour éviter les DLLs système.

Conséquence: les erreurs "There's no data transfer registered…" sont supprimées, avec exécution CUDA quand disponible et CPU propre sinon.

---

## Next (impact élevé, changements modérés)

1) TensorRT épuré + dynamic shapes
- Constat: `TensorRTPredictor` alloue des buffers max profile. Vérifier que les profils dynamiques couvrent: batch=1, résolutions clés (128/256/512) par modèle.
- Action: dans `engine_builder.onnx_to_trt`, générer 2–3 profiles compacts par modèle (min/opt/max). Exposer `precision` par modèle (fp16/fp32, opt INT8 plus tard).

2) Post-traitement de swap: seams & color transfer
- Constat: FaceFusion soigne beaucoup le blend final (pores, matching couleur, feather/r-m).
- Actions:
  - Color transfer (Reinhard ou Poisson matting optionnel) au moment du collage.
  - Bordure adaptative (feather en px proportionnels à la taille du visage) + guided filter.
- Bénéfice: fusion plus naturelle et robuste aux éclairages.

3) Améliorateurs/Restorers: versions légères
- Constat: upscaleurs `.fp16.onnx` dispo; certains pipelines peuvent être trop lourds pour 1080p.
- Action: ajouter des presets "Lite" (tile-size auto, overlap réduit, modèle x2 par défaut, skip sur frames quasi-identiques) + détecter NVRAM libre (`nvidia-smi`) pour adapter.

4) Qualité landmarks
- Constat: vous avez plusieurs détecteurs 68/106/478; exposer un mode "auto" qui choisit le meilleur suivant la taille visage détectée (ex. 203/468 pour gros plans).
- Action: routing simple dans `FaceLandmarkDetectors` + métrique de confiance pour fallback.

5) Data flow zéro-copie PyTorch ↔ ORT
6) Stabilisation pose 3D (optionnelle)
- Constat: nous stabilisons déjà le roll (in-plane). Des mouvements brusques de yaw/pitch peuvent induire des micro-oscillations géométriques.
- Action: estimer une pose 3D légère (solvePnP sur 5/68 points) et lisser yaw/pitch au même titre que le roll, avec seuils/hystérésis indépendants. Appliquer la rotation stabilisée au SimilarityTransform (en gardant la recomposition de translation comme fait pour le roll).
- Bénéfice: visage plus stable lors de hochements/rotation latérale modérée, sans « gélifier » l’animation.

7) Suivi multi-objet robuste (fallback tracking)
- Constat: LK fonctionne bien en court terme. Mais il dérive sur occlusions rapides / changements d’apparence.
- Action: introduire un tracker multi-objet léger (ex. KCF/CSRT) par visage comme « filet de sécurité » quand LK échoue sur > N points; garder la redétection ROI actuelle pour re-synchroniser; réinitialiser le tracker à chaque redétection valide.
- Bénéfice: meilleure continuité lors de pertes ponctuelles du détecteur.

8) Post-blend avancé (guidé par masque)
- Constat: les transitions peau/fond bénéficient d’un feather adaptatif et d’un guided filter.
- Action: ajouter un guided filter (ou bilateral) piloté par le gradient du masque; exposer « Feather Adaptatif (%) » et « Lissage Bord » dans l’UI, avec presets.
- Bénéfice: bords plus naturels, moins d’halos sur fonds texturés.

9) I/O et décodage vidéo (latence)
- Constat: la file d’attente d’images peut être la limite (CPU-bound).
- Actions: (a) Activer NVDEC via PyAV/FFmpeg si dispo, (b) Thread de prélecture/décodage, (c) Buffer circulaire configurable (ex. 16–64 frames) avec backpressure.
- Bénéfice: lissage FPS et réduction de la latence perçue.

10) Cache résultats et reprises
- Constat: répétitions d’inférences pour scrubbing/preview.
- Action: cacher per-frame les landmarks/kps/embeddings (clé = frame_index, visage_id) et invalider sur changements majeurs de paramètres.
- Bénéfice: navigation fluide et comparaisons rapides.

11) Presets UX « Fast / Balanced / Quality » (étendu)
- Action: relier ces presets à plusieurs axes: detector score, max faces, upscale type/tile, blend feather, restorer on/off, stabilisation.
- Bénéfice: sélection rapide adaptée aux contraintes VRAM/qualité.

12) Télémétrie locale et traces
- Action: option opt-in pour journaliser latences par étape, nb fallbacks LK/ROI, échecs TRT→CUDA→CPU; export CSV/JSON par session.
- Bénéfice: diagnostics et régressions plus simples.

13) Tests headless ciblés
- Action: trois images (0/1/2 visages), 10 frames vidéo courte; asserts sur: nb visages, shape sorties, non-crash lors de fallback LK, stabilité roll avec seuil (écart-type < X à preset Medium).
- Bénéfice: garde-fous avant release.
- Constat: déjà très bien avancé grâce à `io_binding`. 
- Action: vérifier l’alignement mémoire (contigu, dtype) et bannir les copies CPU inutiles dans quelques chemins (ex. colorizers, masks) en restant en torch.cuda quand possible.

---

## Changements de code ciblés (avec fichiers)

1) Sorties symboliques pour GhostFace
- Fichier: `app/processors/models_data.py`
- Ajouter un dict de mapping pour les noeuds de sortie ONNX:
  - `onnx_output_names = { 'GhostFacev1': '781', 'GhostFacev2': '1165', 'GhostFacev3': '1549' }`
- Fichier: `app/processors/face_swappers.py`
  - Dans `run_swapper_ghostface`, remplacer les littéraux "1165"/"1549" par une recherche `models_data.onnx_output_names[model_name]`.
- Bénéfice: moins de maintenance si le graph change.

2) Provider hint par modèle + auto-fallback
- Fichier: `app/processors/models_data.py`
  - Étendre les entrées de `models_list` avec `provider_hint: 'Auto'|'TensorRT'|'CUDA'|'CPU'` et `precision_hint: 'fp16'|'fp32'` (optionnel).
- Fichier: `app/processors/models_processor.py`
  - Dans `load_model`, si `provider_override` est None alors utiliser `provider_hint`. Si `TensorRT` indisponible ou échoue la 1ère fois, mémoriser un flag pour désactiver TRT et repasser à CUDA (puis CPU en dernier recours).
  - Conserver `session.get_providers()` pour décider du `run_device` côté appelant (déjà implémenté dans les modules ci-dessus).
- Bénéfice: démarrage plus fiable, moins de bruit TRT, meilleures perfs selon modèle.

3) Warmup après chargement
- Fichier: `app/processors/models_processor.py`
  - Après `load_model(name)`, exécuter un run de chauffe minimal:
    - Créer des tenseurs d’entrée au bon shape/dtype, les binder sur le `run_device` détecté.
    - Lancer `run_with_iobinding` une fois et ignorer le résultat.
- Bénéfice: amortit la 1ère latence (CUDA kernels JIT, épinglage buffers, build TRT côté EP si applicable).

4) Bench rapide intégré
- Fichier (nouveau): `tools/benchmark.py`
  - Charger 2–3 images de `debug/` et mesurer les latences: détection → landmarks → embedding → swap → blend → restore.
  - Sortir un JSON minimal (temps par étape, fps estimé, pic VRAM si dispo).
- Bénéfice: suivi de régression simple avant release.

---

## Stretch (R&D, gains majeurs)

1) Quantization INT8 (sélective)
- Inspiration ONNX Model Zoo: Intel Neural Compressor.
- Action: tenter INT8 sur détecteurs (Retina/SCRFD) et upscalers (BSRGANx2/x4) avec calibration jeu interne. Conserver fallback fp16.
- Bénéfice: 20–40% de perf en plus selon GPU/EP.

2) TensorRT-LLM pour blocs UNet volumineux
- Action: pour modèles UNet lourds (ghost_unet_*), générer engines TRT avec tactic sources réduites + fuse activé. Étudier `ORT-TensorRT EP` vs engines dédiés selon temps de build/latence.

3) Pipeline multi-threads/streams
- Constat: `TensorRTPredictor` gère pool contexts ✅.
- Action: planifier pipeline: stream A (détection + landmarks), stream B (embedding), stream C (swap + blend + restorer). Synchronisations précises pour overlapp.

4) Benchmarks intégrés
- Action: ajouter une commande `python main.py --benchmark` qui rejoue un lot (ex. dossier debug/frames_*.png) et sort un JSON: latences par étape, fps, VRAM.

---

## Modèles ONNX recommandés (Zoo ou SOTA connexe)
- Détection visage: UltraFace (edge), SCRFD (déjà), Yunet (déjà) — garder un fallback très léger pour CPU-only.
- Embedding: ArcFace variations; essayer backbones GhostFaceNet/IR-SE pour compromis perf/qualité.
- Restoration: GFPGANv1.4 (déjà), CodeFormer (déjà). Ajouter un mode "auto-restorer" basé sur score de netteté visage.
- Super-Resolution: Efficient sub-pixel SR; garder BSRGAN et variantes fp16, prévoir un preset rapide.

---

## Changements de code ciblés (suggestions)

1) Symbolic outputs pour GhostFace
- Ajouter dans `models_data.py` un dict `onnx_output_names = { 'GhostFacev2': '1165', 'GhostFacev3': '1549', ... }` et utiliser ces noms via lookup, plutôt que des littéraux dans `run_swapper_ghostface`.

2) Provider hint par modèle
- Étendre `models_list` avec `provider_hint` et `precision_hint`. Modifier `load_model` et `switch_providers_priority` pour utiliser ces hints quand `provider_name` est "Auto".

3) Warmup
- Après chaque `load_model`, exécuter un run avec tenseurs vides: utile pour `CUDAExecutionProvider` (JIT kernels) et `TRT` (build engine via EP).

4) Bench et profiling
- Ajouter `tools/benchmark.py` pour chronométrer: détection → landmarks → embedding → swap → blend → restore.

5) Stabilisation yaw/pitch
- Fichiers: `app/processors/utils/faceutil.py`, `app/processors/workers/frame_worker.py`
- Ajouter solvePnP (OpenCV) pour estimer yaw/pitch/roll à partir de 5 ou 68 points; lisser yaw/pitch avec OneEuro/hystérésis; étendre `get_face_similarity_tform(..., yaw_override=None, pitch_override=None)` pour appliquer une rotation limitée (approx 2D via petite-angle ou warp 3D léger si dispo).
- Garder le recalcul de translation pour préserver l’alignement Y quand overrides actifs.

6) Fallback trackers
- Fichier: `frame_worker.py`
- Ajouter un tracker CSRT/KCF par visage quand LK perd > 40% des points; utiliser le bbox tracké pour re-détecter dans l’ROI (déjà implémenté), puis réinitialiser LK.

7) Blend guidé
- Fichiers: `faceutil.py` (post-blend), `swapper_layout_data.py` (UI)
- Implémenter guided/bilateral sur la zone de bord; exposer « Feather Adaptatif (%) », « Lissage Bord (%) » avec presets dans l’UI.

8) File d’images et NVDEC
- Fichiers: `video_processor.py` + `dependencies/ffmpeg.exe`
- Thread de décodage, file de frames, option NVDEC; métriques de remplissage pour backpressure.

---

## UX
- Presets de qualité: Fast / Balanced / Quality (agissent sur detectors, tile-size, restorer, post-blend).
- Gestion des modèles: onglet listant modèles absents + bouton de download (déjà partiellement présent via `download_file`).
- Messages VRAM: afficher estimations avant lancement (720p/1080p/4k) selon GPU détecté.
 - Guides intégrés: info-bulles enrichies pour Stabilisation Rotation et Fallback Détection (quand activer, valeurs typiques: seuil roll 3°, preset Medium; fallback 3–5 frames, ROI padding 25–40%).

---

## Maintenance
- Tests: smoke tests headless sur 3 images (aucune face / 1 face / 2 faces). Assertions sur shapes de sorties.
- CI locale (optionnelle): script PowerShell qui lance les smoke tests et génère artefacts (JSON bench, logs).
- Verrouillage versions: `requirements_cuXXX.txt` sont présents; ajouter un `pip-tools` ou `uv` lock si possible.
 - Lint/Type: activer mypy partiel et ruff, avec règles tolérantes au début puis resserrer.

---

## Plan d’adoption
- Semaine 1: Quick wins + symbolic outputs + warmup + provider hints
- Semaine 2: Post-blend avancé + presets UX + bench script
- Semaine 3–4: Profiles TRT dynamiques + INT8 POC sur 1–2 modèles

---

Si vous voulez, je peux implémenter en priorité la cartographie des sorties ONNX et le warmup des modèles (peu risqué et immédiatement utile).

---

## Dépannage — ONNX Runtime et périphériques (CPU/GPU)

Symptômes observés dans certains environnements Windows:
- UserWarning: "Specified provider 'CUDAExecutionProvider' is not in available provider names. Available providers: 'AzureExecutionProvider, CPUExecutionProvider'"
- RuntimeError: "There's no data transfer registered for copying tensors from DeviceType:1 (CUDA) to DeviceType:0 (CPU)" lors du `io_binding`.
- ImportError: "No module named 'yaml'" (dépendance facultative de `huggingface_hub`).

Ce que fait désormais VisoMaster (automatique):
- Détection des providers ONNX Runtime disponibles au démarrage. Si CUDA/TensorRT ne sont pas présents, VisoMaster bascule proprement en CPU (allocation des tenseurs et `io_binding` côté CPU pour éviter tout transfert interdit). Vous verrez un message: "GPU providers not available... Falling back to CPUExecutionProvider".

Actions recommandées si vous voulez activer le GPU:
1) Vérifier le GPU et les drivers NVIDIA (GeForce/Studio) à jour.
2) Installer la version GPU d'ONNX Runtime correspondant à votre CUDA:
  - CUDA 11.x: onnxruntime-gpu (cu116/cu118)
  - CUDA 12.x: onnxruntime-gpu (cu121/cu122/cu124)
3) S'assurer que `cudnn` et (optionnel) TensorRT sont installés et visibles dans le PATH.

Si vous utilisez l'environnement Python portable fourni, lancez via `Start_Portable.bat` pour éviter de prendre un Python système. Sinon, installez les paquets requis dans votre venv:
- PyYAML (corrige l'erreur "No module named 'yaml'")
- onnxruntime-gpu (si vous voulez le support CUDA)

Note: même sans GPU disponible, VisoMaster fonctionnera désormais en mode CPU sans lever l'erreur de `io_binding`.

---

## Warmup & Providers — mode d’emploi rapide

Objectif: réduire la latence du premier run et détecter tôt les soucis de providers.

- Lancer un warmup global:

```powershell
& 'e:\projects\VisoMaster\dependencies\Python\python.exe' 'e:\projects\VisoMaster\tools\warmup.py'
```

- Cibler certains modules seulement (ex. FaceParser et Landmarks 478):

```powershell
& 'e:\projects\VisoMaster\dependencies\Python\python.exe' 'e:\projects\VisoMaster\tools\warmup.py' faceparser landmarks478
```

## Guide éclair — réglages Stabilisation/Detectors
- Rotation Stabilization
  - Quand l’activer: scènes à micro-rotations visibles (handheld, léger tremblement).
  - Preset: Medium par défaut; High si jitter fort mais attention à l’effet « gomme » des expressions rapides.
  - Seuil roll: 3° conseillé; 2° si portraits serrés; 4–5° si mouvements plus amples, pour ne pas surcontraindre.
- Detection Fallback
  - Fallback Frames: 3–5 pour combler les trous brefs; 0 pour désactiver complètement.
  - ROI Re-detect: laisser activé; Padding 25–40% (35% par défaut) selon vitesse/zoom du sujet.

- Hints par modèle: `FaceParser` est orienté CUDA→CPU par défaut pour éviter des échecs TRT. Vous pouvez étendre cette logique dans `ModelsProcessor.provider_hints` si d’autres modèles sont mieux en CUDA.

- Fallback runtime TensorRT: si une erreur d’"enqueue" TRT apparaît, VisoMaster désactive automatiquement TRT pour la session, recharge le modèle en CUDA→CPU et relance l’inférence, sans intervention manuelle.

- Bonnes pratiques:
  - Lancer le warmup après un changement d’environnement (mise à jour CUDA/TensorRT/OnnxRuntime) ou après un téléchargement de modèles.
  - Conserver `provider_name` sur "Auto" si vous souhaitez que VisoMaster choisisse intelligemment (TRT si stable, sinon CUDA puis CPU).

---

## Annexes — commandes utiles (Windows PowerShell)

- D démarrage (environnement portable recommandé):

```powershell
./Start_Portable.bat
```

- Vérifier l’environnement ONNX Runtime / CUDA / Torch:

```powershell
dependencies/Python/python.exe tools/check_env.py
```

- Option projetée pour calmer TensorRT (si bruyant/inutilisable):
  - Ajouter un flag dans la config (ex.: `provider_name = "Auto"`) et désactivation automatique de TRT après le premier échec de build/chargement.
  - Doc utilisateur: "TRT désactivé automatiquement, CUDA EP utilisé à la place".

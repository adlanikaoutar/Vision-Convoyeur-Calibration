#  Vision-Convoyeur-Calibration

Système de Vision par Ordinateur industriel pour convoyeur : mesure automatique des bandes métalliques avec correction Fish-Eye, calibration image/métrique adaptative et seuillage de précision.

##  Fonctionnalités Principales

- **Correction de distorsion Fish-Eye** : Redressement optique de l'image en utilisant les paramètres intrinsèques de la caméra.
- **Détection par IA (YOLOv8)** : Localisation automatique et robuste des plaques/bandes sur le convoyeur.
- **Raffinement des bords par Gradient (Sobel)** : Affinage sub-pixel des bounding boxes générées par YOLO pour une mesure industrielle de haute précision, robuste face aux reflets et parasites.
- **Calibration Métrique Adaptative** : Découpage du convoyeur en zones d'étalonnage pour calculer un ratio (Cm/Pixel) dynamique en fonction de la position de la bande dans l'image, corrigeant ainsi la perspective.
- **Outils de Calibration interactifs** : Scripts dédiés pour étalonner facilement le système à la souris.

##  Technologies Utilisées

- **Python 3.10+**
- **OpenCV (`cv2`)** : Traitement d'image, calibration, filtres Sobel.
- **Ultralytics (`YOLO`)** : Modèle de détection d'objets.
- **NumPy** : Calculs matriciels et mathématiques.
- **PyTorch** : Moteur d'inférence pour l'IA.

##  Structure du Projet

```text
├── main3.py                # Script principal (Pipeline complet : Calibration -> YOLO -> Mesure)
├── calcul_ratio1.py        # Outil : Calcul du ratio Cm/Pixel global
├── calibration_zones.py    # Outil : Étalonnage interactif des zones de perspective
├── test_cap_calb.py        # Outil : Vérification de la calibration sur image fixe
├── test1.py / test2.py     # Scripts d'expérimentation (distorsion, perspective)
├── models/                 # Dossier contenant le modèle IA entraîné (best.pt)
├── captures/               # Images brutes de test
├── captures_calibrees/     # Images après redressement optique
└── requirements.txt        # Dépendances du projet
```

##  Installation et Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/VOTRE_NOM/Vision-Convoyeur-Calibration.git
cd Vision-Convoyeur-Calibration
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
# Activer sur Windows :
venv\Scripts\activate
# Activer sur Mac/Linux :
source venv/bin/activate
```

### 3. Installer les dépendances
Installez les librairies principales (OpenCV, NumPy, Ultralytics) :
```bash
pip install opencv-python numpy ultralytics
```

### 4. Lancer l'application
Pour éviter les problèmes de liaisons dynamiques OpenCV sur Windows, lancez le script avec l'exécutable du venv :
```bash
venv\Scripts\python.exe main3.py
```
*Appuyez sur la touche `Échap` pour fermer la fenêtre de visualisation.*

##  Explication du Pipeline de Mesure

1. **Lecture** : Chargement de la capture du convoyeur.
2. **Calibration** : Application de la matrice de la caméra (`K`, `D`) pour supprimer la distorsion Fish-Eye, suivi d'une rotation de 0.6°.
3. **Détection YOLO** : Le modèle IA trouve les bandes et génère des coordonnées brutes `(x1, y1, x2, y2)`.
4. **Raffinement Sobel** : Analyse du gradient d'intensité autour des bords pour trouver le bord *réel* de la plaque, contournant les imperfections de l'IA.
5. **Mesure Adaptative** : Le centre de la bbox est calculé, le système cherche le ratio `Cm/Pixel` correspondant à sa zone géographique sur le convoyeur, et convertit les pixels en centimètres.

##  Licence
Ce projet est privé/confidentiel. Tous droits réservés.
``

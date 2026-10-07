import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# =====================================================
# CHEMINS
# =====================================================
IMG_PATH = r"captures/capture_2026-04-20_19-01-03.jpg"      # Image test
MODEL_PATH = r"models/best (9).pt"       # Ton modele entraine
DOSSIER_CALIBREES = "captures_calibrees"

# =====================================================
# PARAMETRES INDUSTRIELS (FIXES)
# =====================================================
ROTATION_ANGLE = 0.6
HFOV_DEG = 70
D = np.array([-0.15, 0.02, 0.0, 0.0], dtype=np.float64)
BALANCE = 0.2

# --- PARAMETRE PAR DEFAUT (si la bande est hors des zones) ---
CM_PAR_PIXEL_DEFAUT = 0.3350

# --- ZONES D'ETALONNAGE ADAPTATIVES ---
ZONES_ETALONNAGE = [
    {'x_start': 333, 'x_end': 567, 'ratio': 0.3459},
    {'x_start': 621, 'x_end': 860, 'ratio': 0.3389},
    {'x_start': 914, 'x_end': 1151, 'ratio': 0.3417},
    {'x_start': 1208, 'x_end': 1448, 'ratio': 0.3372},
    {'x_start': 1502, 'x_end': 1748, 'ratio': 0.3290},
    {'x_start': 1805, 'x_end': 2047, 'ratio': 0.3344},
    {'x_start': 2107, 'x_end': 2353, 'ratio': 0.3289},
    {'x_start': 2407, 'x_end': 2653, 'ratio': 0.3292},
    {'x_start': 2707, 'x_end': 2953, 'ratio': 0.3292},
    {'x_start': 3010, 'x_end': 3252, 'ratio': 0.3345},
    {'x_start': 3306, 'x_end': 3555, 'ratio': 0.3253},
    {'x_start': 3612, 'x_end': 3810, 'ratio': 0.4089},
]

# =====================================================
# PARAMETRES DU RAFFINEMENT DE BORDURE (NOUVEAU)
# =====================================================
# Marge de recherche autour de la bbox YOLO (en pixels) : zone dans laquelle
# on va chercher le "vrai" bord par gradient, au cas ou YOLO ait rate des px.
MARGE_RECHERCHE = 25

# Taille du lissage gaussien applique au profil de gradient (impair, px)
LISSAGE_PROFIL = 5

# Seuil minimal de gradient (valeur absolue, sur le profil lisse, intensite
# 0-255) pour qu'une transition soit consideree comme un "vrai" bord
# candidat. Sert a ignorer le bruit de texture du convoyeur (faibles
# variations) ; la vraie transition plaque/fond depasse largement ce seuil.
SEUIL_GRADIENT = 15.0

# Si True, sauvegarde une image de debug montrant les profils de gradient
DEBUG_PROFILS = True
DOSSIER_DEBUG = "debug_profils"


# =====================================================
# FONCTION POUR TROUVER LE BON RATIO SELON LA POSITION
# =====================================================
def get_ratio_local(x_center):
    for zone in ZONES_ETALONNAGE:
        if zone['x_start'] <= x_center <= zone['x_end']:
            return zone['ratio']
    print("  (Attention: Bande hors zone etalonnee, utilisation du ratio par defaut)")
    return CM_PAR_PIXEL_DEFAUT


# =====================================================
# FONCTION DE CALIBRATION (TAILLE ORIGINALE)
# =====================================================
def calibrer_image(image):
    h, w = image.shape[:2]

    f = w / (2 * math.tan(math.radians(HFOV_DEG / 2)))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), BALANCE, (w, h))
    undistorted = cv2.undistort(image, K, D, None, new_K)

    if roi[2] > 0 and roi[3] > 0:
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    h2, w2 = undistorted.shape[:2]
    cX, cY = w2 / 2, h2 / 2
    M_rot = cv2.getRotationMatrix2D((cX, cY), ROTATION_ANGLE, 1.0)
    cos = abs(M_rot[0, 0])
    sin = abs(M_rot[0, 1])
    new_w = int((h2 * sin) + (w2 * cos))
    new_h = int((h2 * cos) + (w2 * sin))
    M_rot[0, 2] += (new_w / 2) - cX
    M_rot[1, 2] += (new_h / 2) - cY

    calibrated = cv2.warpAffine(undistorted, M_rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return calibrated


# =====================================================
# RAFFINEMENT DES BORDS PAR PROFIL DE GRADIENT
# =====================================================
def _lisser_profil(profil, taille=LISSAGE_PROFIL):
    """Lisse un profil 1D avec une moyenne mobile pour reduire le bruit
    avant de chercher le maximum de gradient (evite de tomber sur un pic
    de bruit isole)."""
    if taille <= 1:
        return profil
    noyau = np.ones(taille, dtype=np.float64) / taille
    return np.convolve(profil, noyau, mode='same')


def _meilleur_candidat(profil, idx_yolo_relatif, signe_attendu, seuil):
    """
    Cherche, dans un profil de gradient 1D, le meilleur candidat de bord.

    Au lieu de prendre betement le maximum absolu du gradient (ce qui
    accroche n'importe quel objet parasite qui traverse la zone de
    recherche : bras, rouleau, reflet...), on :
      1. Ne garde que les positions dont le gradient a le signe attendu
         pour ce type de bord ET depasse le seuil minimal (= vraie
         transition, pas du bruit de texture).
      2. Parmi ces candidats valides, on choisit celui le PLUS PROCHE de
         la position predite par YOLO (idx_yolo_relatif), pas le plus
         fort. YOLO se trompe rarement de beaucoup -> le vrai bord est
         presque toujours le candidat valide le plus proche de sa
         prediction, meme si un parasite plus loin a un gradient plus
         fort.
      3. Si aucun candidat ne depasse le seuil, on retombe sur le
         maximum absolu classique (comportement de secours).

    signe_attendu : +1 si on attend une intensite croissante (gradient
                    positif) au bord, -1 si on attend une intensite
                    decroissante (gradient negatif).
    """
    if signe_attendu > 0:
        candidats = np.where(profil >= seuil)[0]
    else:
        candidats = np.where(profil <= -seuil)[0]

    if len(candidats) > 0:
        # Parmi les candidats valides, le plus proche de la prediction YOLO
        idx = candidats[np.argmin(np.abs(candidats - idx_yolo_relatif))]
    else:
        # Filet de securite : aucun candidat franc -> max absolu (ancien comportement)
        idx = int(np.argmax(np.abs(profil)))

    return int(idx)


def _trouver_bord_vertical(gray, y1, y2, x_centre, marge, sens, seuil=SEUIL_GRADIENT):
    """
    Cherche un bord VERTICAL (frontiere gauche ou droite de la plaque)
    en analysant le gradient horizontal (Sobel en X) moyenne sur la
    hauteur [y1:y2], dans une fenetre de recherche autour de x_centre.

    sens = 'gauche' -> transition fond->plaque en allant de gauche a
                        droite : l'intensite AUGMENTE -> gradient positif
    sens = 'droite' -> transition plaque->fond en allant de gauche a
                        droite : l'intensite DIMINUE -> gradient negatif

    On ne cherche pas le gradient maximal absolu (fragile face aux
    parasites comme un bras ou un rouleau qui traverse la zone), mais le
    candidat de bon signe le plus proche de x_centre (cf _meilleur_candidat).

    Retourne la coordonnee x affinee (int).
    """
    h, w = gray.shape[:2]
    y1c, y2c = max(0, y1), min(h, y2)
    if y2c <= y1c:
        return x_centre

    x_min = max(0, x_centre - marge)
    x_max = min(w, x_centre + marge)
    if x_max - x_min < 3:
        return x_centre

    bande = gray[y1c:y2c, x_min:x_max].astype(np.float64)

    # Gradient horizontal (Sobel X), moyenne sur toutes les lignes de la bande
    # -> profil 1D le long de l'axe X, robuste au bruit ligne par ligne.
    sobel_x = cv2.Sobel(bande, cv2.CV_64F, 1, 0, ksize=3)
    profil = np.mean(sobel_x, axis=0)
    profil = _lisser_profil(profil)

    signe_attendu = 1 if sens == 'gauche' else -1
    idx_yolo_relatif = x_centre - x_min
    idx = _meilleur_candidat(profil, idx_yolo_relatif, signe_attendu, seuil)

    x_affine = x_min + idx
    return x_affine


def _trouver_bord_horizontal(gray, x1, x2, y_centre, marge, sens, seuil=SEUIL_GRADIENT):
    """
    Equivalent de _trouver_bord_vertical mais pour les bords HORIZONTAUX
    (haut / bas de la plaque), via le gradient vertical (Sobel en Y).

    sens = 'haut' -> transition fond->plaque en descendant : intensite
                      augmente -> gradient positif
    sens = 'bas'  -> transition plaque->fond en descendant : intensite
                      diminue -> gradient negatif
    """
    h, w = gray.shape[:2]
    x1c, x2c = max(0, x1), min(w, x2)
    if x2c <= x1c:
        return y_centre

    y_min = max(0, y_centre - marge)
    y_max = min(h, y_centre + marge)
    if y_max - y_min < 3:
        return y_centre

    bande = gray[y_min:y_max, x1c:x2c].astype(np.float64)

    sobel_y = cv2.Sobel(bande, cv2.CV_64F, 0, 1, ksize=3)
    profil = np.mean(sobel_y, axis=1)
    profil = _lisser_profil(profil)

    signe_attendu = 1 if sens == 'haut' else -1
    idx_yolo_relatif = y_centre - y_min
    idx = _meilleur_candidat(profil, idx_yolo_relatif, signe_attendu, seuil)

    y_affine = y_min + idx
    return y_affine


def refine_bbox_edges(image_bgr, x1, y1, x2, y2, marge=MARGE_RECHERCHE,
                       debug=False, debug_path=None):
    """
    Affine une bbox YOLO (x1,y1,x2,y2) en recherchant les vrais bords de
    la plaque par analyse de gradient (Sobel) dans une fenetre de recherche
    "marge" pixels autour de chaque bord initial.

    Principe : la plaque metallique a une transition nette d'intensite
    avec le convoyeur (cf l'image fournie : plaque claire/uniforme vs
    convoyeur plus sombre/texture). Le gradient d'intensite est donc
    maximal exactement a la frontiere -> on cherche ce maximum au lieu
    de faire confiance aveuglement a la bbox du reseau de neurones.

    Retourne (x1_aff, y1_aff, x2_aff, y2_aff).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Petit flou pour reduire le bruit capteur avant de deriver (Sobel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h_box = y2 - y1
    w_box = x2 - x1

    # On reduit legerement la hauteur/largeur de la bande d'analyse pour
    # eviter que les coins (souvent bruites) ne polluent le profil moyen.
    marge_interne_v = max(2, int(h_box * 0.15))
    marge_interne_h = max(2, int(w_box * 0.15))

    # --- Bord gauche : on analyse une bande verticale au centre de la bbox ---
    x1_aff = _trouver_bord_vertical(
        gray, y1 + marge_interne_v, y2 - marge_interne_v, x1, marge, 'gauche'
    )
    # --- Bord droit ---
    x2_aff = _trouver_bord_vertical(
        gray, y1 + marge_interne_v, y2 - marge_interne_v, x2, marge, 'droite'
    )
    # --- Bord haut ---
    y1_aff = _trouver_bord_horizontal(
        gray, x1 + marge_interne_h, x2 - marge_interne_h, y1, marge, 'haut'
    )
    # --- Bord bas ---
    y2_aff = _trouver_bord_horizontal(
        gray, x1 + marge_interne_h, x2 - marge_interne_h, y2, marge, 'bas'
    )

    # Garde-fou : si le raffinement donne un resultat absurde (boite inversee
    # ou trop petite), on retombe sur la bbox YOLO d'origine pour ce cote.
    if x2_aff <= x1_aff + 5:
        x1_aff, x2_aff = x1, x2
    if y2_aff <= y1_aff + 5:
        y1_aff, y2_aff = y1, y2

    if debug:
        _sauver_debug_profils(gray, (x1, y1, x2, y2), (x1_aff, y1_aff, x2_aff, y2_aff),
                               marge, debug_path)

    return int(x1_aff), int(y1_aff), int(x2_aff), int(y2_aff)


def _sauver_debug_profils(gray, bbox_yolo, bbox_affine, marge, chemin):
    """Sauvegarde une image cote-a-cote montrant la bbox YOLO (rouge) et la
    bbox affinee (vert) pour verifier visuellement la qualite du raffinement."""
    if chemin is None:
        return
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = bbox_yolo
    x1a, y1a, x2a, y2a = bbox_affine
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)   # YOLO = rouge
    cv2.rectangle(vis, (x1a, y1a), (x2a, y2a), (0, 255, 0), 2)  # affine = vert
    cv2.imwrite(chemin, vis)


# =====================================================
# CHARGEMENT DU MODELE YOLO
# =====================================================
print("Chargement du modele IA...")
model = YOLO(MODEL_PATH)

# =====================================================
# TRAITEMENT DE L'IMAGE
# =====================================================
print(f"Lecture de l'image : {IMG_PATH}")
img = cv2.imread(IMG_PATH)

if img is None:
    raise FileNotFoundError("Image introuvable. Verifie le chemin dans IMG_PATH.")

# 1. Calibration
print("Calibration en cours (taille originale)...")
img_calibre = calibrer_image(img)

# 2. Sauvegarde de l'image calibree
print("Sauvegarde de l'image calibree...")
os.makedirs(DOSSIER_CALIBREES, exist_ok=True)

nom_fichier_original = os.path.basename(IMG_PATH)
nom_fichier_calibre = f"calibree_{nom_fichier_original}"
chemin_sauvegarde = os.path.join(DOSSIER_CALIBREES, nom_fichier_calibre)

cv2.imwrite(chemin_sauvegarde, img_calibre)
print(f"Image calibree sauvegardee dans : {chemin_sauvegarde}")

# 3. Detection YOLO
print("Detection de la bande...")
results = model(img_calibre, conf=0.3)

# 4. Mesure et Affichage
img_finale = img_calibre.copy()
nb_bandes = 0

if DEBUG_PROFILS:
    os.makedirs(DOSSIER_DEBUG, exist_ok=True)

for result in results:
    boxes = result.boxes
    for box in boxes:
        nb_bandes += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # --- RAFFINEMENT DES BORDS (NOUVEAU) ---
        # On affine la bbox YOLO en cherchant les vrais bords de la plaque
        # par analyse de gradient (Sobel), dans une fenetre de +/- MARGE_RECHERCHE
        # pixels autour de chaque bord detecte par le reseau de neurones.
        debug_chemin = None
        if DEBUG_PROFILS:
            debug_chemin = os.path.join(DOSSIER_DEBUG, f"debug_bande_{nb_bandes}.jpg")

        x1, y1, x2, y2 = refine_bbox_edges(
            img_calibre, x1, y1, x2, y2,
            marge=MARGE_RECHERCHE,
            debug=DEBUG_PROFILS,
            debug_path=debug_chemin
        )

        largeur_pixels = x2 - x1
        hauteur_pixels = y2 - y1

        # --- CALCUL ADAPTATIF ---
        x_center = (x1 + x2) / 2
        ratio_utilise = get_ratio_local(x_center)

        largeur_cm = largeur_pixels * ratio_utilise
        hauteur_cm = hauteur_pixels * ratio_utilise

        # Dessin du rectangle vert (bbox affinee)
        cv2.rectangle(img_finale, (x1, y1), (x2, y2), (0, 255, 0), 3)

        texte = f"{largeur_cm:.1f} cm"
        print(f"  -> Bande detectee ! Largeur : {largeur_cm:.1f} cm | Hauteur : {hauteur_cm:.1f} cm (Ratio utilise: {ratio_utilise:.4f})")

        (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(img_finale, (x1, y1 - 50), (x1 + tw + 10, y1), (0, 0, 0), -1)
        cv2.putText(img_finale, texte, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

if nb_bandes == 0:
    print("Aucune bande detectee sur cette image.")

# 5. Affichage du resultat final (fenetre ajustee)
cv2.namedWindow("Controle Industriel - Mesure Adaptative", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controle Industriel - Mesure Adaptative", 1280, 720)
cv2.imshow("Controle Industriel - Mesure Adaptative", img_finale)
print("\nAppuyez sur Echap pour quitter.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
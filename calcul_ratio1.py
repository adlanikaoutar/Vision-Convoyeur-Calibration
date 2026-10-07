import cv2
import numpy as np
import math

# =====================================================
# CHEMIN DE TON IMAGE TEST
# =====================================================
IMG_PATH = r"captures/capture_2026-04-20_19-01-03.jpg"

# =====================================================
# PARAMÈTRES CAMÉRA (LES TIENS)
# =====================================================
# NEW_WIDTH A ÉTÉ SUPPRIMÉ POUR GARDER LA TAILLE ORIGINALE
ROTATION_ANGLE = 0.6
HFOV_DEG = 70
D = np.array([-0.15, 0.02, 0.0, 0.0], dtype=np.float64)
BALANCE = 0.2

LARGEUR_REELLE_CM = 324.0 # La largeur réelle de ton convoyeur

# =====================================================
# FONCTION DE CALIBRATION (TAILLE ORIGINALE)
# =====================================================
def calibrer_image(image):
    # Pas de redimensionnement, on utilise la taille d'origine
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
# TRAITEMENT ET CLICS
# =====================================================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image introuvable.")

img_calibre = calibrer_image(img)

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            
            # Dessiner le point
            cv2.circle(img_calibre, (x, y), 5, (0, 0, 255), -1)
            
            if len(points) == 2:
                # Dessiner la ligne entre les deux points
                cv2.line(img_calibre, points[0], points[1], (0, 255, 0), 2)
                
                # Calculer la distance en pixels
                largeur_pixels = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
                
                # Calculer le ratio
                cm_par_pixel = LARGEUR_REELLE_CM / largeur_pixels
                
                print("\n============================================")
                print(f"Largeur en pixels : {largeur_pixels:.2f} px")
                print(f">>> CM_PAR_PIXEL = {cm_par_pixel:.4f} <<<")
                print("COPIE CETTE VALEUR DANS TON SCRIPT PRINCIPAL !")
                print("============================================\n")
            
            cv2.imshow("Clique les bords du convoyeur", img_calibre)

# Utilisation de WINDOW_NORMAL pour pouvoir redimensionner la fenêtre si l'image est très grande
cv2.namedWindow("Clique les bords du convoyeur", cv2.WINDOW_NORMAL)
cv2.imshow("Clique les bords du convoyeur", img_calibre)
cv2.setMouseCallback("Clique les bords du convoyeur", mouse_callback)

print("INSTRUCTIONS :")
print("1. Clique sur le bord GAUCHE du convoyeur")
print("2. Clique sur le bord DROIT du convoyeur")
print("3. Regarde la console pour ton ratio CM_PAR_PIXEL")
print("Appuie sur Echap pour quitter.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
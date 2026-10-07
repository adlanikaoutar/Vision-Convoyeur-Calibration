import cv2
import math

# =====================================================
# CHEMIN DE L'IMAGE CALIBRÉE À TESTER
# =====================================================
# Change ce nom pour correspondre à l'image présente dans ton dossier
IMG_PATH = r"captures_calibrees/calibree_capture_2026-04-20_13-46-16.jpg"

# =====================================================
# PARAMÈTRE DE MESURE
# =====================================================
# ⚠️ ATTENTION : METS ICI LA VALEUR TROUVÉE AVEC CALCUL_RATIO.PY 
# SUR L'IMAGE EN TAILLE ORIGINALE ! (ex: 0.2500 au lieu de 0.9585)
CM_PAR_PIXEL = 0.3230 

# =====================================================
# CHARGEMENT DE L'IMAGE (TAILLE ORIGINALE, PAS DE RESIZE)
# =====================================================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image introuvable : {IMG_PATH}")

print(f"Image chargée en taille originale : {img.shape[1]}x{img.shape[0]} pixels")

# Listes pour stocker les points et les lignes
current_points = []
lignes_dessinees = []

def dessiner_tout():
    """Redessine l'image + toutes les lignes validées + le point en cours"""
    temp = img.copy() # On travaille directement sur l'image originale
    
    # 1. Dessiner les lignes déjà validées
    for (p1, p2, dist_m) in lignes_dessinees:
        cv2.line(temp, p1, p2, (255, 0, 255), 3) # Ligne Violette épaisse
        milieu = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        
        texte = f"{dist_m:.2f} m"
        
        # Texte plus gros pour être visible sur l'image haute résolution
        (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(temp, (milieu[0] - 5, milieu[1] - 45), (milieu[0] + tw + 5, milieu[1] - 5), (0, 0, 0), -1)
        cv2.putText(temp, texte, (milieu[0], milieu[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

    # 2. Dessiner le premier point si on est en train de cliquer
    if len(current_points) == 1:
        cv2.circle(temp, current_points[0], 10, (0, 255, 0), -1)

    cv2.imshow("Test Calibration Haute Res", temp)

def mouse_callback(event, x, y, flags, param):
    global current_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 2:
            current_points.append((x, y))
            
            if len(current_points) == 2:
                p1, p2 = current_points[0], current_points[1]
                
                distance_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                distance_cm = distance_px * CM_PAR_PIXEL
                distance_m = distance_cm / 100.0
                
                lignes_dessinees.append((p1, p2, distance_m))
                
                print(f"Ligne tracée : {distance_m:.3f} mètres ({distance_cm:.1f} cm)")
                current_points = []
            
            dessiner_tout()

# =====================================================
# LANCEMENT DE LA FENÊTRE
# =====================================================
# 1. Créer une fenêtre redimensionnable
cv2.namedWindow("Test Calibration Haute Res", cv2.WINDOW_NORMAL)
# 2. Forcer la taille d'affichage à l'écran (l'image interne reste en HD)
cv2.resizeWindow("Test Calibration Haute Res", 1280, 720)
# 3. Afficher l'image
cv2.imshow("Test Calibration Haute Res", img)
cv2.setMouseCallback("Test Calibration Haute Res", mouse_callback)

print("=" * 50)
print("MODE VÉRIFICATION SUR IMAGE CALIBRÉE (TAILLE RÉELLE)")
print("=" * 50)
print("1. Clique sur le bord GAUCHE d'un moteur")
print("2. Clique sur le bord DROIT du moteur suivant")
print("3. Vérifie que la distance est bien ~0.81 m")
print("")
print("Touches :")
print("  C = Effacer toutes les lignes")
print("  ESC = Quitter")
print("=" * 50)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("c"):
        lignes_dessinees = []
        current_points = []
        dessiner_tout()
        print("Lignes effacées.")
        
    elif key == 27:
        break

cv2.destroyAllWindows()
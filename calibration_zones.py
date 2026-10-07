import cv2
import math
import os

# =====================================================
# CHEMIN DE L'IMAGE CALIBRÉE À TESTER
# =====================================================
IMG_PATH = r"captures_calibrees/calibree_capture_2026-04-20_19-01-03.jpg"
DISTANCE_REELLE_CM = 81.0  # La distance entre les moteurs

# =====================================================
# CHARGEMENT DE L'IMAGE
# =====================================================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image introuvable : {IMG_PATH}")

current_points = []
zones = [] # Stockera les zones et leurs ratios

def dessiner_tout():
    temp = img.copy()
    
    # Dessiner les zones déjà validées
    for zone in zones:
        p1, p2 = zone['p1'], zone['p2']
        cv2.line(temp, p1, p2, (255, 0, 255), 3)
        # Afficher les limites de la zone en X
        cv2.line(temp, (zone['x_start'], 0), (zone['x_start'], img.shape[0]), (0, 255, 255), 1)
        cv2.line(temp, (zone['x_end'], 0), (zone['x_end'], img.shape[0]), (0, 255, 255), 1)
        
        milieu = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        texte = f"{zone['ratio']:.4f} cm/px"
        cv2.putText(temp, texte, (milieu[0] - 50, milieu[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if len(current_points) == 1:
        cv2.circle(temp, current_points[0], 10, (0, 0, 255), -1)

    cv2.imshow("Etalonnage Zones", temp)
    return temp # On retourne l'image pour pouvoir la sauvegarder

def mouse_callback(event, x, y, flags, param):
    global current_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 2:
            current_points.append((x, y))
            
            if len(current_points) == 2:
                p1, p2 = current_points[0], current_points[1]
                
                # Calcul de la distance en pixels
                distance_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                # Calcul du ratio local
                ratio_local = DISTANCE_REELLE_CM / distance_px
                
                # Définir les limites en X de cette zone
                x_start = min(p1[0], p2[0])
                x_end = max(p1[0], p2[0])
                
                # Sauvegarder la zone
                zones.append({
                    'x_start': x_start,
                    'x_end': x_end,
                    'ratio': ratio_local,
                    'p1': p1,
                    'p2': p2
                })
                
                print(f"Zone ajoutee ! De X={x_start} a X={x_end} -> Ratio = {ratio_local:.4f} cm/px")
                current_points = []
            
            dessiner_tout()

def generer_code_et_sauvegarder():
    # 1. Récupérer l'image avec les dessins
    img_avec_lignes = dessiner_tout()
    
    # 2. Sauvegarder l'image
    nom_sauvegarde = "verification_etalonnage_zones.jpg"
    cv2.imwrite(nom_sauvegarde, img_avec_lignes)
    print(f"\n✅ Image avec les lignes sauvegardee sous : {nom_sauvegarde}")
    
    # 3. Générer le code pour la console
    print("\n" + "="*60)
    print("COPIE-COLLE CE CODE DANS TON SCRIPT main.py :")
    print("="*60)
    print("ZONES_ETALONNAGE = [")
    for zone in zones:
        print(f"    {{'x_start': {zone['x_start']}, 'x_end': {zone['x_end']}, 'ratio': {zone['ratio']:.4f}}},")
    print("]")
    print("="*60)

# =====================================================
# LANCEMENT
# =====================================================
cv2.namedWindow("Etalonnage Zones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etalonnage Zones", 1280, 720)
cv2.imshow("Etalonnage Zones", img)
cv2.setMouseCallback("Etalonnage Zones", mouse_callback)

print("INSTRUCTIONS :")
print("1. Clique sur le bord GAUCHE d'un espace de 81 cm")
print("2. Clique sur le bord DROIT de ce meme espace")
print("3. Recommence pour les autres espaces (gauche, centre, droite...)")
print("4. Quand tu as fini, appuie sur la touche 'S' pour Sauvegarder l'image et exporter le code")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        generer_code_et_sauvegarder()
    elif key == 27:
        break

cv2.destroyAllWindows()
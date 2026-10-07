import cv2
import numpy as np
import math

# =====================================================
# CHEMIN IMAGE & PARAMÈTRES
# =====================================================
IMG_PATH = r"captures\image_fisheye_rotation.jpg"

NEW_WIDTH = 1280
DISPLAY_W = 1280
HFOV_DEG = 70

OUTPUT_NAME = "convoyeur_rectifie.jpg"
PREPROCESS_NAME = "image_fisheye_rotation.jpg"
SCREENSHOT_NAME = "capture_rectangle_test.jpg" # Nom de la capture d'écran

# =====================================================
# ROTATION AVEC CONSERVATION DES BORDS
# =====================================================
def rotate_image_keep_bounds(image, angle_deg):
    h, w = image.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

# =====================================================
# CHARGEMENT IMAGE
# =====================================================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image introuvable. Vérifie le chemin.")

ratio = NEW_WIDTH / img.shape[1]
img = cv2.resize(img, (NEW_WIDTH, int(img.shape[0] * ratio)))
h, w = img.shape[:2]

# Matrice caméra de base
f = w / (2 * math.tan(math.radians(HFOV_DEG / 2)))
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)

# =====================================================
# ÉTAPE 1 : CALIBRATION INTERACTIVE (CURSEURS)
# =====================================================
def nothing(x):
    pass

cv2.namedWindow("Etape 1 : Reglage Curseurs", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etape 1 : Reglage Curseurs", 1000, 600)

# Curseurs : K1 et K2 positifs pour dézoomer le centre. Rotation mise à 0.8 par défaut (8/10)
cv2.createTrackbar("K1 (x1000)", "Etape 1 : Reglage Curseurs", 0, 500, nothing)
cv2.createTrackbar("K2 (x1000)", "Etape 1 : Reglage Curseurs", 0, 200, nothing)
cv2.createTrackbar("Balance (x100)", "Etape 1 : Reglage Curseurs", 0, 100, nothing)
cv2.createTrackbar("Rotation (x10)", "Etape 1 : Reglage Curseurs", -10, 50, nothing)

print("ETAPE 1 : Ajuste les curseurs pour dézoomer le centre (monte K1).")
print("Appuie sur ESPACE pour passer à la sélection des 4 points.")

processed = None
prev_params = ()

while True:
    k1 = cv2.getTrackbarPos("K1 (x1000)", "Etape 1 : Reglage Curseurs") / 1000.0
    k2 = cv2.getTrackbarPos("K2 (x1000)", "Etape 1 : Reglage Curseurs") / 1000.0
    balance = cv2.getTrackbarPos("Balance (x100)", "Etape 1 : Reglage Curseurs") / 100.0
    rotation = cv2.getTrackbarPos("Rotation (x10)", "Etape 1 : Reglage Curseurs") / 10.0

    current_params = (k1, k2, balance, rotation)
    if current_params != prev_params:
        D = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), balance, (w, h))
        undistorted = cv2.undistort(img, K, D, None, new_K)
        
        if roi[2] > 0 and roi[3] > 0:
            x, y, w_roi, h_roi = roi
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        processed = rotate_image_keep_bounds(undistorted, rotation)

        # Affichage avec grille verte
        ph, pw = processed.shape[:2]
        disp_scale = 1000 / pw
        display_img = cv2.resize(processed, (1000, int(ph * disp_scale)))
        grid_img = display_img.copy()
        for x_g in range(0, 1000, 80):
            cv2.line(grid_img, (x_g, 0), (x_g, grid_img.shape[0]), (0, 255, 0), 1)
        for y_g in range(0, grid_img.shape[0], 80):
            cv2.line(grid_img, (0, y_g), (1000, y_g), (0, 255, 0), 1)
            
        cv2.imshow("Etape 1 : Reglage Curseurs", grid_img)
        prev_params = current_params

    key = cv2.waitKey(100) & 0xFF
    if key == 27:
        exit()
    elif key == ord(' '):
        break

cv2.destroyWindow("Etape 1 : Reglage Curseurs")
cv2.imwrite(PREPROCESS_NAME, processed)

# =====================================================
# ÉTAPE 2 : SÉLECTION DES 4 POINTS & TEST RECTANGLE
# =====================================================
ph, pw = processed.shape[:2]
scale = DISPLAY_W / pw
DISPLAY_H = int(ph * scale)
display_base = cv2.resize(processed, (DISPLAY_W, DISPLAY_H))

points = []

def draw_points():
    temp = display_base.copy()
    for i, p in enumerate(points):
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(temp, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(temp, str(i + 1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if len(points) > 1:
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in points], np.int32)
        cv2.polylines(temp, [pts], False, (0, 255, 255), 2)

    if len(points) == 4:
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in points], np.int32)
        # Rectangle Jaune + Diagonales Rouges pour TESTER la calibration
        cv2.polylines(temp, [pts], True, (0, 255, 255), 3)
        cv2.line(temp, pts[0], pts[2], (0, 0, 255), 2)
        cv2.line(temp, pts[1], pts[3], (0, 0, 255), 2)
        
        cv2.putText(temp, "Calibration parfaite? V = Valider | E = Retour curseurs", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # =====================================================
        # CAPTURE D'ÉCRAN AUTOMATIQUE DU RECTANGLE
        # =====================================================
        cv2.imwrite(SCREENSHOT_NAME, temp)
        print(f"\n📸 Capture d'ecran automatique : {SCREENSHOT_NAME}")

    cv2.imshow("Etape 2 : Selectionne 4 points", temp)

def correct_perspective():
    src_pts = np.float32(points)
    p1, p2, p3, p4 = src_pts

    width_top = np.linalg.norm(p2 - p1)
    width_bottom = np.linalg.norm(p3 - p4)
    out_w = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(p3 - p2)
    height_left = np.linalg.norm(p4 - p1)
    out_h = int(max(height_right, height_left))

    dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    corrected = cv2.warpPerspective(processed, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

    cv2.imwrite(OUTPUT_NAME, corrected)
    print("===================================")
    print("Image sauvegardee :", OUTPUT_NAME)
    print("Matrice homographie :\n", M)
    print("===================================")

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            points.append([x_orig, y_orig])
            draw_points()

# =====================================================
# LANCEMENT ÉTAPE 2
# =====================================================
cv2.imshow("Etape 2 : Selectionne 4 points", display_base)
cv2.setMouseCallback("Etape 2 : Selectionne 4 points", mouse_callback)

print("\nETAPE 2 : Clique les 4 points dans cet ordre :")
print("1. Haut gauche | 2. Haut droite | 3. Bas droite | 4. Bas gauche")
print("V = Valider perspective | E = Retour aux curseurs | R = Reset points")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points = []
        draw_points()

    elif key == ord("e"): # Retour à l'étape 1 si la calibration est mauvaise
        print("Retour aux curseurs de calibration...")
        points = []
        cv2.destroyWindow("Etape 2 : Selectionne 4 points")
        
        # On relance l'étape 1
        cv2.namedWindow("Etape 1 : Reglage Curseurs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Etape 1 : Reglage Curseurs", 1000, 600)
        cv2.createTrackbar("K1 (x1000)", "Etape 1 : Reglage Curseurs", 0, 500, nothing)
        cv2.createTrackbar("K2 (x1000)", "Etape 1 : Reglage Curseurs", 0, 200, nothing)
        cv2.createTrackbar("Balance (x100)", "Etape 1 : Reglage Curseurs", 0, 100, nothing)
        cv2.createTrackbar("Rotation (x10)", "Etape 1 : Reglage Curseurs", 8, 50, nothing)
        prev_params = ()
        
        while True:
            k1 = cv2.getTrackbarPos("K1 (x1000)", "Etape 1 : Reglage Curseurs") / 1000.0
            k2 = cv2.getTrackbarPos("K2 (x1000)", "Etape 1 : Reglage Curseurs") / 1000.0
            balance = cv2.getTrackbarPos("Balance (x100)", "Etape 1 : Reglage Curseurs") / 100.0
            rotation = cv2.getTrackbarPos("Rotation (x10)", "Etape 1 : Reglage Curseurs") / 10.0
            current_params = (k1, k2, balance, rotation)
            if current_params != prev_params:
                D = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), balance, (w, h))
                undistorted = cv2.undistort(img, K, D, None, new_K)
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w_roi, h_roi = roi
                    undistorted = undistorted[y:y+h_roi, x:x+w_roi]
                processed = rotate_image_keep_bounds(undistorted, rotation)
                ph, pw = processed.shape[:2]
                disp_scale = 1000 / pw
                display_img = cv2.resize(processed, (1000, int(ph * disp_scale)))
                grid_img = display_img.copy()
                for x_g in range(0, 1000, 80): cv2.line(grid_img, (x_g, 0), (x_g, grid_img.shape[0]), (0, 255, 0), 1)
                for y_g in range(0, grid_img.shape[0], 80): cv2.line(grid_img, (0, y_g), (1000, y_g), (0, 255, 0), 1)
                cv2.imshow("Etape 1 : Reglage Curseurs", grid_img)
                prev_params = current_params
            key2 = cv2.waitKey(100) & 0xFF
            if key2 == ord(' '):
                break
                
        cv2.destroyWindow("Etape 1 : Reglage Curseurs")
        cv2.imwrite(PREPROCESS_NAME, processed)
        ph, pw = processed.shape[:2]
        scale = DISPLAY_W / pw
        DISPLAY_H = int(ph * scale)
        display_base = cv2.resize(processed, (DISPLAY_W, DISPLAY_H))
        cv2.imshow("Etape 2 : Selectionne 4 points", display_base)
        cv2.setMouseCallback("Etape 2 : Selectionne 4 points", mouse_callback)

    elif key == ord("v") and len(points) == 4: # Valider la perspective
        correct_perspective()

    elif key == 27:
        break

cv2.destroyAllWindows()
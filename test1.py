import cv2
import numpy as np
import math

# =====================================================
# CHEMIN IMAGE
# =====================================================
IMG_PATH = r"captures_calibrees/calibree_capture_2026-04-20_19-01-03.jpg"

# =====================================================
# PARAMÈTRES
# =====================================================
NEW_WIDTH = 1280
DISPLAY_W = 1280
#-0.1
ROTATION_ANGLE = 0.9
HFOV_DEG = 70

# NOUVEAUX PARAMÈTRES (Positifs pour dézoomer le centre)
# K1 = -0.01 (au lieu de 0.006)0.00000001
D = np.array([-0.1, 0.02, 0.0, 0.0], dtype=np.float64)

BALANCE = 0.2

OUTPUT_NAME = "convoyeur_rectifie.jpg"
PREPROCESS_NAME = "image_fisheye_rotation.jpg"


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

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return rotated


# =====================================================
# CHARGEMENT IMAGE
# =====================================================
img = cv2.imread(IMG_PATH)

if img is None:
    raise FileNotFoundError("Image introuvable. Vérifie le chemin.")

# =====================================================
# RESIZE INITIAL
# =====================================================
ratio = NEW_WIDTH / img.shape[1]
img = cv2.resize(img, (NEW_WIDTH, int(img.shape[0] * ratio)))

h, w = img.shape[:2]

# =====================================================
# MATRICE CAMERA APPROXIMATIVE
# =====================================================
f = w / (2 * math.tan(math.radians(HFOV_DEG / 2)))

K = np.array([
    [f, 0, w / 2],
    [0, f, h / 2],
    [0, 0, 1]
], dtype=np.float64) # Changé en float64 pour la fonction undistort

# =====================================================
# CORRECTION DISTORSION (REMPLACÉ FISHEYE PAR STANDARD)
# =====================================================
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), BALANCE, (w, h))

undistorted = cv2.undistort(img, K, D, None, new_K)

# Recadrage pour enlever les bords noirs si BALANCE = 0
if roi[2] > 0 and roi[3] > 0:
    x, y, w_roi, h_roi = roi
    undistorted = undistorted[y:y+h_roi, x:x+w_roi]

# =====================================================
# ROTATION APRÈS DISTORSION
# =====================================================
processed = rotate_image_keep_bounds(undistorted, ROTATION_ANGLE)

cv2.imwrite(PREPROCESS_NAME, processed)

# =====================================================
# AFFICHAGE POUR SÉLECTION SOURIS
# =====================================================
ph, pw = processed.shape[:2]

scale = DISPLAY_W / pw
DISPLAY_H = int(ph * scale)

display_base = cv2.resize(processed, (DISPLAY_W, DISPLAY_H))

points = []


# =====================================================
# DESSIN DES POINTS
# =====================================================
def draw_points():
    temp = display_base.copy()

    for i, p in enumerate(points):
        x = int(p[0] * scale)
        y = int(p[1] * scale)

        cv2.circle(temp, (x, y), 7, (0, 0, 255), -1)

        cv2.putText(
            temp,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    if len(points) > 1:
        pts = np.array(
            [[int(p[0] * scale), int(p[1] * scale)] for p in points],
            np.int32
        )
        cv2.polylines(temp, [pts], False, (0, 255, 255), 2)

    if len(points) == 4:
        pts = np.array(
            [[int(p[0] * scale), int(p[1] * scale)] for p in points],
            np.int32
        )
        cv2.polylines(temp, [pts], True, (0, 255, 255), 3)

    cv2.imshow("Selectionne 4 points", temp)


# =====================================================
# CORRECTION PERSPECTIVE
# =====================================================
def correct_perspective():
    src_pts = np.float32(points)

    p1, p2, p3, p4 = src_pts

    width_top = np.linalg.norm(p2 - p1)
    width_bottom = np.linalg.norm(p3 - p4)
    out_w = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(p3 - p2)
    height_left = np.linalg.norm(p4 - p1)
    out_h = int(max(height_right, height_left))

    dst_pts = np.float32([
        [0, 0],
        [out_w, 0],
        [out_w, out_h],
        [0, out_h]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    corrected = cv2.warpPerspective(
        processed,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR
    )

    # =================================================
    # CORRECTION PERSPECTIVE FINALE LÉGÈRE
    # =================================================
    h2, w2 = corrected.shape[:2]

    src2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2 - 20, h2],
        [20, h2]
    ])

    dst2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ])

    M2 = cv2.getPerspectiveTransform(src2, dst2)

    corrected = cv2.warpPerspective(
        corrected,
        M2,
        (w2, h2),
        flags=cv2.INTER_LINEAR
    )

    cv2.imwrite(OUTPUT_NAME, corrected)

    corrected_display = cv2.resize(corrected, (1280, 450))
    cv2.imshow("Convoyeur rectifie", corrected_display)

    print("===================================")
    print("Image sauvegardée :", OUTPUT_NAME)
    print("Image prétraitée sauvegardée :", PREPROCESS_NAME)
    print("Points sélectionnés :")
    print(points)
    print("Matrice homographie :")
    print(M)
    print("===================================")


# =====================================================
# SOURIS
# =====================================================
def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            x_orig = int(x / scale)
            y_orig = int(y / scale)

            points.append([x_orig, y_orig])
            draw_points()

            if len(points) == 4:
                correct_perspective()


# =====================================================
# LANCEMENT
# =====================================================
cv2.imshow("Selectionne 4 points", display_base)
cv2.setMouseCallback("Selectionne 4 points", mouse_callback)

print("Clique les 4 points dans cet ordre :")
print("1. Haut gauche du convoyeur")
print("2. Haut droite du convoyeur")
print("3. Bas droite du convoyeur")
print("4. Bas gauche du convoyeur")
print("")
print("Touches :")
print("R = recommencer")
print("ESC = quitter")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points = []
        draw_points()
        print("Sélection réinitialisée.")

    elif key == 27:
        break

cv2.destroyAllWindows()
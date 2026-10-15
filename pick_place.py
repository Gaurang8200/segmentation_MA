import cv2
import numpy as np
import json
from ultralytics import YOLO
from sklearn.decomposition import PCA
import os
from datetime import datetime
import function_pool as fp

# =============================================================================j
# KONFIGURATION - Transformationsmatrizen
# =============================================================================

# Transformationsmatrix: Werkstück-Ablageposition → Roboterbasis
# Beschreibt die gewünschte Pose des Werkstücks nach dem Ablegen (in Basis-Koordinaten)
# Position in mm, Rotation als 3x3 Matrix
TARGET_TO_BASE = np.array([
    [1.0,  0.0,  0.0,   415.5],   # X-Achse + Translation in X
    [0.0,  1.0,  0.0,   256.33],   # Y-Achse + Translation in Y
    [0.0,  0.0,  1.0,       25],   # Z-Achse + Translation in Z
    [0.0,  0.0,  0.0,    1.0]    # Homogene Koordinate
], dtype=float)

# =============================================================================
# KONFIGURATION - Kamera & Modell
# =============================================================================

CAMERA_INDEX   = 0
CAMERA_BACKEND = cv2.CAP_V4L2
CAMERA_WIDTH   = 640                       # Bildbreite in Pixeln
CAMERA_HEIGHT  = 480                       # Bildhöhe in Pixeln
MODEL_PATH     = "runs/segment/train-8/weights/best.pt"
CALIB_PATH     = "output_wp2camera.json"
TRANSFORM_PATH = "output_c2f.json"          # Camera-to-Flange Transformation
ROBOT_POSES_PATH = "robot_poses.json"       # Roboter-Positionen
TCP_POSE_FOR_CALIBRATION = "p15"            # Welche Pose für pixel2robot verwendet wird
WINDOW_NAME    = "Pick & Place"
OUTPUT_DIR     = "results/pick_place"
USE_FULL_MASK  = True                       # True = alle Maskenpixel | False = nur Randpixel
WORKPIECE_Z_HEIGHT = 25                     # Angenommene Greifhöhe Zylinder
GRIPPER_Z_OFFSET = 0                        # Z-Offset des Greifers (z.B. Höhe des Sauggreifers) in mm
PCA_PIXEL_OFFSET = 30                       # Pixel-Offset für PCA-Achsenpunkte
SNAPSHOT_DIR = "results/pick_place_snapshots"  # Ordner für Snapshots
PCA_ARROW_SCALE = 60                        # Skalierung der PCA-Achsenpfeile (in Pixeln)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# =============================================================================
# HILFSFUNKTIONEN - Transformationsmatrizen
# =============================================================================

def load_camera_calibration(filepath):
    """Kamera-Matrix und Verzerrungskoeffizienten laden."""
    with open(filepath, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=float)
    dist_coeffs = np.array(calib_data['dist_coefs'], dtype=float).flatten()
    
    print(f"[INFO] Kamerakalibrierung geladen")
    print(f"[DEBUG] Kamera-Matrix:\n{camera_matrix}")
    return camera_matrix, dist_coeffs


def load_camera_to_flange_transform(filepath):
    """Transformation Flange → Kamera aus JSON laden."""
    with open(filepath, 'r') as f:
        transform_data = json.load(f)
    
    fTc = np.array(transform_data['fTc'], dtype=float)
    
    print(f"[INFO] Flange → Kamera Transformation geladen")
    print(f"[DEBUG] fTc:\n{fTc}")
    return fTc


def load_tcp_pose(filepath, pose_name="p15"):
    """TCP-Position aus robot_poses.json laden."""
    with open(filepath, 'r') as f:
        poses_data = json.load(f)
    
    if pose_name not in poses_data['Posen']:
        raise ValueError(f"Pose '{pose_name}' nicht gefunden in {filepath}")
    
    pose = poses_data['Posen'][pose_name]
    
    position = np.array([pose['x'], pose['y'], pose['z']], dtype=float)
    roll = pose['a']   # Rotation um Z
    pitch = pose['b']  # Rotation um Y
    yaw = pose['c']    # Rotation um X
    
    print(f"[DEBUG] TCP-Position '{pose_name}': x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f} mm")
    print(f"[DEBUG] TCP-Orientierung: roll={np.degrees(roll):.2f}°, pitch={np.degrees(pitch):.2f}°, yaw={np.degrees(yaw):.2f}°")
    
    R = rpy_to_rotation_matrix(roll, pitch, yaw, degrees=False)
    tcp_to_base = create_transformation_matrix(position, R)
    
    print(f"[INFO] TCP → Basis (Pose {pose_name}) geladen")
    return tcp_to_base


def create_transformation_matrix(position, rotation_matrix):
    """
    4x4 homogene Transformationsmatrix erstellen.
    
    Args:
        position: [x, y, z] in mm
        rotation_matrix: 3x3 Rotationsmatrix
    
    Returns:
        4x4 Transformationsmatrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T


def rpy_to_rotation_matrix(roll, pitch, yaw, degrees=False):
    """
    RPY zu Rotationsmatrix
    Roll = Rotation um Z, Pitch = Rotation um Y, Yaw = Rotation um X
    Reihenfolge (aktiv, intrinsisch): ZY'X'' => Rz(roll) @ Ry(pitch) @ Rx(yaw)
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(yaw), -np.sin(yaw)],
        [0, np.sin(yaw), np.cos(yaw)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    return R


def invert_transformation(T):
    """Inverse einer homogenen Transformationsmatrix berechnen."""
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv


def rotation_matrix_to_euler_angles(R):
    """Rotationsmatrix zu [Roll(Z), Pitch(Y), Yaw(X)] in Grad."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    if not singular:
        yaw_x = np.arctan2(R[2, 1], R[2, 2])
        pitch_y = np.arctan2(-R[2, 0], sy)
        roll_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw_x = np.arctan2(-R[1, 2], R[1, 1])
        pitch_y = np.arctan2(-R[2, 0], sy)
        roll_z = 0
    
    return np.degrees([roll_z, pitch_y, yaw_x])


# =============================================================================
# HILFSFUNKTIONEN - Bildverarbeitung
# =============================================================================

def get_point_cloud(mask, use_full_mask=True):
    """Punktwolke aus Segmentierungsmaske extrahieren."""
    if use_full_mask:
        yx = np.column_stack(np.where(mask > 0))
        return yx[:, ::-1].astype(float)  # (x, y)
    else:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.array([])
        return np.vstack([c.reshape(-1, 2) for c in contours]).astype(float)


# =============================================================================
# WERKSTÜCKPOSE - PCA-basiert
# =============================================================================

def detect_workpiece_pose(mask, camera_matrix, tvec, rvec, bTf_all, fTc, pose_nr):
    """
    Werkstück-Pose bestimmen via PCA.
    
    Transformationskette:
        workpiece_to_base = TCP_to_base @ camera_to_TCP @ workpiece_to_camera
    
    Args:
        mask: Binäre Segmentierungsmaske
        camera_matrix: 3x3 Kamera-Matrix
        tvec: Translationsvektoren aus Kalibrierung
        rvec: Rotationsvektoren aus Kalibrierung
        bTf_all: Liste aller Roboter-Posen (Basis → Flansch)
        fTc: Flansch → Kamera Transformation
        pose_nr: Roboter-Pose-Nummer (0-basiert für Array-Zugriff)
    
    Returns:
        tuple: (workpiece_to_base, workpiece_to_camera, pca_data)
               oder (None, None, None) bei Fehler
               pca_data = dict mit Visualisierungsinformationen
    """
    
    print(f"\n[DEBUG] ===== WERKSTÜCKPOSE-BERECHNUNG START =====")
    
    # =========================================================================
    # 1. PCA im Pixelraum
    # =========================================================================
    
    points = get_point_cloud(mask, use_full_mask=USE_FULL_MASK)
    
    if len(points) < 10:
        print("[WARN] Zu wenige Punkte für PCA")
        return None, None, None
    
    pca = PCA(n_components=2)
    pca.fit(points)
    
    print(f"[DEBUG] PCA-Eigenwerte: {pca.singular_values_}")
    
    # Zentroid P
    P = pca.mean_
    print(f"[DEBUG] Zentroid P: ({P[0]:.2f}, {P[1]:.2f}) px")
    
    # Eigenvektoren
    e_1 = pca.components_[0]  # größter Eigenwert -> X-Achse
    e_2 = pca.components_[1]  # zweiter Eigenwert -> Y-Achse
    
    print(f"[DEBUG] Eigenvektor e_1 (X): [{e_1[0]:.3f}, {e_1[1]:.3f}]")
    print(f"[DEBUG] Eigenvektor e_2 (Y): [{e_2[0]:.3f}, {e_2[1]:.3f}]")
    
    # Achsenpunkte definieren
    X_plus = P + PCA_PIXEL_OFFSET * e_1
    Y_plus = P + PCA_PIXEL_OFFSET * e_2

    print(f"Y_plus vor Korrektur: ({Y_plus[0]:.2f}, {Y_plus[1]:.2f}) px")

    print(f"[DEBUG] X^+: ({X_plus[0]:.2f}, {X_plus[1]:.2f}) px")
    print(f"[DEBUG] Y^+: ({Y_plus[0]:.2f}, {Y_plus[1]:.2f}) px")
    
    # =========================================================================
    # 2. Pixel → Kamera (pixel2robot / calc_pixel2robot)
    # =========================================================================
    
    # Pixel-Koordinaten im Format [[x], [y], [1]]
    pixel_P = np.array([[P[0]], [P[1]], [1]])
    pixel_X_plus = np.array([[X_plus[0]], [X_plus[1]], [1]])
    pixel_Y_plus = np.array([[Y_plus[0]], [Y_plus[1]], [1]])
    
    # calc_pixel2robot gibt zurück: (result, trans_result, rot_mat, xp_yp_zc, result2, Spitze_mat)
    # xp_yp_zc (4. Rückgabewert) enthält [X_cam, Y_cam, Z_cam] im Kamera-KS
    _, _, _, cam_P, _, _ = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_all, fTc, pixel_P, pose_nr, "", False)
    _, _, _, cam_X_plus, _, _ = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_all, fTc, pixel_X_plus, pose_nr, "", False)
    _, _, _, cam_Y_plus, _, _ = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_all, fTc, pixel_Y_plus, pose_nr, "", False)
    
    # cam_P, cam_X_plus, cam_Y_plus sind 3x1 Arrays
    cam_P = cam_P.flatten()
    cam_X_plus = cam_X_plus.flatten()
    cam_Y_plus = cam_Y_plus.flatten()
    
    print(f"[DEBUG] cam_P: [{cam_P[0]:.2f}, {cam_P[1]:.2f}, {cam_P[2]:.2f}] mm")
    print(f"[DEBUG] cam_X_plus: [{cam_X_plus[0]:.2f}, {cam_X_plus[1]:.2f}, {cam_X_plus[2]:.2f}] mm")
    print(f"[DEBUG] cam_Y_plus: [{cam_Y_plus[0]:.2f}, {cam_Y_plus[1]:.2f}, {cam_Y_plus[2]:.2f}] mm")
    
    # =========================================================================
    # 3. Normierte Achsenvektoren (3D, Kamera-KSYS)
    # =========================================================================
    
    x_vec = cam_X_plus - cam_P
    x_vec = x_vec / np.linalg.norm(x_vec)
    
    y_vec = cam_Y_plus - cam_P
    y_vec = y_vec / np.linalg.norm(y_vec)
    
    z_vec = np.cross(x_vec, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)

    print(f"[DEBUG] z_vec vor Korrektur: [{z_vec[0]:.3f}, {z_vec[1]:.3f}, {z_vec[2]:.3f}]")
    
    # Z-Achse soll aus Werkstück herauszeigen (zur Kamera)
    if z_vec[2] > 0:
        z_vec = -z_vec
        y_vec = -y_vec
        Y_plus = P - PCA_PIXEL_OFFSET * e_2

    print(f"Y_plus nach Korrektur: ({Y_plus[0]:.2f}, {Y_plus[1]:.2f}) px")

    print(f"[DEBUG] z_vec nach Korrektur: [{z_vec[0]:.3f}, {z_vec[1]:.3f}, {z_vec[2]:.3f}]")
    
    print(f"[DEBUG] x_vec: [{x_vec[0]:.3f}, {x_vec[1]:.3f}, {x_vec[2]:.3f}]")
    print(f"[DEBUG] y_vec: [{y_vec[0]:.3f}, {y_vec[1]:.3f}, {y_vec[2]:.3f}]")
    print(f"[DEBUG] z_vec: [{z_vec[0]:.3f}, {z_vec[1]:.3f}, {z_vec[2]:.3f}]")
    
    # =========================================================================
    # 4. Rotationsmatrix Werkstück → Kamera
    # =========================================================================
    
    R_workpiece_to_camera = np.column_stack([x_vec, y_vec, z_vec])
    workpiece_to_camera = create_transformation_matrix(cam_P, R_workpiece_to_camera)
    
    print(f"[DEBUG] workpiece_to_camera Rotation:\n{R_workpiece_to_camera}")
    print(f"[DEBUG] workpiece_to_camera Translation: [{cam_P[0]:.2f}, {cam_P[1]:.2f}, {cam_P[2]:.2f}]")
    print(f"[DEBUG] workpiece_to_camera (4x4):\n{workpiece_to_camera}")
    
    # =========================================================================
    # 5. Transformationskette: Werkstück → Basis
    # =========================================================================
    
    # fTc wird als Parameter übergeben
    camera_to_TCP = invert_transformation(fTc)
    print(f"[DEBUG] fTc (Flange → Camera):\n{fTc}")
    print(f"[DEBUG] camera_to_TCP (4x4):\n{camera_to_TCP}")
    
    # TCP → Basis
    TCP_to_base = bTf_all[pose_nr]
    print(f"[DEBUG] TCP_to_base (Pose {pose_nr}) (4x4):\n{TCP_to_base}")
    
    # Transformationskette
    # workpiece_to_base = TCP_to_base @ camera_to_TCP @ workpiece_to_camera
    workpiece_to_base = TCP_to_base @ fTc @ workpiece_to_camera
    
    # Z-Position durch feste Werkstückhöhe ersetzen
    z_original = workpiece_to_base[2, 3]
    workpiece_to_base[2, 3] = WORKPIECE_Z_HEIGHT
    print(f"[DEBUG] Z-Position korrigiert: {z_original:.2f} mm → {WORKPIECE_Z_HEIGHT} mm (WORKPIECE_Z_HEIGHT)")

    print(f"[DEBUG] -----")
    # print(f"[DEBUG] TRANSFORMATIONSKETTE: workpiece_to_base = TCP_to_base @ camera_to_TCP @ workpiece_to_camera")
    print(f"[DEBUG] TRANSFORMATIONSKETTE: workpiece_to_base = TCP_to_base @ fTc @ workpiece_to_camera")
    print(f"[DEBUG] workpiece_to_base (4x4):\n{workpiece_to_base}")
    print(f"[DEBUG] workpiece_to_base Position: [{workpiece_to_base[0,3]:.2f}, {workpiece_to_base[1,3]:.2f}, {workpiece_to_base[2,3]:.2f}] mm")
    euler_base = rotation_matrix_to_euler_angles(workpiece_to_base[:3, :3])
    print(f"[DEBUG] workpiece_to_base Orientierung: Roll={euler_base[0]:.2f}°, Pitch={euler_base[1]:.2f}°, Yaw={euler_base[2]:.2f}°")
    
    # =========================================================================
    # 6. PCA-Daten für Visualisierung
    # =========================================================================

    # Plot-Achsen aus den finalen X+/Y+-Punkten ableiten, damit der Plot immer
    # mit ggf. korrigierten Achsrichtungen (z.B. Y-Flip) uebereinstimmt.
    e_1_plot = X_plus - P
    e_2_plot = Y_plus - P
    e_1_plot = e_1_plot / np.linalg.norm(e_1_plot)
    e_2_plot = e_2_plot / np.linalg.norm(e_2_plot)

    angle_2d = np.degrees(np.arctan2(e_1_plot[1], e_1_plot[0]))
    
    pca_data = {
        'points': points,
        'center': P,
        'components': np.vstack([e_1_plot, e_2_plot]),
        'variances': pca.explained_variance_,
        'angle_2d': angle_2d,
        'X_plus': X_plus,
        'Y_plus': Y_plus
    }
    
    print(f"[DEBUG] ===== WERKSTÜCKPOSE-BERECHNUNG ENDE =====\n")
    
    return workpiece_to_base, workpiece_to_camera, pca_data


def calculate_grip_pose(workpiece_to_base):
    """
    Greifposition des TCP relativ zur Basis berechnen.
    
    Formel:
        gripTCP_to_base = workpiece_to_base @ inv(workpiece_to_gripTCP)
    
    Args:
        workpiece_to_base: 4x4 Transformationsmatrix Werkstück → Basis
    
    Returns:
        gripTCP_to_base: 4x4 Transformationsmatrix Greif-TCP → Basis
    """
    
    print(f"\n[DEBUG] ===== GREIFPOSITION-BERECHNUNG START =====")
    
    # =========================================================================
    # 1. Transformation: Werkstück → Greifer-TCP (beim Greifen)
    # =========================================================================
    
    # Rotationsmatrix: Greifer zeigt nach unten (180° um Y-Achse)
    # Hauptdiagonale: [-1, 1, -1]
    R_workpiece_to_gripTCP = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0]
    ])
    
    # Translation: TCP am Werkstück-Ursprung (+ optionaler Z-Offset)
    t_workpiece_to_gripTCP = np.array([0.0, 0.0, GRIPPER_Z_OFFSET])
    
    # 4x4 Transformationsmatrix erstellen
    workpiece_to_gripTCP = create_transformation_matrix(t_workpiece_to_gripTCP, R_workpiece_to_gripTCP)
    
    print(f"[DEBUG] workpiece_to_gripTCP Rotation (hardcodiert):\n{R_workpiece_to_gripTCP}")
    print(f"[DEBUG] workpiece_to_gripTCP Translation: [0, 0, {GRIPPER_Z_OFFSET}] mm")
    print(f"[DEBUG] workpiece_to_gripTCP (4x4):\n{workpiece_to_gripTCP}")
    
    # =========================================================================
    # 2. Inverse: gripTCP → Werkstück
    # =========================================================================
    
    gripTCP_to_workpiece = invert_transformation(workpiece_to_gripTCP)
    print(f"[DEBUG] gripTCP_to_workpiece (4x4):\n{gripTCP_to_workpiece}")
    
    # =========================================================================
    # 3. Transformationskette: Greif-TCP → Basis
    # =========================================================================
    
    gripTCP_to_base = workpiece_to_base @ gripTCP_to_workpiece
    
    print(f"[DEBUG] -----")
    print(f"[DEBUG] TRANSFORMATIONSKETTE: gripTCP_to_base = workpiece_to_base @ inv(workpiece_to_gripTCP)")
    print(f"[DEBUG] gripTCP_to_base (4x4):\n{gripTCP_to_base}")
    print(f"[DEBUG] gripTCP_to_base Position: [{gripTCP_to_base[0,3]:.2f}, {gripTCP_to_base[1,3]:.2f}, {gripTCP_to_base[2,3]:.2f}] mm")
    
    euler_grip = rotation_matrix_to_euler_angles(gripTCP_to_base[:3, :3])
    print(f"[DEBUG] gripTCP_to_base Orientierung: Roll={euler_grip[0]:.2f}°, Pitch={euler_grip[1]:.2f}°, Yaw={euler_grip[2]:.2f}°")
    
    print(f"[DEBUG] ===== GREIFPOSITION-BERECHNUNG ENDE =====\n")
    
    return gripTCP_to_base, workpiece_to_gripTCP


def calculate_place_pose(workpiece_to_gripTCP):
    """
    Ablageposition des TCP relativ zur Basis berechnen.
    
    Formel:
        placeTCP_to_base = TARGET_TO_BASE @ inv(workpiece_to_gripTCP)
    
    Args:
        workpiece_to_gripTCP: 4x4 Transformationsmatrix Werkstück → Greifer-TCP
    
    Returns:
        placeTCP_to_base: 4x4 Transformationsmatrix Ablage-TCP → Basis
    """
    
    print(f"\n[DEBUG] ===== ABLAGEPOSITION-BERECHNUNG START =====")
    
    # =========================================================================
    # 1. Ziel-Werkstückpose (TARGET_TO_BASE)
    # =========================================================================
    
    print(f"[DEBUG] TARGET_TO_BASE (Werkstück-Zielpose) (4x4):\n{TARGET_TO_BASE}")
    print(f"[DEBUG] TARGET_TO_BASE Position: [{TARGET_TO_BASE[0,3]:.2f}, {TARGET_TO_BASE[1,3]:.2f}, {TARGET_TO_BASE[2,3]:.2f}] mm")
    euler_target = rotation_matrix_to_euler_angles(TARGET_TO_BASE[:3, :3])
    print(f"[DEBUG] TARGET_TO_BASE Orientierung: Roll={euler_target[0]:.2f}°, Pitch={euler_target[1]:.2f}°, Yaw={euler_target[2]:.2f}°")
    
    # =========================================================================
    # 2. Inverse: gripTCP → Werkstück
    # =========================================================================
    
    print(f"[DEBUG] workpiece_to_gripTCP (4x4):\n{workpiece_to_gripTCP}")
    
    gripTCP_to_workpiece = invert_transformation(workpiece_to_gripTCP)
    print(f"[DEBUG] gripTCP_to_workpiece (4x4):\n{gripTCP_to_workpiece}")
    
    # =========================================================================
    # 3. Transformationskette: Ablage-TCP → Basis
    # =========================================================================
    
    placeTCP_to_base = TARGET_TO_BASE @ gripTCP_to_workpiece
    
    print(f"[DEBUG] -----")
    print(f"[DEBUG] TRANSFORMATIONSKETTE: placeTCP_to_base = TARGET_TO_BASE @ inv(workpiece_to_gripTCP)")
    print(f"[DEBUG] placeTCP_to_base (4x4):\n{placeTCP_to_base}")
    print(f"[DEBUG] placeTCP_to_base Position: [{placeTCP_to_base[0,3]:.2f}, {placeTCP_to_base[1,3]:.2f}, {placeTCP_to_base[2,3]:.2f}] mm")
    
    euler_place = rotation_matrix_to_euler_angles(placeTCP_to_base[:3, :3])
    print(f"[DEBUG] placeTCP_to_base Orientierung: Roll={euler_place[0]:.2f}°, Pitch={euler_place[1]:.2f}°, Yaw={euler_place[2]:.2f}°")
    
    print(f"[DEBUG] ===== ABLAGEPOSITION-BERECHNUNG ENDE =====\n")
    
    return placeTCP_to_base


# =============================================================================
# SNAPSHOT & VISUALISIERUNG
# =============================================================================

import matplotlib
matplotlib.use('Agg')  # Nicht-interaktives Backend
import matplotlib.pyplot as plt

def save_pca_plot(pca_data, timestamp, frame_shape):
    """PCA-Visualisierung speichern
    
    Args:
        pca_data: Dictionary mit PCA-Daten
        timestamp: Zeitstempel für Dateiname
        frame_shape: Tupel (height, width) der Kameraauflösung
    """
    
    points = pca_data['points']
    center = pca_data['center']
    components = pca_data['components']
    variances = pca_data['variances']
    angle_2d = pca_data['angle_2d']
    X_plus = pca_data['X_plus']
    Y_plus = pca_data['Y_plus']
    
    print(f"Y_plus vor Plot: ({Y_plus[0]:.2f}, {Y_plus[1]:.2f}) px")

    # Bildgröße dynamisch basierend auf Seitenverhältnis der Kamera berechnen
    height, width = frame_shape
    aspect_ratio = width / height
    fig_height = 8
    fig_width = fig_height * aspect_ratio
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Punktwolke
    ax.scatter(points[:, 0], points[:, 1], s=1, color='blue', alpha=0.4, label='Punktwolke')
    
    # Zentroid
    ax.plot(*center, marker="+", markersize=14, color="red", label=f"Zentroid (2D-Winkel: {angle_2d:.1f}°)")
    
    # Achsenpunkte
    ax.plot(*X_plus, marker="x", markersize=10, color="green", label="X^+")
    ax.plot(*Y_plus, marker="x", markersize=10, color="orange", label="Y^+")
    
    # PCA-Achsen als Pfeile
    scale = np.sqrt(variances)
    for i, (comp, s) in enumerate(zip(components, scale)):
        ax.annotate(
            "",
            xy=(center[0] + comp[0] * s * PCA_ARROW_SCALE / scale[0],
                center[1] + comp[1] * s * PCA_ARROW_SCALE / scale[0]),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle="->", color="red", lw=2)
        )
    
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invertiert, da Y-Achse nach unten zeigt
    ax.set_title(f"Werkstück-KSYS via PCA ({width}x{height} px)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.legend(loc="upper right", markerscale=5)
    
    filepath = os.path.join(SNAPSHOT_DIR, f"pca_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[INFO] PCA-Plot gespeichert: {filepath}")


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    """Hauptprogramm für Pick & Place."""
    
    print("\n" + "="*70)
    print("PICK & PLACE - Transformationskette")
    print("="*70)
    
    # Kalibrierungsdaten laden
    tvec, rvec, camera_matrix, dist_coeffs = fp.read_wp2c(CALIB_PATH)
    fTc = fp.read_c2f(TRANSFORM_PATH)
    bTf_all, _ = fp.read_bTf(ROBOT_POSES_PATH)
    
    print(f"[INFO] Kamerakalibrierung geladen (tvec, rvec, camera_matrix)")
    print(f"[INFO] Flansch \u2192 Kamera Transformation geladen (fTc)")
    print(f"[INFO] Roboter-Posen geladen (bTf_all)")
    
    print(f"\n[INFO] System bereit")
    print(f"[INFO] Leertaste = Werkstück erkennen & Transformationskette berechnen")
    print(f"[INFO] 'q' = Beenden")
    
    # Modell laden
    model = YOLO(MODEL_PATH)
    
    # Kamera öffnen
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # Tatsächliche Auflösung auslesen (kann von gewünschter Auflösung abweichen)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Kameraauflösung: {actual_width}x{actual_height}")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Hauptschleife
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Objekterkennung
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Anzeige
        cv2.imshow(WINDOW_NAME, annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(" "):  # Leertaste: Transformationskette berechnen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if results[0].masks is None or len(results[0].masks) == 0:
                print("[WARN] Kein Objekt erkannt")
                continue
            
            # Erstes erkanntes Werkstück
            mask = results[0].masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # Pose-Nummer (0-basiert f\u00fcr Array-Zugriff)
            pose_nr = int(TCP_POSE_FOR_CALIBRATION.replace('p', ''))
            
            # Werkst\u00fcck-Pose bestimmen
            workpiece_to_base, workpiece_to_camera, pca_data = detect_workpiece_pose(
                mask_resized, camera_matrix, tvec, rvec, bTf_all, fTc, pose_nr
            )
            
            if workpiece_to_base is not None:
                print("\n" + "="*70)
                print("ERGEBNIS: WERKSTÜCK → BASIS")
                print("="*70)
                print(f"Position: x={workpiece_to_base[0,3]:.2f}, y={workpiece_to_base[1,3]:.2f}, z={workpiece_to_base[2,3]:.2f} mm")
                euler = rotation_matrix_to_euler_angles(workpiece_to_base[:3, :3])
                print(f"Orientierung: Roll={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Yaw={euler[2]:.2f}°")
                print("="*70 + "\n")
                
                # Greifposition berechnen
                gripTCP_to_base, workpiece_to_gripTCP = calculate_grip_pose(workpiece_to_base)
                
                print("\n" + "="*70)
                print("ERGEBNIS: GREIF-TCP → BASIS")
                print("="*70)
                print(f"Position: x={gripTCP_to_base[0,3]:.2f}, y={gripTCP_to_base[1,3]:.2f}, z={gripTCP_to_base[2,3]:.2f} mm")
                euler_grip = rotation_matrix_to_euler_angles(gripTCP_to_base[:3, :3])
                print(f"Orientierung: Roll={euler_grip[0]:.2f}°, Pitch={euler_grip[1]:.2f}°, Yaw={euler_grip[2]:.2f}°")
                print("="*70 + "\n")
                
                # Ablageposition berechnen
                placeTCP_to_base = calculate_place_pose(workpiece_to_gripTCP)
                
                print("\n" + "="*70)
                print("ERGEBNIS: ABLAGE-TCP → BASIS")
                print("="*70)
                print(f"Position: x={placeTCP_to_base[0,3]:.2f}, y={placeTCP_to_base[1,3]:.2f}, z={placeTCP_to_base[2,3]:.2f} mm")
                euler_place = rotation_matrix_to_euler_angles(placeTCP_to_base[:3, :3])
                print(f"Orientierung: Roll={euler_place[0]:.2f}°, Pitch={euler_place[1]:.2f}°, Yaw={euler_place[2]:.2f}°")
                print("="*70 + "\n")
                
                # PCA-Plot speichern
                save_pca_plot(pca_data, timestamp, frame.shape[:2])
        
        elif key == ord("q"):  # q: Beenden
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
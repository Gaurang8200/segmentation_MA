import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.decomposition import PCA
import os
from datetime import datetime

# --- Konfiguration ---
CAMERA_INDEX   = 0                      # 0 = intern, 1 = extern
CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # Windows: CAP_DSHOW | Linux: CAP_ANY
WINDOW_NAME    = "SEG Object Demo"
MODEL_PATH     = "runs/segment/train-8/weights/best.pt"        # Pfad zum trainierten Segmentierungsmodell
OUTPUT_DIR     = "results/SEG_pca"             # Ordner für PCA-Plots
SNAPSHOT_DIR   = "results/SEG_snapshots"       # Ordner für Kameraaufnahmen
USE_FULL_MASK  = True                   # True = alle Maskenpixel | False = nur Randpixel
POINT_COLOR    = "blue"                 # Farbe der Punktwolke
PCA_ARROW_SCALE = 60                    # Skalierung der PCA-Achsenpfeile (in Pixeln)

# Output-Ordner automatisch erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Modell laden
model = YOLO(MODEL_PATH)

# Kamera öffnen
cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Fenster einmalig erstellen (bleibt dauerhaft offen)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def get_point_cloud(mask, use_full_mask=True):
    """Punktwolke aus einer binären Segmentierungsmaske extrahieren."""
    if use_full_mask:
        # Alle Pixel innerhalb der Maske (y, x) -> umkehren zu (x, y)
        yx = np.column_stack(np.where(mask > 0))
        return yx[:, ::-1].astype(float)  # (x, y)
    else:
        # Nur Randpixel (Kontur)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.array([])
        return np.vstack([c.reshape(-1, 2) for c in contours]).astype(float)


def compute_pca(points):
    """PCA auf 2D-Punktwolke anwenden, Mittelpunkt und Achsen zurückgeben."""
    pca = PCA(n_components=2)
    pca.fit(points)
    return pca.mean_, pca.components_, pca.explained_variance_


def save_plot(all_points, all_centers, all_components, all_variances, timestamp):
    """Punktwolken und PCA-Achsen aller Objekte in einem Plot speichern."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [POINT_COLOR, "green", "cyan", "orange", "purple"]

    for i, (points, center, components, variances) in enumerate(
        zip(all_points, all_centers, all_components, all_variances)
    ):
        color = colors[i % len(colors)]
        label_pts = f"Objekt {i + 1} – Punktwolke"
        label_ctr = f"Objekt {i + 1} – Mittelpunkt"

        # Punktwolke
        ax.scatter(
            points[:, 0], points[:, 1],
            s=1, color=color, alpha=0.4, label=label_pts
        )

        # Mittelpunkt
        ax.plot(*center, marker="+", markersize=4, color="red", label=label_ctr)
        # ax.plot(*center, marker="+", markersize=14, color="red", label=label_ctr)

        # PCA-Achsen als rote Pfeile (skaliert nach Varianz)
        scale = np.sqrt(variances)
        for comp, s in zip(components, scale):
            ax.annotate(
                "",
                xy=(center[0] + comp[0] * s * PCA_ARROW_SCALE / scale[0],
                    center[1] + comp[1] * s * PCA_ARROW_SCALE / scale[0]),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2)
            )

    ax.set_aspect("equal")
    ax.invert_yaxis()   # Bildkoordinaten: Y zeigt nach unten
    ax.set_title("Punktwolke & PCA-Orientierung")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.legend(loc="upper right", markerscale=5)

    filepath = os.path.join(OUTPUT_DIR, f"pca_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot gespeichert: {filepath}")


def save_snapshot(frame, timestamp):
    """Kameraaufnahme (annotiertes Bild) speichern."""
    filepath = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Aufnahme gespeichert: {filepath}")


def process_snapshot(frame, annotated_frame, results):
    """Snapshot auslösen: Maske extrahieren, PCA berechnen, Plot und Bild speichern."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results[0].masks is None or len(results[0].masks) == 0:
        print("[WARN] Kein Objekt erkannt – kein Snapshot gespeichert.")
        return

    all_points    = []
    all_centers   = []
    all_components = []
    all_variances  = []

    for idx, mask_tensor in enumerate(results[0].masks.data):
        mask = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        points = get_point_cloud(mask_resized, use_full_mask=USE_FULL_MASK)

        if len(points) < 10:
            print(f"[WARN] Objekt {idx + 1}: Zu wenige Punkte für PCA – übersprungen.")
            continue

        center, components, variances = compute_pca(points)
        print(f"[INFO] Objekt {idx + 1}: Mittelpunkt x={center[0]:.1f}, y={center[1]:.1f}")

        all_points.append(points)
        all_centers.append(center)
        all_components.append(components)
        all_variances.append(variances)

    if all_points:
        save_plot(all_points, all_centers, all_components, all_variances, timestamp)
        save_snapshot(annotated_frame, timestamp)
    else:
        print("[WARN] Keine verwertbaren Masken gefunden.")


# --- Hauptschleife ---
print(f"[INFO] Leertaste = Snapshot  |  'q' = Beenden")
print(f"[INFO] Plots werden gespeichert in: {os.path.abspath(OUTPUT_DIR)}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):      # Leertaste: Snapshot auslösen
        process_snapshot(frame, annotated_frame, results)
    elif key == ord("q"):    # q: Beenden
        break

cap.release()
cv2.destroyAllWindows()

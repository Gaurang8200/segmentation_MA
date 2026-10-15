import sys
import cv2
from ultralytics import YOLO

# demo for detection and tracking with YOLOv26n 

# models
""" 
yolo26n-seg.pt 
yolo26s-seg.pt 
yolo26m-seg.pt 
yolo26l-seg.pt 
yolo26x-seg.pt 
"""

# --- Konfiguration ---
CAMERA_INDEX  = 0                # 0 = intern, 1 = extern
CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # Windows: CAP_DSHOW | Linux: CAP_ANY
WINDOW_NAME   = "YOLOv26 Live Demo"

# Modell laden
model = YOLO("yolo26n-seg.pt")
# model = YOLO("runs/segment/train-4/weights/best.pt")

# Kamera öffnen
cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Fenster einmalig erstellen (bleibt dauerhaft offen)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Inferenz mit Segmentation-Optionen
    results = model.predict(
    frame, 
    verbose=False,
    retina_masks=True,  # Hochauflösende Masken
    conf=0.5           # Confidence Threshold
)

    # Ergebnisse zeichnen
    annotated_frame = results[0].plot()

    # Anzeigen
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # Beenden mit q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
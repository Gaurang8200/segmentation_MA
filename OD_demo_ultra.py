from ultralytics import YOLO
import cv2

# models
""" yolo26n.pt 
yolo26s.pt 
yolo26m.pt 
yolo26l.pt 
yolo26x.pt """

# detection
model = YOLO("yolo26n.pt")

model.predict(
    source=0,      # Webcam
    show=True    
)

""" # tracking 
model = YOLO("yolo26n.pt")

model.track(
    source=0,
    show=True,
    persist=True
) """
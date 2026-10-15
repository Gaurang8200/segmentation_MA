from ultralytics import YOLO
import albumentations as A # albumentations for data augmentation

# Load a COCO-pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# Albumentations sind nach installation teilweise automatisch aktiviert
""" results = model.train(data="coco8.yaml", epochs=100, imgsz=640, augmentations=A.Compose([
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])) """

# Trainingsergebnisse in eigenem Ordner speichern
""" results = model.train(
    data="coco8.yaml", 
    epochs=100, 
    imgsz=640,
    project="meine_experimente",  # Hauptordner
    name="versuch1"               # Unterordner
) """

results = model.train(data="datasets/box/box-seg.yaml", epochs=100, imgsz=640, augmentations=A.Compose([
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
]))
from ultralytics import YOLO
import albumentations as A

# Load a model
# model = YOLO("yolo26n-seg.yaml")  # build a new model from YAML
model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

""" results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640, augmentations=A.Compose([
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])) """

# Train the model with custom data

results = model.train(data="datasets/box_zylinder-seg/box_zylinder-seg.yaml", epochs=100, imgsz=640)

""" results = model.train(data="datasets/box_zylinder-seg/box_zylinder-seg.yaml", epochs=100, imgsz=640, augmentations=A.Compose([
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])) """
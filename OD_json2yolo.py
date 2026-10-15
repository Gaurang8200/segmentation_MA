import json
from pathlib import Path

# LabelMe zu YOLO Konvertierung
json_dir = Path("datasets/box/json_labels")
output_dir = Path("datasets/box/labels/train")
output_dir.mkdir(parents=True, exist_ok=True)

# Klassen-Mapping
classes = {"safety goggles": 0}

for json_file in json_dir.glob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    # YOLO Format: class x_center y_center width height (normalisiert)
    yolo_labels = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            label = shape['label']
            class_id = classes.get(label, 0)
            
            # Bounding Box Koordinaten
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            
            # Zu YOLO Format konvertieren
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Speichern als .txt
    output_file = output_dir / f"{json_file.stem}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_labels))

print(f"Konvertierung abgeschlossen! {len(list(json_dir.glob('*.json')))} Dateien konvertiert.")
import json
from pathlib import Path

# LabelMe zu YOLO Segmentierung Konvertierung
json_dir = Path("datasets/box-seg/json_labels")
output_dir = Path("datasets/box-seg/labels/train")
output_dir.mkdir(parents=True, exist_ok=True)

# Klassen-Mapping
classes = {"box": 0}

for json_file in json_dir.glob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # YOLO Segmentierung Format: class x1 y1 x2 y2 ... xn yn (normalisiert)
    yolo_labels = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            label = shape['label']
            class_id = classes.get(label)
            if class_id is None:
                print(f"Unbekannte Klasse '{label}' in {json_file.name} – wird übersprungen.")
                continue

            # Polygon-Punkte normalisieren
            coords = []
            for x, y in shape['points']:
                coords.append(f"{x / img_width:.6f}")
                coords.append(f"{y / img_height:.6f}")

            yolo_labels.append(f"{class_id} " + " ".join(coords))

    # Speichern als .txt
    output_file = output_dir / f"{json_file.stem}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_labels))

print(f"Konvertierung abgeschlossen! {len(list(json_dir.glob('*.json')))} Dateien konvertiert.")

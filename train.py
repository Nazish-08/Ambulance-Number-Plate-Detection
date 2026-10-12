
from pathlib import Path
from ultralytics import YOLO

dataset = Path(
    r"C:\Users\Nazish\OneDrive\Desktop\Ambulance Dataset"
    r"\Ambulance_YOLO_Annotated_Only\data.yaml"
)

if not dataset.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {dataset}")

model = YOLO("yolo11n.pt")

model.train(
    data=str(dataset),
    epochs=50,
    imgsz=640,
    batch=4,
    patience=10,
    project=r"C:\Users\Nazish\OneDrive\Desktop\Ambulance Dataset\runs",
    name="ambulance_plate",
)

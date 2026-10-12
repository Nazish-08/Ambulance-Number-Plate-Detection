
from ultralytics import YOLO
from pathlib import Path

model = YOLO(
    r"runs\ambulance_plate\weights\best.pt"
)

image_path = input("Enter the full path of an ambulance image: ").strip().strip('"')

if not Path(image_path).is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

model.predict(
    source=image_path,
    conf=0.25,
    save=True,
    show=False
)

print("Detection complete. Check the saved prediction image.")

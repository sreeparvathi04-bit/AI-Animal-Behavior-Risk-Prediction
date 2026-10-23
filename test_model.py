from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"

print("Checking model...")
print("Model exists:", os.path.exists(MODEL_PATH))

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("✅ Model loaded successfully!")

print("\nModel classes:")
print(model.names)

print("\nNumber of classes:", len(model.names))

print("\nClass mapping:")
for class_id, class_name in model.names.items():
    print(f"{class_id} = {class_name}")
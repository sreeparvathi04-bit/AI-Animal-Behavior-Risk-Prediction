from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "vedios2/test_video.mp4"

OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_detection.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Checking files...")

if not os.path.exists(MODEL_PATH):
    print("❌ best.pt not found!")
    exit()

if not os.path.exists(VIDEO_PATH):
    print("❌ Video not found!")
    exit()

print("✅ Model found")
print("✅ Video found")

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("✅ Model loaded!")
print("Classes:", model.names)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information:")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

print("\nStarting detection...")
print("This may take some time on your laptop.")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO detection
    results = model(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    if frame_number % 50 == 0:
        print(
            f"Processed {frame_number}/{total_frames} frames"
        )

cap.release()
out.release()

print("\n✅ Detection completed!")
print("Output video:")
print(OUTPUT_PATH)
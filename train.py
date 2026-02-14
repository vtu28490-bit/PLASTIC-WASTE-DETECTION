from ultralytics import YOLO

# Load YOLOv8 Nano model (lightweight)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="plastic_detection"
)

print("Training Completed ✅")

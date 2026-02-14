from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/plastic_detection/weights/best.pt")

# Load image
image_path = "test.jpg"
img = cv2.imread(image_path)

# Detect plastic waste
results = model(img)

# Show results
annotated_frame = results[0].plot()

cv2.imshow("Plastic Waste Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

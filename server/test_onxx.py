from ultralytics import YOLO

# Load the exported ONNX model
onnx_model = YOLO("./models/best.onnx", task="detect")

# Run inference
results = onnx_model("./test_image/image.png", save=False, show=True)
print(f"结果数量: {len(results)}")

# Print the bounding box results in the format x1, y1, x2, y2, score, cls
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates in xyxy format
        score = box.conf.item()       # Get the confidence score
        cls = box.cls.item()          # Get the class label
        print(f"{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}, {score:.4f}, {cls:.0f}")
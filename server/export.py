from ultralytics import YOLO

# Load a model
model = YOLO("./models/best.pt")  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
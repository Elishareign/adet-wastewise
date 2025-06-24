from ultralytics import YOLO

model = YOLO("weights/best.onnx")  # Use the correct path to your model
print(model.names)

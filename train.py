from ultralytics import YOLO

# Load model (can be a pretrained or your own trained model)
model = YOLO('yolov8s.pt')

# Train
model.train(data='C:\ADET\waste-detection-main\waste-detection.yolov8\data.yaml', epochs=50)


# Export to ONNX
model.export(format='onnx')
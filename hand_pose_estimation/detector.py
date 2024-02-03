from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='handdb.yaml', epochs=3, imgsz=640)

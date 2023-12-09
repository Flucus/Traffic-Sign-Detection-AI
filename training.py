from ultralytics import YOLO, settings

if __name__ == "__main__":
    # View all settings
    print(settings)

    # build a new model from scratch
    # model = YOLO("yolov8n.yaml")
    # continue training 
    model = YOLO("runs/detect/train/weights/best.pt")

    # train the model
    model.train(data="traffic_sign_dataset/dataset.yaml", mode="detect", epochs=200, batch=32, workers=128, imgsz=640, save_period=5)
    # evaluate model performance on the validation set
    metrics = model.val()
    # export the model to ONNX format
    path = model.export(format="onnx")

import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


if __name__ == "__main__":
    # load model
    model = YOLO("runs/detect/train/weights/best.pt")

    # read recognition dir
    dataset_path = "recognition/"
    image_name_list = os.listdir(dataset_path)
    image_path_list = list(map(lambda x: dataset_path + x, image_name_list))
    results = model.predict(image_path_list)

    found_class = set()
    for image_path, result in zip(image_path_list,results):
        img = cv2.imread(image_path)
        annotator = Annotator(img)
        boxes = result.boxes
        for box in boxes:
            # get box coordinates in (top, left, bottom, right) format
            coordinates = box.xyxy[0]
            class_id = box.cls
            class_name = "-".join(model.names[int(class_id)].split("--")[1:-1])
            found_class.add(class_name)
            annotator.box_label(coordinates, class_name)
            # annotator.box_label(coordinates, f"{class_name} {box.conf[0]}")
            # annotator.box_label(coordinates, f"{box.conf[0]:.5f} {box.cls[0]:.5f}")
        img = annotator.result()
        cv2.imshow('YOLO V8 Detection', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(found_class)
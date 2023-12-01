import cv2
import flask
from flask import request, jsonify, send_file
import numpy as np
from ultralytics import YOLO

app = flask.Flask(__name__)

# load model
model = YOLO("runs/detect/train/weights/best.pt")


# define the route for the image post request
@app.route("/detect", methods=["POST"])
def detect():
    # get the image from the request
    image = request.files["image"].read()
    # decode the image and convert it to a NumPy array
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
    # detect objects in the image
    results = model.predict(image)
    # process the results
    objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # get box coordinates in (top, left, bottom, right) format
            coordinates = box.xyxy.tolist()[0]
            class_id = box.cls
            class_name = "-".join(model.names[int(class_id)].split("--")[1:-1])
            objects.append({"coordinates": coordinates, "name": class_name})
    # send the results back to the client
    return jsonify(objects)


# define the route to serve the index.html file
@app.route("/", methods=["GET"])
def index():
    return send_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

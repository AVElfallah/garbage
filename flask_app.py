from roboflow import Roboflow
from flask import Flask, render_template,request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import os
import tempfile

import cv2


rf = Roboflow(api_key="GpeMi130wV64IR7GHOx0")
project = rf.workspace("college-ojdd3").project("garbagedetectionapi")
#dataset = project.version(1).download("yolov5")


# from roboflow import Roboflow
# rf = Roboflow(api_key="GpeMi130wV64IR7GHOx0")
# project = rf.workspace().project("garbagedetectionapi")
model = project.version(1).model

app = Flask(__name__)

@app.route('/')
def index():
    # Return an HTML template with the welcome message
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def predict():
    print("hello")
    file = request.files['image']
    image = Image.open(file)
    #image_array = np.array(image)

    # Save the uploaded image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
    temp_file_path = temp_file.name
    print(temp_file_path)
    image.save(r'./', format='JPEG')
    pathtest=image.filename
    print(temp_file_path)
    print(pathtest)
    with open(pathtest, 'rb') as f:
        lst = model.predict(pathtest, confidence=1, overlap=1).json()
        print(lst)
        pass



    # Load the predicted image
    predicted_image = Image.open("prediction.jpg")

    # Create a byte stream to hold the image data
    image_stream = io.BytesIO()
    predicted_image.save(image_stream, format='JPEG')
    image_stream.seek(0)

    size = []
    MAX = 1000000000
    size_image = (int(lst["image"]["width"]) * int(lst["image"]["height"])) % MAX
    # print(size_image)
    for key, val in lst.items():
        if (key == "predictions"):
            for i in val:
                for k, v in i.items():
                    if (k == "width"):
                        w = v
                    elif (k == "height"):
                        h = v
                size.append(h * w)

    all_size = 0
    for i in size:
        all_size += i

    all_size = all_size % MAX
    sco=(all_size / size_image * 100)
    print(sco, "%")
    if all_size / size_image * 100 >= 50:
        print("Alert! The trash is full")
        return jsonify({"result": "Alert! The trash is full",
                        "score": sco,
                        "image_url": "/prediction.jpg"}), 200
    else:
        print("Trash is not full")
        return jsonify({"result": "Trash is not full",
                        "score": sco,
                        "image_url": "/prediction.jpg"}), 200


# visualize your prediction
# model.predict("your_image.jpg", confidence=20, overlap=10).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=50, overlap=30).json())
if __name__ == '__main__':
    app.run(debug=True)

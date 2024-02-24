from flask import Flask, render_template, request
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the JSON file that contains the model architecture
with open("kkmodel.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model architecture from the JSON
loaded_model_imageNet = model_from_json(loaded_model_json)

# Load the model weights
loaded_model_imageNet.load_weights("DenseNet201_ph1_weights.hdf5")


def secondcheck(filepath):
    img = cv2.imread(filepath)
    x = np.expand_dims(img, axis=0)
    x1 = preprocess_input(x)
    result1 = loaded_model_imageNet.predict(x1)
    ress = list((result1 * 100).astype("int"))
    return ress


def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    img_rgb = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
    x = np.expand_dims(img_rgb, axis=0)
    x = preprocess_input(x)
    result = loaded_model_imageNet.predict(x)
    return result, img_rgb


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/find-specialists")
def find_specialists():
    # Render the find-specialists.html template
    return render_template("find-specialists.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    ressu = None
    if request.method == "POST":
        f = request.files["file"]
        filepath = "static/uploads/" + f.filename
        file_name = f.filename
        f.save(filepath)
        result, img = process_image(filepath)
        p = list((result * 100).astype("int"))
        pp = list(p[0])

        # if pp[1] < pp[0]:
        #     ressu = secondcheck(filepath)

    return render_template("result.html", result=pp, img_path=file_name)


if __name__ == "__main__":
    app.run(debug=True)

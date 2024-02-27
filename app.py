from flask import Flask, render_template, request
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the benign vs malignant model
with open("static/model/best_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model_imageNet = model_from_json(loaded_model_json)
loaded_model_imageNet.load_weights("static/model/best_model.hdf5")

# Load the Melanoma vs Basal Cell Carcinoma model
with open("static/model/malignant_model.json", "r") as json_file:
    malignant_model_json = json_file.read()
malignant_model_imageNet = model_from_json(malignant_model_json)
malignant_model_imageNet.load_weights("static/model/malignant_model.hdf5")


def secondcheck(img):
    xw = np.expand_dims(img, axis=0)
    x1w = preprocess_input(xw)
    result2 = malignant_model_imageNet.predict(x1w)
    ressuu2 = list((result2 * 100).astype("int"))
    ress2 = list(ressuu2[0])

    if ress2[0] > ress2[1]:
        disea = 3
    else:
        disea = 4

    return ress2, disea


def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, threshold = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY)
    final_image = cv2.inpaint(img, threshold, 1, cv2.INPAINT_TELEA)
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
        ressu = list(p[0])
        dis = 1

        if ressu[1] > ressu[0]:
            ressu, dis = secondcheck(img)

    return render_template("result.html", result=ressu, img_path=file_name, disease=dis)


if __name__ == "__main__":
    app.run(debug=True)

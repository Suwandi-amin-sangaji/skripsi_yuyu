# from flask import Flask, request, render_template, redirect, jsonify
# # this is use for url_for() working inside javascript which is help us to navigate the url
# from flask_jsglue import JSGlue

# import util
# import os
# from werkzeug.utils import secure_filename
# from jinja2 import escape
from flask import Flask, request, render_template, redirect, jsonify
from flask_jsglue import JSGlue
import util
import os
from werkzeug.utils import secure_filename
import cv2
import base64
import numpy as np

application = Flask(__name__)

# JSGlue is use for url_for() working inside javascript which is help us to navigate the url
jsglue = JSGlue()  # create a object of JsGlue
# and assign the app as a init app to the instance of JsGlue
jsglue.init_app(application)

util.load_artifacts()


# Home
@application.route("/")
def home():
    return render_template("home.html")

# camera
@application.route("/camera")
def camera():
    return render_template("camera.html")

# Upload
@application.route("/obat", methods=["POST"])
def obat():
    image_data = request.files["file"]
    # save the image to upload
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(basepath, "uploads", secure_filename(image_data.filename))
    image_data.save(image_path)

    predicted_value, details, accuracy = util.obat(image_path)
    os.remove(image_path)

    if predicted_value is None or details is None:
        return jsonify(message="Data tidak ada di dalam set pelatihan")

    return jsonify(predicted_value=predicted_value, details=details, accuracy=accuracy)



# here is route of 404 means page not found error
@application.errorhandler(404)
def page_not_found(e):
    # here i created my own 404 page which will be redirect when 404 error occured in this web app
    return render_template("404.html"), 404


if __name__ == "__main__":
    application.run(debug=True)

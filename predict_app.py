from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template
from flask_cors import CORS
import tensorflow as tf
import base64
import numpy as np 
import io
from PIL import Image, ImageOps
from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras 
import cv2
from PIL import Image

app = Flask(__name__)
CORS(app)

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def get_model():
    global model 
    model = load_model("MIRNet_50epochs",custom_objects={'peak_signal_noise_ratio':peak_signal_noise_ratio,
                                                    'charbonnier_loss':charbonnier_loss})
    


def preprocess_image(original_image):
    size = (400,600)
    original_image = original_image.convert(colors=24)
    original_image = original_image.resize(size)
    print(original_image)
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = tf.convert_to_tensor(image[:,:,:3])
    image = np.expand_dims(image, axis=0)

    return image



print("Loading model...")
get_model()
print("Model loaded!")

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    precessed_image = preprocess_image(image)
    
    prediction = model.predict(precessed_image)
    output_image = prediction[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3))

    img = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

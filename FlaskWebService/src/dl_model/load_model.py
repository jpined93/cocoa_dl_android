<<<<<<< HEAD
import pickle
import tensorflow as tf
import keras
from PIL import Image
import numpy as np

def load(model_path):
    # Loading the pretrained model
    
    Model_json = model_path+"model.json"
    Model_weights = model_path+"model.h5"

    model_json = open(Model_json, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(Model_weights)

    return model


def model_predict(img,model):
    img = img.resize((300, 300))
    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won’t make correct prediction!
    #x = preprocess_input(x)
    preds = model.predict(x)
    return preds

=======
import pickle
import tensorflow as tf
import keras
from PIL import Image
import numpy as np

def load(model_path):
    # Loading the pretrained model
    
    Model_json = model_path+"model.json"
    Model_weights = model_path+"model.h5"

    model_json = open(Model_json, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(Model_weights)

    return model


def model_predict(img,model):
    img = img.resize((300, 300))
    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won’t make correct prediction!
    #x = preprocess_input(x)
    preds = model.predict(x)
    return preds

>>>>>>> d9a335c10659820a03379a8e21a5c7c9c43bfb44

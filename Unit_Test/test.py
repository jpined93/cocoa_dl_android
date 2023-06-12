# import sys
# sys.path.append('C:\Users\USUARIO\Desktop\Tesis\gitrepo\cocoa_dl_android\FlaskWebService2\src')
# import reader
import unittest
from unittest import TestCase
from faker import Faker


from flask import Flask, jsonify,request
import pickle
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from rembg import remove
import test_images

#MODEL_PATH="/app/"
MODEL_PATH="C:/Users/USUARIO/Desktop/Tesis/gitrepo/cocoa_dl_android/FlaskWebService2/src/"
# Load Model
#==============================================================================

Model_json = MODEL_PATH+"model.json"
#Model_weights = MODEL_PATH+"model.h5"

model_json = open(Model_json, 'r')
loaded_model_json = model_json.read()
model_json.close()
# model = tf.keras.models.model_from_json(loaded_model_json)
# model.load_weights(Model_weights)

models_path=["model_fito.h5","model_mazorca_negra.h5","model_monoliosis_ef.h5","model_monoliosis_intermedia_sf.h5"]
models=[]

for path in models_path:
    Model_weights = MODEL_PATH+path
    tmp= tf.keras.models.model_from_json(loaded_model_json)
    tmp.load_weights(Model_weights)
    models.append(tmp)

app=Flask(__name__)

@app.route("/uImg", methods=['GET','POST'])
def val_img():
    try:
        if request.method=="POST":
            d=request.get_data()
            try:
                im_bytes = base64.b64decode(d)   # im_bytes is a binary image
                im_file = BytesIO(im_bytes)  # convert image to file-like object
                img = Image.open(im_file)   # img is now PIL Image object
                img=remove(img)
                img = img.convert(mode='RGB')
                img = img.resize((300, 300))

                print ("image decoded")
            except Exception as e:
                print(f"Exception decoding img: {e}" )
                return jsonify({f"error":f"Exception decoding img: {e}"})
            
            try:
                x = tf.keras.utils.img_to_array(img)
                # x = np.true_divide(x, 255)
                x = np.expand_dims(x, axis=0)
                print ("preprocess compleated")
            except Exception as e:
                print(f"Exception preprocessing img: {e}" )
                return jsonify({f"error":f"Exception preprocessing img: {e}"})

            try:
                preds=[]
                for model in models:
                    individual_preds = model.predict(x)
                    individual_preds=individual_preds.tolist()[0]
                    preds.append(individual_preds[0]) 

                class_pred=np.argmax(np.array(preds))
                class_prob=preds[class_pred]
                
                if class_prob<0.5:
                    class_pred="Sano"
                    class_prob=1-class_prob
                elif class_pred==0:
                    class_pred="Lasiodiplodia"
                elif class_pred==1:
                    class_pred="Mazorca Negra"
                elif class_pred==2:
                    class_pred="Monoliosis"
                elif class_pred==3:
                    class_pred="Monoliosis"

                message=f"{class_pred}"
                return jsonify({"img":message})
            except Exception as e:
                print(f"Exception making predictions img: {e}" )
                return None           
        elif request.method=="GET":
            #data=request.args.get("img")
            d=request.args.get("img")
            return jsonify({"img":"get is working"})
    except Exception as e:
        print(e)
        return jsonify({"exception":e})

#app.run(debug=True)



class test(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.image_string="""
        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
        dfv/AXoGbnIvhEHAAAAAAElFTkSuQmCC
        """


        self.bad_string="""
        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
        cVhtu6H5tK7nw+Ox8m
        dszEzlm/B7o/c7bHJ+CIqBsDeXu7isDeZgy66nK1EsCqHUgILrfWqYTaXNzQji1X2CzYR9uhCLOy
        7eoO6+5YbYPmA9Po96xKarKDFarx5qYxmFnUh0Tt7b091iNd278i0VsNNxOAaRyINWZRkHPuh88/
        Gk93rjD3r7//ZlZTG785bJtAzgahqqSP4xirvmiVM6xtRWpMWQqzBiCkGRAu2l7LFe4c4j0pNQi2
        kjTjhnV+vYpzhxFNlGsXzWo0lnMWKeozOR7HkYjLMZzfidfcKTC4p/HCqQh9uYZzbh7Gdd9V9Yef
        f/nw4YOFuOyvMca3b69arV7VvkTGMq3IwvmlTlRVROZhdM4ZrNu6HN35ebpJsuWQGkBL4rgqqXKt
        dfv/AXoGbnIvhEHAAAAAAElFTkSuQmCC
        """

    def setUp(self):
        self.data_factory = Faker()
        self.client = app.test_client()

    def test_health_check(self):
        getRoutes_request = self.client.get("/uImg", headers={'Content-Type': 'application/json'})
        #self.assertEqual(getRoutes_request.status_code, 200)
        self.assertEqual(getRoutes_request.get_data(), b'{"img":"get is working"}\n')

    def test_image_prediction(self):   
        getRoutes_request = self.client.post("/uImg",data=self.image_string, headers={'Content-Type': 'application/json'})
        print(getRoutes_request.get_data())
        self.assertEqual(getRoutes_request.status_code, 200)
        #self.assertEqual(getRoutes_request.get_data(), b'{"img":"get is working"}\n')

    def test_bad_image_prediction(self):   
        getRoutes_request = self.client.post("/uImg",data=self.bad_string, headers={'Content-Type': 'application/json'})
        print(getRoutes_request.get_data())
        #self.assertEqual(getRoutes_request.status_code, 200)
        self.assertEqual(getRoutes_request.get_data(), b'{"error":"Exception decoding img: Incorrect padding"}\n')

unittest.main()
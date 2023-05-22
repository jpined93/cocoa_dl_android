# from flask import Flask
# helloworld=Flask(__name__)

# @helloworld.route("/")
# def run():
#     return "{\"message\":\"Hey there python\"}"
# if __name__=='__main__':
#     helloworld.run(host="0.0.0.0", port=int("3000"),debug=True)

from flask import Flask, jsonify,request
import pickle
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from rembg import remove

MODEL_PATH="/app/"
#MODEL_PATH="C:/Users/USUARIO/Desktop/Tesis/gitrepo/cocoa_dl_android/FlaskWebService2/src/"
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

app.run(debug=True)

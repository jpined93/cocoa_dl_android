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


def decode_img(img):
    try:
        im_bytes = base64.b64decode(img)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        img = Image.open(im_file)   # img is now PIL Image object
        img = img.resize((300, 300))
        img=remove(img)

        print ("image decoded")
        return img 
    except Exception as e:
        print(f"Exception decoding img: {e}" )
        return None
    

def preprocessing_img(img):
    try:
        x = tf.keras.utils.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
        print ("preprocess compleated")
        return x
    except Exception as e:
        print(f"Exception preprocessing img: {e}" )
        return None
    
def predict_img(x):
    try:
        preds=[]
        for model in models:
            individual_preds = model.predict(x)
            individual_preds=individual_preds.tolist()[0]
            preds.append(individual_preds[0])  

        class_pred=4 if np.argmax(np.array(preds))<0.5 else np.argmax(np.array(preds))
        class_prob=preds[class_pred] if class_pred!=4 else 1-preds[class_pred]
        if class_pred==0:
            class_pred=f"Fito {class_prob:.0%}"
        elif class_pred==1:
            class_pred=f"Mazorca negra {class_prob:.0%}"
        elif class_pred==2:
            class_pred=f"Monoliasis {class_prob:.0%}"
        elif class_pred==3:
            class_pred=f"Monoliasis Intermedia {class_prob:.0%}"
        elif class_pred==4:
            class_pred=f"Sano {class_prob:.0%}"

        print ("image predicted")
        return jsonify({"img":str(class_pred)})
    except Exception as e:
        print(f"Exception making predictions img: {e}" )
        return None


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
                    class_pred="Fito"
                elif class_pred==1:
                    class_pred="Mazorca negra"
                elif class_pred==2:
                    class_pred="Monoliasis"
                elif class_pred==3:
                    class_pred="Monoliasis Intermedia"

                message=f"Predicción: {class_pred} con probabilidad de {class_prob}"
                return jsonify({"img":message})
            except Exception as e:
                print(f"Exception making predictions img: {e}" )
                return None

            # try:
            #     preds=[]
            #     for model in models:
            #         individual_preds = model.predict(x)
            #         individual_preds=individual_preds.tolist()[0]
            #         preds.append(individual_preds[0])  

            #     class_pred=4 if np.argmax(np.array(preds))<0.5 else np.argmax(np.array(preds))
            #     class_prob=preds[class_pred] if class_pred!=4 else 1-np.argmax(np.array(preds))
            #     if class_pred==0:
            #         class_pred=f"Fito {class_prob:.0%}"
            #     elif class_pred==1:
            #         class_pred=f"Mazorca negra {class_prob:.0%}"
            #     elif class_pred==2:
            #         class_pred=f"Monoliasis {class_prob:.0%}"
            #     elif class_pred==3:
            #         class_pred=f"Monoliasis Intermedia {class_prob:.0%}"
            #     elif class_pred==4:
            #         class_pred=f"Sano {class_prob:.0%}"

            #     print ("image predicted")
            #     return jsonify({"img":str(class_pred)})
            # except Exception as e:
            #     print(f"Exception making predictions img: {e}" )
            #     return None
            
            
            # try:
            #     preds = model.predict(x)
            #     preds=preds.tolist()[0]
            #     if np.argmax(np.array(preds))==0:
            #         class_pred="Cat"
            #     elif np.argmax(np.array(preds))==1:
            #         class_pred="Dog"
                
            #     print ("image predicted")
            #     return jsonify({"img":str(class_pred)})
            # except Exception as e:
            #     print(f"Exception making predictions img: {e}" )
            #     return jsonify({f"error":f"Exception making predictions img: {e}"})



            # Transfor image to PIL
            #==============================================================================

            # im_bytes = base64.b64decode(d)   # im_bytes is a binary image
            # im_file = BytesIO(im_bytes)  # convert image to file-like object
            # img = Image.open(im_file)   # img is now PIL Image object
            # img = img.resize((300, 300))
            

            # # Preprocessing the image
            # x = tf.keras.utils.img_to_array(img)
            # # x = np.true_divide(x, 255)
            # x = np.expand_dims(x, axis=0)
            # # Be careful how your trained model deals with the input
            # # otherwise, it won’t make correct prediction!

            

            # preds = model.predict(x)
            # preds=preds.tolist()[0]
            # if preds[0]==1:
            #     class_pred="Cat"
            # elif preds[1]==1:
            #     class_pred="Dog"
            # return jsonify({"img":str(class_pred)})
            
        elif request.method=="GET":
            #data=request.args.get("img")
            d=request.args.get("img")
            return jsonify({"img":"get is working"})
    except Exception as e:
        print(e)

app.run(debug=True)

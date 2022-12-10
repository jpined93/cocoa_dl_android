<<<<<<< HEAD
from ast import Not
import os
import re
import traceback
import json
from flask import request, jsonify, send_file
from datetime import datetime
from flask_restful import Resource,reqparse
from flask import current_app as app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from dl_model.load_model import load,model_predict
import tensorflow as tf
import keras 


class Health(Resource):    
    def get(self):
        return {"resultado": "OK", "mensaje": "service is alive"}, 200

class Image(Resource):

    def get(self):
        from PIL import Image
        from io import BytesIO
        import base64
        import json

        
        
        MODEL_PATH="C:/Users/Lobo_/Desktop/Cocoa_DL/cocoa_dl_android/FlaskWebService/src/dl_model/"
        model=load(MODEL_PATH)


        #para leer el body request.json["img"] siendo img la etiqueta del JSON


        y = json.loads(request.json["img"])
        img = Image.open(BytesIO(base64.b64decode(y["img"])))
        

        #preds=model_predict(img,model)
        #preds=preds.tolist()[0]
        return json_image #{"resultado":preds}, 200

class PostImage(Resource):
    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201



=======
from ast import Not
import os
import re
import traceback
import json
from flask import request, jsonify, send_file
from datetime import datetime
from flask_restful import Resource,reqparse
from flask import current_app as app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from dl_model.load_model import load,model_predict
import tensorflow as tf
import keras 


class Health(Resource):    
    def get(self):
        return {"resultado": "OK", "mensaje": "service is alive"}, 200

class Image(Resource):

    def get(self,json_image):
        from PIL import Image
        from io import BytesIO
        import base64
        import json
        
        MODEL_PATH="C:/Users/Lobo_/Desktop/Cocoa_DL/cocoa_dl_android/FlaskWebService/src/dl_model/"
        model=load(MODEL_PATH)


        y = json.loads(json_image)
        img = Image.open(BytesIO(base64.b64decode(y["img"])))
        

        #preds=model_predict(img,model)
        #preds=preds.tolist()[0]
        return json_image #{"resultado":preds}, 200

class PostImage(Resource):
    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201



>>>>>>> d9a335c10659820a03379a8e21a5c7c9c43bfb44
        return json_image
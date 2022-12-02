from ast import Not
import os
import re
import traceback
import json
from flask import request, jsonify, send_file
from datetime import datetime
from flask_restful import Resource
from flask import current_app as app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity



class Health(Resource):    
    def get(self):
        return {"resultado": "OK", "mensaje": "service is alive"}, 200

class Image(Resource):
    def get(self):
        return {"resultado": "OK", "mensaje": "service is alive"}, 200
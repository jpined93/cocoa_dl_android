import os
from pathlib import Path
import flask
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_restful import Api
from dl_model.image_service import Image,Health


app = Flask(__name__)
api = Api(app)

api.add_resource(Health, '/h')
api.add_resource(Image, '/image')


jwt = JWTManager(app)
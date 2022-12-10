import os
from pathlib import Path
import flask
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_restful import Api,reqparse
from dl_model.image_service import Image,Health,PostImage


app = Flask(__name__)
api = Api(app)


parser = reqparse.RequestParser()
parser.add_argument('task')



api.add_resource(Health, '/')
api.add_resource(Image, '/image')
api.add_resource(PostImage, '/l_img')

jwt = JWTManager(app)
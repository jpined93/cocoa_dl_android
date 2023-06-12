# Cocoa_DL
# Desarrollo de una aplicación móvil para la detección de enfermedades en los frutos de cacao / Development of a mobile application for cocoa fruit diseases detection.

## Table of contents
* [Authors](#Authors)
* [DataSet](#DataSet)
* [Modelation](#Modelation)
* [Web Service](#Web_Service)
* [Mobile App](#Mobile_App)

## Authors
| Organization   | Name | Email | 
|----------|-------------|-------------|
| PUJ-Bogota | Sebastián Pineda| juanspineda@javeriana.edu.co|
| PUJ-Bogota  |  Camilo Cano | c-cano@javeriana.edu.co |

## DataSet

The data set was built with a combined effort using open search images and images collect on the department of Guaviare.
<br>
<br>
The data consist on 700 images distributed on 5 categories (Sanas, Mazorca Negra, Monoliosis Etapa Intermedia, Monoliosis Etapa Final y Phytophthora pod rot)

## Modelation
On the src folder you will find the notebooks used for building the model:

* model_experimentation\src\All_models.ipynb : All models and parameter combination for experimentation.
* model_experimentation\src\Background removal.ipynb : Preprocessing of images removing background.

## Web Service
On the src folder you will find the main files for running the service.

* FlaskWebService2\src\app.py: Flask App.
* FlaskWebService2\src\Dockerfile: File use to build the docker image on the virtual machine.
* FlaskWebService2\src\model_fito.h5: Weights of the network specifically for detecting Phytophthora pod rot.
* FlaskWebService2\src\model_mazorca_negra.h5 : Weights of the network specifically for detecting Mazorca Negra.
* FlaskWebService2\src\model_monoliosis_ef.h5 : Weights of the network specifically for detecting Monoliosis Etapa Final.
* FlaskWebService2\src\model_monoliosis_intermedia_sf.h5 : Weights of the network specifically for detecting Monoliosis Etapa Intermedia.
* FlaskWebService2\src\model.json : json defining model arquitecture used.
* FlaskWebService2\src\requirements.txt : Libraries required 

### installation using Dockerfile
1. cd to src folder or folder containing all src files and Docker file and run the following command docker build -t tesis_flask_service:0.0.1 .
2. Run image using container using the following  command docker container run -d -p 5000:5000 tesis_flask_service:0.0.1


### Installation using local virtual environment
1. Create virtual enviroment python -m venv C:\Users\Lobo_\Desktop\Cocoa_DL\cocoa_dl_android\FlaskWebService\venv
2. incase of lack of permmisions allow scripts run with following command:
Set-ExecutionPolicy Unrestricted -Scope Process
3. Set directory to FlaskWebService
4. Activate virtual enviroment from FlaskWebService folder venv/Scripts/activate
5. On Virtual enviroment run py -m pip install -r requirements.txt if not already installed
6. Activate Flask app enviroment $env:FLASK_APP ="C:\Users\Lobo_\Desktop\Cocoa_DL\cocoa_dl_android\FlaskWebService\src\app.py"
or $env:FLASK_APP ="C:/Users/Lobo_/Desktop/Cocoa_DL/cocoa_dl_android/FlaskWebService/src/app_simple.py"
7. exec on console: flask run


## Mobile App


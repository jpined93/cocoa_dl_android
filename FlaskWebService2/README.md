Creación de imagen docker y levantamiento del servicio


crear imagen-->docker build -t tesis_flask_service:0.0.1 .


correr la imagen --> docker container run -d -p 5000:5000 tesis_flask_service:0.0.1
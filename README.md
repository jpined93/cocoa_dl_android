# NLP_2022_3
# Titulo del proyecto: Clasificador de afinidad en discursos de la Segunda Guerra mundial
>La Segunda Guerra Mundial fue uno de los grandes hitos de la historia geopolítica del siglo XX, conflicto que involucró prácticamente a todas las partes del mundo durante los años 39- 45. Las principales luchas se dieron entre las potencias del Eje (Alemania, Italia y Japón) y los Aliados (Francia, Gran Bretaña, Estados Unidos, la Unión Soviética y, en menor medida, China), y muchas de estas luchas fueron determinadas o impulsadas por discursos emblemáticos entre todas las partes involucradas.
El conjunto de datos está compuesto por 3521 discursos de la segunda guerra mundial, traducidos al inglés, obtenidos de fuentes abiertas, se encuentran acompañados del Autor, Fecha de alocución, facción a la que pertenece y país.


## Tabla de Contenidos
* [Autores](#autores)
* [Análisis exploratorio](#análisis-exploratorio)
* [Descripción de la solución](#descripción-de-la-solución)
* [Requerimientos](#requerimientos)
* [Instalacion](#instalación)

## Autores
A continuación se presenta el listado de autores y su respectivo correo electronico.

| Organización   | Nombre del Miembro | Correo electronico | 
|----------|-------------|-------------|
| PUJ-Bogota | Sebastián Pineda| juanspineda@javeriana.edu.co|
| PUJ-Bogota  |  Camilo Cano | c-cano@javeriana.edu.co |
| PUJ-Bogota  |  Daniel Latorre   | latorreldaniel@javeriana.edu.co| 
| PUJ-Bogota  |  Cesar Ramirez   | ce-ramirez@javeriana.edu.co|

## Análisis exploratorio

Distribución de textos por año:

![1](https://user-images.githubusercontent.com/99692504/203151861-80767a17-e2e4-4f3c-893a-e410cd95c188.PNG)

La mayor cantidad de discursos se encuentran concentrados hacia la etapa tardía de la guerra, entre 1940 y 1944. Siendo los mayores representantes Hitler,Stalin, Roosevelt y Churchill.

Análisis de frecuencia de palabras por Autor:

![2](https://user-images.githubusercontent.com/99692504/203152344-a6c1d96e-6caf-4ccf-8e90-b9d68bf95c6a.PNG)

Al estudiar las frecuencias de palabras por cada autor podemos ver nuevamente que Churchill, Roosevelt y Stalin tienen la mayor cantidad de palabras, seguidos por Hitler sin importar si tomamos a consideración la cuenta de palabras distintas o no.

Longitud Media de los discursos por Autor:

![3](https://user-images.githubusercontent.com/99692504/203152661-883756bd-da4c-497d-af60-4cc2e2bf6288.PNG)

Se resalta la variabilidad en la longitud de sus alocuciones, representada por el coeficiente de variación, que para los casos puntuales de Patton y Stalin es considerablemente alto.


## Descripción de la solución

Se desarrolló e implementó una red neuronal recurrente bidireccional (BRNN), esta estructura se compone de dos capas ocultas opuestas que se concatenan a la salida. Emplea dos direcciones, una dirección positiva (estados adelante) y otra dirección negativa (estados atrás). La salida de los dos estados no está conectada a las entradas de los estados de dirección opuesta; las dos neuronas direccionales no tienen interacciones. Se escogen dos redes de memoria a largo plazo (LSTM) debido a su capacidad de agregar o eliminar información a esta celda de estado usando una serie de compuertas (forget, update y output) que permiten discriminar entre la información relevante e irrelevante. Se emplea la herramienta Tensor Flow para su desarrollo. Al existir un desbalance de clases (autores) se aplican técnicas de nivelación de datos, en este caso Oversampling. Para la construcción del modelo se define una estructura con dos capas bidireccionales LSTM y una capa densa de decisión. Para evitar el sobre ajuste se emplea un Dropout de 0,5 y al ser una red de clasificación multi clase se aplica una función de pérdida Cross entropy. Se emplean 10 épocas para el entrenamiento en búsqueda de los mejores resultados.

## Resultados.

Se obtiene un Accuracy en entrenamiento de 92.40% y un Accuracy para pruebas de 82.83%.

Resultados entrenamiento Accuracy por épocas:

![accurracy](https://user-images.githubusercontent.com/99692504/203153681-9e629bb9-37f5-4107-9f20-f914b1fbec58.PNG)

Resultados entrenamiento Pérdida por épocas:

![loss](https://user-images.githubusercontent.com/99692504/203153814-2fbaacd0-298e-43cd-bc94-2f349532fe20.PNG)

Resultados Autor discursos vs precisión, Recall y F1 score.

![resultados](https://user-images.githubusercontent.com/99692504/203153898-f0a39d4a-dbb5-489e-b445-38de56af177f.PNG)

## GUI

Para consumir el modelo desarrollado se genera un Dashboard con el framework Dash, para Python que está basado en Flask, Plotly y ReactJS. Se integra la base de datos, el modelo de clasificación y un Callback. Como entrada se tiene un espacio de texto libre al cual se aplican técnicas de limpieza de texto que será consumido por el modelo de clasificación desarrollado. Retorna el Top 3 de los autores al cual tiene más afinidad. Se agrega un selector desplegable que permite hacer comparación de análisis de datos entre el autor a establecer y el discurso cargado.

![WhatsApp Image 2022-11-20 at 9 10 56 PM](https://user-images.githubusercontent.com/99692504/203154087-cd243863-3c91-4b34-bd96-667890347846.jpeg)


## Requerimientos

### Librerias Empleadas 
- nltk~=3.7
- dash~=2.6.2
- dash-bootstrap-components~=1.2.1
- dash-core-components~=2.0.0
- dash-html-components~=2.0.0
- dash-table~=5.0.0
- spacy~=3.4.1
- numpy~=1.23.3
- plotly~=5.10.0
- autocorrect~=2.6.1
- tqdm~=4.64.1
- sklearn~=0.0
- scikit-learn~=1.1.2
- clean-text~=0.6.0
- pandas~=1.5.0
- wheel~=0.37.1
- setuptools~=59.8.0
- matplotlib~=3.6.0
- py~=1.11.0
- Werkzeug~=2.2.2
- pip~=22.3
- cryptography~=38.0.3
- Flask~=2.2.2
- tensorflow~=2.10.0

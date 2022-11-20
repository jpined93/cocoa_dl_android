# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:11:55 2022

@author: jpined93
"""
# Dashboard 
#===============================================================================
import plotly.express as px
from dash import Dash, html, dcc
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import nltk
nltk.download('stopwords')
import spacy
nlp = spacy.load('en_core_web_sm')
#===============================================================================

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import string
import re
import tensorflow as tf
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#style.use('ggplot') or plt.style.use('ggplot')
# Preprocesado y modelado
# ==============================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

ROOT_PATH="C:/Users/Lobo_/Desktop/Interface Flask/"


datos=pd.read_excel(ROOT_PATH+"Discursos.xlsx")
bandos=datos.groupby("Gruop").agg("count")["Country"]
bandos=pd.DataFrame(bandos)
bandos["Group"]=bandos.index
datos['Date'] = pd.to_datetime(datos['Date'])
datos['Year']= pd.DatetimeIndex(datos['Date']).year
datos['iso_alpha']=datos['Country'].replace(['France'], 'FRA')
datos['iso_alpha']=datos['iso_alpha'].replace(['Russia'], 'RUS')
datos['iso_alpha']=datos['iso_alpha'].replace(['Germany'], 'DEU')
datos['iso_alpha']=datos['iso_alpha'].replace(['Japan'], 'JPN')
datos['iso_alpha']=datos['iso_alpha'].replace(['Italy'], 'ITA')
datos['iso_alpha']=datos['iso_alpha'].replace(['UK'], 'GBR')


# Distribución temporal de los discursos
# ==============================================================================
df_temp=pd.DataFrame(datos.groupby(["Date","Autor"]).size().reset_index().rename(columns={0: "Conteo"}))
df_temp=df_temp[df_temp["Date"]>='1937-01-01 00:00:00']

# Distribución espacial de los discursos
# ==============================================================================
df_geo=pd.DataFrame(datos.groupby(["Year","iso_alpha"]).size().reset_index().rename(columns={0: "Conteo"}))



# Obtención de listado de stopwords del inglés
# ==============================================================================
stop_words = list(stopwords.words('english'))
# Se añade la stoprword: amp, ax, ex
stop_words.extend(("amp", "xa", "xe"))



# Limpieza de datos
# ==============================================================================
def limpiar_tokenizar(texto):
    stop_words = list(stopwords.words('english'))
    # Se añade la stoprword: amp, ax, ex
    stop_words.extend(("amp", "xa", "xe"))
    # Se convierte todo el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1 and token not in stop_words ]
    
    return(nuevo_texto)


datos['texto_tokenizado'] = datos['Text'].apply(lambda x: limpiar_tokenizar(x))

# Unnest de la columna texto_tokenizado
# ==============================================================================

def tidy_data(datos):
    datos_tidy = datos.explode(column='texto_tokenizado')
    datos_tidy = datos_tidy.drop(columns='Text')
    datos_tidy = datos_tidy.rename(columns={'texto_tokenizado':'token'})
    datos_tidy = datos_tidy[~(datos_tidy["token"].isin(stop_words))]
    
    return datos_tidy
    
   

# Filtrado para excluir stopwords
# ==============================================================================



# Top 5 palabras más utilizadas por cada autor
# ==============================================================================
def top_w_autor(top):
    
    t=datos_tidy.groupby(['Autor','token'])['token'] \
     .count() \
     .reset_index(name='count') \
     .groupby('Autor') \
     .apply(lambda x: x.sort_values('count', ascending=False).head(top))
     
    return t

# Cálculo del log of odds ratio de cada palabra (elonmusk vs mayoredlee)
# ==============================================================================
# Pivotaje 
# ==============================================================================



def dynamic_unpivot(tidy_data):
    datos_pivot = tidy_data.groupby(["Autor","token"])["token"] \
                    .agg(["count"]).reset_index() \
                    .pivot(index = "token" , columns="Autor", values= "count")
    datos_pivot = datos_pivot.fillna(value=0)
    datos_pivot.columns.name = None

    #despivotaje
    # ==============================================================================
    datos_unpivot = datos_pivot.melt(value_name='n', var_name='Autor', ignore_index=False)
    datos_unpivot = datos_unpivot.reset_index()
    
    return datos_unpivot




# Selección de los autores elonmusk y mayoredlee


def log_odds_Autores(Autor1,Autor2,n_words,datos_unpivot,datos_tidy):
    # ==============================================================================
    datos_unpivot = datos_unpivot[datos_unpivot.Autor.isin([Autor1, Autor2])]
    # Se añade el total de palabras de cada autor
    datos_unpivot = datos_unpivot.merge(datos_tidy.groupby('Autor')['token'].count().rename('N'),how = 'left',on  = 'Autor')
    datos_unpivot
    # Cálculo de odds y log of odds de cada palabra
    datos_logOdds = datos_unpivot.copy()
    datos_logOdds['odds'] = (datos_logOdds.n + 1) / (datos_logOdds.N + 1)
    datos_logOdds = datos_logOdds[['token', 'Autor', 'odds']] \
                        .pivot(index='token', columns='Autor', values='odds')
    datos_logOdds.columns.name = None
    datos_logOdds['log_odds']     = np.log(datos_logOdds[Autor1]/datos_logOdds[Autor2])
    datos_logOdds['abs_log_odds'] = np.abs(datos_logOdds.log_odds)
    datos_logOdds['autor_frecuente'] = np.where(datos_logOdds.log_odds > 0,Autor1,Autor2)
    datos_logOdds=datos_logOdds.sort_values('abs_log_odds', ascending=False)
    
    top_30 = datos_logOdds[['log_odds', 'abs_log_odds', 'autor_frecuente']].groupby('autor_frecuente').apply(lambda x: x.nlargest(n_words, columns='abs_log_odds').reset_index()).reset_index(drop=True).sort_values('log_odds')
    
    return top_30


#===========================================================================
import re
import string
from string import digits
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
import keras



# CLean String
#=============================================================================
def clean_string(text, stem="None"):
    final_string = ""
    # Make lower
    text = text.lower()
    # Remove line breaks
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[0-9]', '', text)
    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']
    text_filtered = [word for word in text if not word in useless_words]
    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered
    final_string = ' '.join(text_stemmed)

    user_input_clean = pd.DataFrame({'sentence': [i+'.' for i in final_string.split('$')]})

    new_model = keras.models.load_model(ROOT_PATH+"model_text_classifier_f.h5")

    import pickle

    with open(ROOT_PATH+"tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)

    Autors_list=['Chamberlain','Churchill','Eisenhower','Gaulle','Gobbels','Himmler','Hirohito','Hitler','King Georges','Mussolini','Patton','Roosevelt','Stalin','Truman']
    
    
    
    user_seq = tokenizer.texts_to_sequences(user_input_clean["sentence"].head(1))
    max_length = 378
    user_vector = tf.keras.preprocessing.sequence.pad_sequences(user_seq, maxlen=max_length, padding='post', truncating='post')
    prediction_list=new_model.predict(user_vector)[0].tolist()
    prediction_user = np.argmax(new_model.predict(user_vector),axis=1)
    
    top3_pred=pd.DataFrame({'Autors':Autors_list,'Prob':prediction_list})
    top3_pred=top3_pred.sort_values(by=['Prob'], ascending=False).head(3)
    
    return top3_pred  #Autors_list[prediction_user][0],prediction_list[0][prediction_user][0]

#=============================================================================


#for item in t.values.tolist():
#    print(item[0])



# DataFrames
# ==============================================================================

datos_tidy=tidy_data(datos)
datos_unpivot=dynamic_unpivot(datos_tidy)
word_freq_per_author=pd.DataFrame(datos_tidy.groupby(by='Autor')['token'].count())
word_freq_per_author["Autor"]=word_freq_per_author.index
unpivot_data=dynamic_unpivot(datos_tidy)


l_od=log_odds_Autores("Roosevelt","Churchill",15,unpivot_data,datos_tidy)

# Build Components
# ==============================================================================

app=Dash(__name__, external_stylesheets= [dbc.themes.SLATE])




mytitle=dcc.Markdown(children='')


# Cuadros de Texto
# ==============================================================================
mytext= dbc.Card(
    id="mytext",children="Nombre del Autor",
    style={'height':'100px', 'width':'300px'},
 )
myinput=dbc.Textarea(
    id="myinput",value="# El discurso va aqui",
    style={'height':'300px'},

)


# Graficas
# ==============================================================================

mygraph=dcc.Graph(
      id="mygraph",figure={}
)

Log_Odds_Plot=dcc.Graph(
    id="Log_Odds_Plot",figure={}
)

mygraph2=dcc.Graph(
    id="mygraph2",figure={}
)
myCmap=dcc.Graph(id="myCmap",figure={})


n_words_dropdown=dcc.Dropdown(id="n_words_dropdown",
    options=[5,10,15,20]
    ,value=5 # default value
    ,clearable=False
)



autor_dropdown=dcc.Dropdown(id="autor_dropdown",
    options=list(datos_tidy["Autor"].unique())
    ,value=list(datos_tidy["Autor"].unique())[0]  # default value
    ,clearable=False
)

# Imagenes
# ==============================================================================
html_img_well = html.Img(id = 'html-img', src = 'https://images.squarespace-cdn.com/content/v1/5e4976ff878c9b6addef705f/6b9d32d2-2fc5-4830-a65d-c2131a87bc66/Screen+Shot+2020-02-20+at+5.07.48+PM.jpg',
                                style={'height':'300px', 'width':'300px'}) 



def drawFigure(custom_plot):
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=custom_plot.update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])

# Text field
def drawText(custom_text):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(custom_text),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])


def drawImageFrame(imgObject):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                   imgObject
                ]) 
            ])
        ),
    ])


# Data
#df = px.data.iris()

#Layout customization




fig1=px.bar(data_frame=word_freq_per_author,x="Autor",y="token")
fig2=px.line(data_frame=df_temp,x="Date",y="Conteo",color='Autor')
fig3 = px.choropleth(
            df_geo
            ,locations="iso_alpha"
            #,scope="europe"
            ,color="Conteo" # lifeExp is a column of gapminder
            ,hover_name="iso_alpha" # column to add to hover information
            ,color_continuous_scale=px.colors.sequential.Plasma
            ,animation_frame='Year'
            
            )
#fig4=px.bar(data_frame=log_odds_Autores('Roosevelt', 'Churchill',15,unpivot_data),x="log_odds",y="token", color="autor_frecuente"  )



app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText("Dashboard Analisis de Discursos")
                ], width=12)
            ], align='center'), 
            dbc.Row([
                dbc.Col([
                    drawText("Discurso personalizado para analizar")
                ], width=6),
                dbc.Col([
                    drawText("Autor con más afinidad")
                ], width=6)
            ], align='center'), 
            
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    myinput
                ], width=6),
                dbc.Col([
                    mytext
                ], width=3),
                dbc.Col([
                    html_img_well#drawImageFrame(html_img_well)
                ], width=3)
            ], align='center'), 
            
    
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    drawText("Comparar con un autor")
                ], width=6),
                dbc.Col([
                    autor_dropdown #drawText("Texto 3")
                ], width=2),
                dbc.Col([
                    n_words_dropdown #drawText("Texto 3")
                ], width=1),
            ], align='center'),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    mygraph
                ], width=3),
                dbc.Col([
                    drawFigure(fig2) 
                ], width=3),
                dbc.Col([
                    Log_Odds_Plot #drawFigure(fig4) 
                ], width=6),
            ], align='center'), 
            
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    drawFigure(fig3)
                ], width=6),
                dbc.Col([
                    drawFigure(fig2)
                ], width=6),
            ], align='center'),
                  
            dbc.Row([
                dbc.Col([
                    drawText("Mapa Coroplético")
                ], width=6),
                dbc.Col([
                    drawText("Análisis Temporal")
                ], width=6),
            ], align='center'), 
            
        ]), color = 'dark'
    )
])





#Callback 


#Ejemplo de callback para cambiar un valor en un componente usando un imput
#@app.callback(
#    Output(mygraph, component_property='children'),
#    Input(myinput, component_property='value')
#)


@app.callback(
    
    Output(Log_Odds_Plot, component_property='figure'),
    Output(mygraph, component_property='figure'),
    Output(mytext, component_property='children'),
    Output(html_img_well, component_property='src'),
    Input(myinput, component_property='value'),
    Input(autor_dropdown, component_property='value'),
    Input(n_words_dropdown, component_property='value')
    )




# dependiendo de lo que se ponga en el call back la funcion que devuelve los valorews
# debe ajustarsepara regresar o solicitar la misma cantidad de inputs o outputs

def update_output(user_input,autor_dropdown,n_words_dropdown):
    
    #Append user_input to dataframe
    temp=pd.DataFrame([[1,"Custom","2022-11-12 00:00:00","Custom","Custom",user_input,2022,"CUSTOM",limpiar_tokenizar(user_input)]],columns=datos.columns)
    
    temp = datos.append(temp, ignore_index=True)
    
    datos_tidy_fake=tidy_data(temp)
    
    
    word_freq_per_author_fake=pd.DataFrame(datos_tidy_fake.groupby(by=['Autor','id'])['token'].count())
    word_freq_per_author_fake=word_freq_per_author_fake.reset_index(level = ['Autor',]).groupby("Autor")["token"].agg("mean").reset_index(level = ['Autor',])
    
    unpivot_fake=dynamic_unpivot(datos_tidy_fake)
    
    #print(user_input)
    
    
    # Grafico de frecuencias
    fig=px.bar(data_frame=word_freq_per_author_fake,x="Autor",y="token").update_layout(
        template='plotly_dark',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
    )
    
    #Log odds
    fig4=px.bar(data_frame=log_odds_Autores('Custom', autor_dropdown,n_words_dropdown,unpivot_fake,datos_tidy_fake),x="log_odds",y="token", color="autor_frecuente"  ).update_layout(
        template='plotly_dark',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
    )
    
    if user_input!="" and user_input!="# El discurso va aqui":
        t= clean_string(user_input)
        t=t.values.tolist()
        card_value=[]
        autor_choosed=t[0][0]
        
        for item in t:
            card_value.append(item[0]+" prob: "+str(item[1])+"\n")
            
        print("text classified")
    else:
        t=""
        autor_choosed=""
        card_value="Nombre del Autor"
        print("idle")
    
    img_link='https://images.squarespace-cdn.com/content/v1/5e4976ff878c9b6addef705f/6b9d32d2-2fc5-4830-a65d-c2131a87bc66/Screen+Shot+2020-02-20+at+5.07.48+PM.jpg'
    #img_link='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxIQEhUSEhISEhAVEBUVEBAQFg8PEA8PFRUWFhUVFRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKBQUFDgUFDisZExkrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrK//AABEIAP8AxgMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAABAgADBAUGB//EADsQAAEDAgMFBwIEBQQDAQAAAAEAAhEDIQQSMQVBUWFxBhMiMoGRoVKxQsHR8BUjguHxFDNicpKTwgf/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AuqFVl6j3KpzkELkpclcUpcgZzlWXIFyWUDhyIcqwUQUDSoSmp0y7Qa6Lp09lAQHGXbwDAbxlBzMpBA3lEUzwK9Fhtmt3Ai+gt6zqVs/hLdQ0c7a+qDyBYRr6qsu917Ctspp1F+YXLxnZ8EctbTCDhylzLRV2U9hiSBuWZ7CJEg8wUBDlC5V5kC5BZmUzKuUQUFwcmzKoFMCgtDk0qqUwKCwFFICogveVUU7yqnOQK4pSVHFISgJKCBcllA5KtwlA1HAD16BZpXoOyWzHV6hcPKwtk/U4mzem9B18Fg20WjLHfZbuse7EEkgcToOaryAW/wA+q9PtHZ7KNEgebVx3uOpjhovNUbniSg6WCpaLoMasuF0WtnJBW6mCqn0wR+7LZlVNRhm6Dj4ygHTbpzXltsbPDRma2HTzF+C9viGDTVcjG0paf3BCDxbgdSI5c0it2pUczUeDR3RUoDKISyiCgcFMFXKZBZKMquUyC1pQSgqILnOVbig4oSgBKVxRJSOKBS5CUIQQMCvon/5oz+VUcdBUt1y6r5yvddg8Zko1GzH85sk7g4f2Qek7TYwNaDYC8TvJBA6Bed2XeIWzbuZ7ZgGI8ZLWxG6DpCybMaRa2nyg6+H/AMLbT0WSgLq+hUn1QamMEST6FZ6/KI5fotTBAWWs2Tp1KDE6df37rBiG6zG8+i6GIJH6rn1XDKb7j7oPMbXwwc1zbaSCdxC4FPQcYHuF62vTzep+F5jE08r3N526IEUCARQMCjKSUUDymlVhRBcwqJWFRA5SkoEpCUBJSkpZQJQMSlSlSUDr0vYrEZX1GDV9ORv8lz8FeZldXs891OtSqaMNTuyeOYER9kHr9v1mN8BfL4uJNuUblhwmLa0RIm2hXBxOJpipNRziSZDG5i8nQkDrZYq2Kp1L0GVwcuaZBkRM5ZkDeg+i08WDYfSPlWUMU0ReCG3G8ei8N2fxj6rg0k3jKbgloXrtqbNJAc10OItYkwOaDQ7tFRp2dN+WnVWDa7an+2CeJh0R1Xi3DEB+QUgH6ufVBdf7K7ZnaTHueaLaFN2UEuHiYQADPS4jmg9PjMQDyvcLm13wJ526BShVqViM9J1I/S+DPNpGqmPGVpHtvQYmCWmZMCeZjgvOYzAVWy97C0G+oJE8tQOa9RgHEgFpAe52VhdZoJ3n1XO2hspoqMqGTUc0tqVJJztIIIPIHcg84iFWx32ToCjKRSUDBFLKIKCxqiDSigjkpTvVRQBI5OSlJQKmhVymQPCuw1n03iczHiI6g/kVmlPRdcdQRyINkHoMRgaWc1SB3jazmgkhogTEcTc3WzBYSmwEsDQXCCbuAbvAncsz8Uys+7BMC/NbK0BtrAC8cEHP2W0f6oZdwjgBFl77MC4A/T8rwvZyi7OajrSTlFpyzM/K9o55BFtwKDLtXZry7vGEuO9kkH+kpMDiIkOzgz5X5yLBaf4kGnK8QCbHnwPNbBUBEh0jggytbmJcfKPQlcLblaXECy7WLrLzWOu+UD/w5lehBN2Nc8GSIc0yD91m2hiXMoAk+LLbdd1gfufRdPZjh3TgSfN5bZSd0ry+38cKjsrSS1pufqfp7AWQcwIpQjKAlCFEQggCIQlRBYxFBhUQM9yrJRcq3IISlS5kZQBRQpUDEqSgSgg6mHxJBYZsZnqFs2tiyWRuOsWlcWm7y8nfdWY7EwS12ojhcIH2djK9MkU/HPlBkEcpC9FTwGLxgzur1MNltkplsucNS8wSRykLl7J2lh2RLgLagGx5ld/C7aoNcSHyDcBt0HYwmAa2nkqE1XGznHwn+ngqKGak7JMjceI3Kv8AjlI6Ek8MjvuAs+IxrnaU6ggy1xaQDxF+qDbiHSVysWLrdUXK2lWDZQcfaGOqsdla4BjmeodxHCy5S3Y27cx1mywEoGCMpEUDSoCllQFA6iCMoLGKJWqIGKRwTEpUFRSK4hI5qBCVCiiECkIgqFQIGzQD7+oV2NcyqxrrZtDp+izrNmyGNx8qDds4tY69JhnQkT7r1Oy8Q5pjuKQHFt7+oC4OzK7LWEyL716ijWDYIi+8Qg6tLMbuJH/Ftk76liTwWf8A1zQNRPyuJtjtAyiPEb8BclBuq17lefxzzVqQPI3zHcXcFzv45UxDhTpNLQ43e60N4hdzDUgwQNGxJ3udu9SUHH264spnLfLDiDvA1+JXMw9YPaHDf8EahdHtM8tpOnV5y/m6OULzOycVDiw6OMtn6uCDtoyggEBCIQRAQMiEoTBBYwKKMKiCFKQoXJS5AQUCoCgSgUqFGEEAQTIIAkrUg4QffeOYVzGE/qbD3SYh4psL3bhYcTuCDlOqPpOgGY941HqtTe0VRogtJ9dFk2Ue9p1y7UVWPdxykEEjpPwkdhyTHPdvug2Ve0tZ1mgN63S0fEc1Ql7t5N0aOxyDJOvHQLs7P2UDB14DcOaC7ZdPLBiXus1o4L0VDDZR4ru+AShgsMKfUi7jqsPafancUTB8b/Cz11d6D8kHke1OOFWqQPIzwjgT+IrgGn7/ACtDipFkHT2bju8GVx8Y1GmYcRz5LaCvNObfmN61Yfab2mHeMRro4fqg7iiroV2vEtM8QbEdQrkECYJAmBQXMUUYogDgq3BWuSOCCtRGEIQSUr3QJJAHE2CWtUiAAS4+Vo/FH2HNczEYYE5q7s53Ux/tsP8A9INDtrU9GB9Ux+AQ0P5uNohZXbUeT5A0fU4z9tSqMTioECw3MFhCqwYzOEnmg7+DLql3eX6dBbeQuNt/aHeOyN8jfk8V09o4jumZdDEHpwXlXkmSg7/Y138x7To5sH0utWI2fUpkloLmA2IuWt58uaq7IYRxe2BckuPTRvwvTH+e4Oa4ii1zgzIS01nNcWueYvlDgQBp4ZQVbKoOrMBIyt4m5PQLu0KLWCAIHPWeqvwjf5Ym5Fibajf62Kd5AQKHAC/D4XzntDtPv6xI8jfCzhG8+69J2t2iadPIDBfYAaxvXhJugslRV5lA5A6oOp6x7f3VzVkAOoOpJg87oOvsnAd9mc6cjOEAF303WnbRpMDRTMVIBeGd40hxuSb6RFlwBUggkEEaO3t6HcnL815mdSTJPUoNbMbUH43dJlbcLtUmzgHf9PNzJHsuXRMEE6AgmIJ14GxXSbjyXktrVKfhgnwMc5s+XwDRB28NVDx4TPEaEdQVFwX1Wkyxrmcc785J6wCgg9E5I4qF6UlAqpxmJFNhcfQcSrZXIxtTvKzGfhbd3pc/vmg203mnTzPvVeMzz9LfwsHAQuNisSXGfhX4/FF7uSwOCCd7cEieWoPUbwtlPaAa2BTsHAwBlBeDILr33W4ABZ6NKddNT6KypWBADRDR7k8SgOLql7iTprdUUMPnIaPxODfcwniy2bGpA1Wl05GS98XdlAiG8ySAg7GPqnC4fwWrV/BSOmSl5S/2XewrqdKmxrTAbTaymHBzZa0QInWdfVYaWBNSp31UCRo0GW02t8lJvIbzvM8V1q7A9uVwDmnUOAIPUIM3Z3bHeYithzb+Wx7P+wkPA/pyH0XWx2LbRY57yA1ov1uAOf8AdeWdssYbEjFUi1rGx3jNIb5XZeURZcXtJts4p8NkUWn+W3TN/wA3c+W5Bl2ntF2IqOqO0Nmj6WjQLCkrVMrRAm8dOarZiPqHqEGlAFXYKrTvmkh0CRlJDJ8cT5SRbMJi8KVYIzQ1suIDWzA97xwCComAf3qqwmqHQepUCCQgaQOvuLFMQiEFTw4aGRwOqek4G0QeBRhBwnqg00nIK7AYF9YltNpcQJIkCAog75CBTFKUCELgVHHM93GGj5n8l2cfVyMceXyVxCZYOd/XT8kFaGW6JK3YOjl8TvQILGUhTYSYzEQAdwNiVygFvxtXMCsDPyQOdF1+y1IGqSfws9iT86LkP0Xb7JtnvHcwPiUHqi+TpAGg4D9U7qkCdBxNoWfMvMdodsZ5pUz4R/uPH4j9I6IKe0G2TWcWMMUgf/YRvPLguKSoq3vjrwQO51uchTcq6YJ1VsIKjS4WTAO+r806ICANH71J6pmoBFA0okqtGUBcUrX7kHFVoNdN5bdpIOljFlFXQQQetcUpKhQQcbbWInw7gb8yq20jDQB+EfMqjGeKoRxf+a2Co5ps0H/qY+6CylhA27onglrVJVVSqTq149nKp9QaA+9igWo6Vnp/mrXFUsNz0lBa82XpuytOKM73PcfmB9l5cnRej2fiu6oM+rII63J+6C3b+0CwZGHxEXI/C3lzXlSI5LRjsaJN8ziZPVYPE7XTcEEfVmzfdFlKOqcCFEERlCVEBBRlJKhKAyigEQgikoEoII42QaESiEFlJBCmig9Wi5AlLVdAJ3IPP071SeBJ+VsrOgcz9liwbxLidY9xJRqPJP2QXNcYn2VZpzulXUmW1A9RqkquDBJI5AHUoM9UBth7LN+L0T5pMn/CUtv0QW02g6nK0eZx3f3UxWPL/AwZWAQPqIHFUlhd04bk+WEFTKQF96sKkqAoFhPh6Ze4NFpm/CBMo57a/wCVWXGZBIPFpgoOxhOzlauCaL6NRwBLqWfu6rGtIkkOERfiubi8HUpEtqMLSNbtcP8AyaSF1dn9qsVRbkzUqtPLkyVqVNxNM6szABxB4GVz8ZtA1M0U2UwTZlMkNYODQdAgxSOI9wpK6exsfSpd4arM7jTimC0PaH3IJnS8LLssUnVGCqQKdzU1bma1pdAPExHqgoUW9uDpuFnX5FrkWbJJNniA0uMg2DRJQc1QHVKD++SVqCwlEFLCJQO0qKMRQerWDaNbUDdqtOLq5RO9cnEu8PMoOex8GUH1HExMDlr7qtzkQ6/ogOX9m5TU2AItRQElCn+5SPVtMQ3mUEP7hLKtaP7oOCCuEoddEqstQWEIFUgkae25M2qN9umiC0KINvpdFBECOiZBAmQcFax7mzle9toMEwWnUQlJUKCo2/XT4SNKOIFh1VbSgvJTJE4QM0qItCKDs4h+b3WSvSkQTCSpVWSo6+v3QJiaYbYHMfsko7+SV5lGlY70Fs/IUCXIRaDvThp4IEqBWkqoXI6q5w5oGCUlEFLHMfKAyClDbIOjj8f3TsfGuiCp7VWQtVWNb/CpIHP7IKcsGysp1J1iUR+7pCBw+6AurgaCfsk74ncB7lNA4QlhBO8PAfKcV+I+UhCLUAquzDSAqmLSwKl2tkDtVrVWwSrWhBYwIIsaogcVJVVQJG2TEyEFJCBCLkqC6ncdJH2Ucno+U9T+SQiUDYcX03K7L7paDLFWMElBUCi9O9mqUtQVFEpsqDmoK5RJTZFAwoK0SnaDdQtQVFT2TOYVAwoETAIhqVwKAFyqAutDWJKjbhAArGo5UzGoHplRFjVEH//Z'
    if autor_choosed=="Churchill":
        img_link='https://cdn.britannica.com/25/171125-050-94459F61/Winston-Churchill.jpg'
    elif autor_choosed=="Hitler":
        img_link='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Hitler_portrait_crop.jpg/220px-Hitler_portrait_crop.jpg'
    elif autor_choosed=="Chamberlain":
        img_link='https://upload.wikimedia.org/wikipedia/en/9/9a/Neville_Chamberlain_by_Walter_Stoneman.jpg'
    elif autor_choosed=="Eisenhower":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Gaulle":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Gobbels":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Himmler":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Hirohito":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="King Georges":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Mussolini":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Patton":
        img_link='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBYWFRgWFhUYGBgaHBwZHBwcHBwcHB4YGhocHBwaHBwcIS4lHB4rIRgYJzgmKy8xNTU1GiQ7QDs0Py40NTEBDAwMBgYGEAYGEDEdFh0xMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMf/AABEIAQYAwQMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAADAAIEBQYBBwj/xABBEAACAQIEAwUFBQYFAwUAAAABAgADEQQSITEFQVEGImFxgRMykaGxQlLB0fAHFCNystJiksLh8RWCohYkQ1Py/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AM5xbFk1qwfX+I/9bSvrm9sptpJ/Fj/Hq6f/ACP/AFtITUrm8BgHhygvZ6ybvp0gar290QAunWEwlBnOVFLHwF7ePhBs99Dzmh7GcZTC1WLqCjKQevhaBX8U4E9Ei5BuLkjl4GVhoTW9oO1YrLkSmEU+9exYzLVmJ1GggCVeRnclpxtNj5w6ISL8t7nQQAvcDwvGIOdo6o6c29AL/WI1k273nAQbwgalQbWhlXNqLxtRNNrwGZRa8YyadI9FBG0SoepFoDNdtpxLqes6p9TH5L+sBFOcLTe41EjioQSIZXY6DSA4opFxOhdoNhyMcjC2m4gOL8rawQHjCm+htYzjqBe8BWHWKD9sOkUC/wCIrerWI/8Asf8AraQfaHnvDY+rlr1d7e0f+toxmVrC8B/O8HiUP2QI9l6G84HIgRghMERbzh2ck6aRqJc2OhgCte19I1X9RClgptYmR8TUy961jsB49fSARqgU7ZmOy/n+UBima9nax5ga2HTTQeUDRVwwygs72C2uSWY6HzPIeN956HwbsvRwyBqqrVraE5tUQ72UHRrdTAxnCuz+JxOtKk7D757qert3fnNVw/8AZvWcDPXoJfkGZz47Lb5xvaTtaUQ0kYhgbG3IW5cpSYDtDUXKfa5XIIzuxCqNbW6QNZX/AGcFDTT95Qs5IVSrDNbUhTrdra200BPIyJxH9nWLQFkyVQPutr8CATNzh8Q9Gl/Gql7BCyqre1VsquQVQZhysBqQdt5b8OxdJKWakrMGAq5Bmz2cnVUqG4U5WIGl7Gw1geA4igysVcZXBttbUciIBkI0OhnvvG+z1DHUwzKFqFQyvlsdRoGG5HgdRPFuN4JqFVqbizKcpHiNvMEEEeBgUxW2pj6b35TmIQbgmMS8B7oLgnnHM3JRBpfNHE63MDrFucIguNBrBu/O+nKERx5QGlyN45wWGugiG/hOl2HLSAHJFH5x0igXPEKimrU0+2/9Zle7AHeScWL1an87/wBRkdkB5QCK2m8IzjTKOUiulrRmoOhgGsROMvW8YjnXNHPUgBL62GniZXYvEXqa6gb/AIywfqdhKGq92vA9K/ZxwYlWxb63LLTHjs7/AOkW/wAUv+Iq2pvrBYSpUTDYNKCFk9mmaxAszBCXa+41ckDW5EmcSgeO8Qe9Rr7k3J8YBmBAGu1j9JfcZ4NVesxSmzKdbgaAnU3OwkOhwds2RnUN91P4rf5aYYiBu+w+Oau71Kjl6rkBnYm5A2uNu7c200BIFrz0DC8MUMKveFQDICWY3pAi3O1gTsRpraeb9nOEVcO2ZKNVztZ2pUl+bO1vQGaZePYxSqqmGS52dy9gN7sHHUcoG84fRUd4KoYDLcCxyg6AnmBMP+1vgeekuJQd5LI/ih91j5E2/wC7whW4/jVF/b8MJ+6DVv5X2vKXifaXiWJpPROFplHUqzIrsbHmvfOvpA83Yx0nYnhNakpFSi6jcMyOott9oDwleHgdZQORvOR4cnf0nSgG4gLKekaU66QiNlFoKpfeAqekI9XwkdXvtOZjAL7Twikj9wf7s7AkYpiKjgfff+owd53EuC723LMfixggOpgFVoOotxpFbpOBoHL23g3tCunO0C5OwgR67kA+RkLAYhFZy658ylVW2hZiNb30ta995YVKZ1HUSmdO9YA329djaB6H2a7X0aOHWlUDMVzZSoJ0JuFPQ94gEchyguK9r2cEoi0k5M5zMfJP/wBekwn7xk7q2vzbf0X8/wBGOoLG+56wL+vx5T7+esf8bFVHkoNh8Iw8fxLAKh9mvIIth57a+ci8PwBLAlb9BNVgGp0xd0uNBcC4vvZdDrp0gVOD4Zia2rVahHO7NYDx1/CWtTslVByh+m+u+v4yxxfGqaIfZMwYNYqwFrc7aAjyM0HYjGisaruQTbN8ByEDF/8AQ3R8t206Df4SD2gwVahTD95Lta9zqD5zf8U43Z/4aBwtsx0tqQLBd21PhfleRe3lRavDXcIUKOgIPi4FxqRY67HlA8qp8Yrjaq/+Y2+EnYHEl9TuOfWUZl9wVLJc8z8oEo35jnC8tpx7E6QtNOsCO9oNxJD26wbjSBFFOxiYW1O07n1+U5X2I5AEwNtFLD2HgIoGHr++1uZP1MEHsd9Y9mBJHO5iVNdYHbb8jBh4R3HrBsnMbwCM0ZfrOI5FgfjFU1gIt11lFiu6zAbn6S9y8jYeMz+MqXqseht8NIEa0uOHYa48ZX4WiWJ9PraarhmFtvvAl4bhrG2U2vYeNudukvsDwakBdgSBra9tepIMbgGGgsNJJdyRYaDWBmeMYVDUIpqFUW2Ftpf9gaVqoRtnUqfUfOU+Mc+1VAL3ufQby84MSlWm9stmXT5GBqOEdl0o12FSzo1imml73sfHaN/a3RVOGVcqgAtRAA20cHb0mofGKK3s3AAZQyHky8/Ig/hMx+2K/wD0/ILd+qgueQGZr/8Aj84HgGGoF2Cj1PQTR0KVgByG0Dg6ARbD1PWSlRuUDrsPWMWo2sIKZvtrE4AFusANQA3MZqd4m30nWsQesAVRNrSPVGYEyRUbS99INxpptaB6paKSvYjrOwPMClyT5xo3iZz841G1JgNdBtEg11jtJIwtIFwHOVesCIz6ATtRbcrR2JUBrA3HIxuU8zAMFBAmd4lhyjkgXBNwZfk8tpzJbnAj4TGUciksFItcW1v4dR4zQ0Keo1lMy2Gw+EsqFS2UnoPwgXdMC3jJeGC6knujUyvSqLA35WjnqEgKPMiBAxitUqGoLLuAPDpb0l32a4K/ceo7MhqBBsCCQT01Gg+MqKuKysURC7De2wv4m1/SavsvUxD90KoF1IDvl7w2OgJuPxgbPiHBlcK+Y50N1Y+d7W2AmY/ay5/dKA61Rf0pv+fymjr4utTqItVVKtfvISQG+6wKj0PgdpjP2w4vTD0Rv36h8NlX43f4QPOAPSPpuw53EFTbTrHqCANYB85vpzjWGl9zG5dY935WECO51jBcnUbSSddLQFRSu8AddBygKmgPSxkjlIjqbHpYwPZ8kUXs4oHlN44wbG52tCAQGsvjHBfHURCdAG94HE1J0iYEcoQrzERN/KA1m01GsCFtrDI28cBAjpUOoI/4lrRAekp5i4+BP5Sjx2JyqbbgbyRwfiIRQj+6dVboxABB8DYH4wLrDVLXUmSaVXKN7+Mrs/5zvtjbTWBYYdO9e3O9+c2XZq7NYvYXsPXx8Zi8NiM1he35zTcNxCIVuTYam21wLXJ8L3gej18OCgBO1j8J4Z214p+8Yuo6m6p/DT+VL3PqxY+RE13F+1ZxIahRY+zRW9o+2ZipApqegvcny8Z5zw/CGrQVwbsBlI6ldBr1tADTNvWHUk3+EEAQQCCOtxtHU3IuB1gPL2ItBu55iPDW5RjE3gKnU3nax9Y59BtAMxaA1jppI1Z+6bb2MNUEY6Wub8jA9bzeJinMsUDzVucQ00nIx2N720gOZo+lbwlrwbs3Uro1UtTo0hpnqtkUkbhdCWI8o/CrgKJc4mt7axslOhcZ9L5mY2IXW1hrv6gDhnCa+IJ9jSd7Ak5RoLa+8dL+F7mW3DuxmJq0jWbJRQZtarFGupK2Kkd27C1yRIC9vayUzRw1FaVIhwVsxvn3Yu7ZswFrEdNpnTxPElGT2jZXUIwLswKKcwUg6WB19T1MCZgsjVkSs6ombK7DvhVG5BW+bwIvvGcXxFP2j/u2cUdlzas1t3Ogyg8lOoG+ugqhTue8xbmRstvLn6x7wI2K1Uw+FptVUBEZ2tfKoLHQa2A15RjjfyneBY16NVTTNqgNkbLnKs3dzKv2mF9BrryvaAcYh6DtTqA3RmRluCVZSVI06EESywuMptsbHnJXb7h5XFFnrCq1RQ5OYtlLFjkW5JyLoAedj0maSjvrqBy3/wB4GmSvRU3Z7DwPykfH8dbEEUqC5E2Y8z6yhqcPqjL7RWUNqt9L/OXHCsOVKhSFYkBSSBZibDU6DUjeBPxFb93wzKumnqcw39byP2aa2HI55vrJH7UMYWxApeyFIoqqVAtfKAAdO7bKBbKAN995zhdPKi6Wuov5g/7/ACgS+KPZc66spB0UNcXv3kPvAcxvaF43xfDYzI9KglKoFIqBCACRbKQthtruL7DW0ivUKt8R+H5SDieHo9yws42ZdG9bQNQOxebDriaVdXurOUKsjZV0IUalmBuOQ21kDH9lsTRpiq9Jglrk2BKgm3etqOW9t5nfYV09ysSOjdDyvvLrA9tsXSp+xdfaUwpUAWYWJvqLXOviYFS730g0UW3mxavwvGUQFX93xNrnIO7f7pVmvYjw0PO0qeOdl6mHpisHSrSJy50v3SeTKRdSYGetvB117p05GJ3M5UBKnyges+1MUDrFA8253vOudNACeV9h4mD2zHlHBrjSAJqWa3tHZwNluQg62UTllUWVQo8BGNUEbe9oD738hBu+lhHO2loN2GkB9MaeJ+ka5l5wnjWCREWrRd3VizHKjKwNRHAIYgnu0wlr2s7/AHjK/i+LSrXd6aZEJBVcqJYBQLZU7u4JuN73OpgQLcuci4hCLMuhHTfwkpd7zrJcEdR84Gk4iiVMBhqlPDsopl6b1fdVmbI2itcsuZqgWzAAq+guAKrgNMPiKanxfr7u3zt8JG4dxqqlCphQxVHcMQDl7wuGzW94EWBB07otJvZGk7YhqqgFUKqfAMwzEeQU/EQNZ2+yLgkbQv7VQmmo0JYfAayj4LVptUpM7FEYqxYWutjckAg5jcDu7m9hqRL/AParw1zQw7K2YZ3Istt1W1yDqd9bCYLgvGDQBa12S+UEKQGbYnMDazXPds2gAIuTA5iqgrYqo4YupdiGb3it9CdTc5QBe+s0i6KPAjbodPjeZ/guHtbSaB9iDtawOu41+UCPiW15xytfUeR/Xx+En8N4zhkUe2w7uwJF1Sm6Mt6pBOYr3v4oHMH2VM8jINfFUmrOaIZabksqOAGUE3y90sNNQNYA3MFcgXEfiBZvA6TtNO78fhAGwRxZ0UnqRqD57iTMHWqorItQvTewem5JBA6NuDcA632lYgN4I4sq2lxAn47BBVzrfLexB3B5XI3lS7WDeRlxhqmei+u4LW8Rr+Eq8SllN+YMD0/WKTPZ+UUDyZhvIiYm2n6/W0nVzYHylO0A2aOLwOfSP3EDi1OcMrgjnAokWUrqIElRBOvh/wARyPcfr4R1oArxIZ0jWcECJjBlYkaZhcefP9eM1/YOn/Ac2+2fkqzMcQpXS4+yb+nOansXUthiP8bfhA9B4ggxGDUbmmQ3mLEEfAzybtHw5VxHc0VhmYdGH56fOehYfiDU0Yg2FtjPP3qmpUd21LHTpbYQJeAp2Ek1nGlo9KdgBtA1IA0tzAPoPxhl32/XxjAOY8o2i96iJ3iGdFdlFyoZgDbkWAN7QO4xe6bHc33va3S/W/ynMGb6frTlL7CcFS5p1GdXLslwpyXFwr57EFc4GbbTnB8G4ShzFlr92mXFkdsr5Qcr5UNhqQfKBR0Kd2bw1+EpeItZiPG02TcPYjOlKpYBs90cWA1zG40FiQf5ZiMcb1CPGBb4F8iAnUHTzH6vIeJJAK9L29JJI/8Ab/yt9f8AiMxhBUHmVv8AK34QPZMnhOQ2WKB4piXuALa3lbV3tLPFbj1MqKrawHU9jJCHwkaiNSP1pJI9PiIBlAnSBaNuOo+InLmAGoCDcDzHX/eEpuGGnw6HxjiCYJqJGoNj+tD4QH28ZycSoDoRZvl6H8I9oBUQEWOx0+MseytUKroT3gx+GkqkffeEo1sjhvvaHz5fKBpON4qyEDdtJUYCgbi3Kdx1bMQBsPr+UPg16AnygTyv639JGqrc2HhDNilW4tdtsoNz6nYchB8J429GujlKdr3uRcjY2Ba4UnXvZSdYGw4J2BqVKLPUGR2FkV+6LG3eYqSwNr2Ful+k0K8MehkwoyISFrJUBbKKlDIrArYDvKRfX7THlMinayqHAq4p2COVYoWAZeoVAildNCDfvbG2uWx+Kao5u7uMxy53LaHzJIgelV+PNUoNUL0aZStldLFmUNYEglxfUgnQc+kWD4pkY3qqQysubNRyEe0ys1vaFtiHHkZ55wqo1N82VSLFSLkXBH8unKWQ4xVOJq13RT7RcmQEhQLILgde4D5lusD0bh2Do1FVv3vvMhDITT0zqVZStgw3PPlPA+JUsmJqITcK7Jfrla156b/6tqNiVqey7ihiVzm7FsmpOXUdw6W+1PLuNCzseedz/wCR/KBdGmfZOvQg+mv5yFiR3FO1lZfx/GW1AZ6Bbqq/KVeJUeyN+RPzX/aB7jlEUJYRQPCsb18LfGUlQ96XOOY85R1D3oEpLZ7eH6+snZAJUvUKvfTYS1DXAIPSBwAGPFJOYEm4zgtSlQo4h8qpWZgi375VVvny/d218R1Er2I3YhR4mARKSeXl/wAzrhVBa57oudTIZxyLtmby0HxMZUxjOMoUKDvzJgPSzrqNfzgWqMu4zL8x684SgCIYtfcQBU3De6b+HP4RtYk90b7esZXwo3GkdggxYg3YjRQBcljoALak+AgTXrIgGYknoNz+UC3Eaj91O4u3d94jxMlY/s1icOFqYig9MOdCwFr72JBNj/hNjoekClgBbnAscAgRfrfc/nIeOx+UqwFypBt1tuD5jSFzXGkr8at/jA170UYB1UWYAjQbEAgyO7WO28zmF4xWpqEGVlGwYXsOgO9pOp9oUbR0KHqveHwOv1gXeHa+w2+scKt9Rz/X1kTA4lHYFGVrWuo336HWXSYGm2CXEI59otd6VZDYZWzMyZRvbLlPO9+VjAHQN7fCYXtCLVLeLX88xmtxGJKI5U6qpN+hOg9bkTE8Te7jmbC/mdTA1nAtcPY9DaV2OQii/h+Rlj2cH8MDwP0kPiNO1OqOikwPcMonJI9jFA+fMe12MpK+8ucXuTKeudYDK+oB9JPpMcqkb2F/ESvLaWljQ91elhAdjMZUYrdmcqoRSzFgiDZUBOgkQLckE3sdz5SXkkcr3z5A/WA5aQhFpWMYBaGR4HEawEWeJk00gyCIDnlt2NxQp4pHZsgzZc5A7rHZrnQbMt+jHrKYmFwDe/8A9p+sD2/9p3F6IwD0WZWqVcuRAQWBVwxc9FAXc7kjrPEwh0Hnyt+toVKlhYKLeAnWqQFTJtB1V1X+Zf6hHZ41je3mv1EB4pCJ8ECLxwGp9ZKw2oMCmw+Du7jUZRcEdZecOrvYZwCTYFrd4gaC55kX3gcCg9tUH+Bfxk+ktmUeUBvHGFLDqB9plv6d76gTHs5ZrzS9stBRF9LN8e7+czmES7CBsez7d4L4fhGcYFkq/wApgeDtaosk8Y92tf7rQPc7RTmWKB8741LEymrDWajtVhTSrultrEeREytVoAnkzAMSpHTUeshF5Owa5Vv119NYEznAEd8/yj6mGJkdm7x8h9TAJFtGZoiYEhH3HifrHsgPSQs2vzhUqwGVKZEdgD7/AI5f9U69W41kns7hRWxCUi2UVGAJG9gGNhf7RtYeJgIp5QbtPY+0PYPBfuVSpQQ03p02qK2ZmDZFLENmJBBsdRtf0njINxe/hbWB1zODdP5l/qETCcC95f5l/qECUALyVhhoZGp7/reSsONTAZw8/wAar/Kv4yxp++v62kDhzfxap8FH1k+h72/I/QwK7tnUBFEDox+OUfhKXh6W1Mu+1lC5U8lCj43/ANpSUH5QLzhRu4k3jXuVf5DK/hfvA+M0eF4UcT7RL2BW5/XrA9dikr90PWKB47+0rCAOjrswZSfFTceupnmlVDcz3TtfwcVsNUCgl076DxXcDzW88XxKqDaBXMstEGw8B9JXPaWL7Kw6A/KB0nWCY970H1MI/WBZ7EeOkB4nCIs0RaAw/l9BFe0Vtfh9I9lgNDQnD7hmKmzLlYHobwREJw895/5f9QgbjiHbfEVsOcOVRFZQtRlLFmH2h3j3Aee9+omVKRubXpEXgNyiMfceY+s7eMdtR/MPqIBqb2NvH8ZY4c6n9c5Ute585Y4V7mB3A+/VPkPlJ9D3h5H6SBhDY1CbasPko/OSqdbUc/8AeAPtOxs46BfmVmaw+pl92gq5ldr3DOqj01/0ypwNK+sC3wAsRPQeyNAnO46Aeu9vlMNg6d2Gk9Z4Lgv3fDqp99u8fM8oG1sYo6KBQ4al9ec8j/ad2U/d6pxCL/CqHWw0R+YPQHces9ipJpGY/D08Qj0XGZGGVvzHQg6wPmB0k6mbovlb4aS17W9lq+DchlLU79yoASpHIMfst4H0lPgx3SvMaiB3TkfSBrC4tCut9YFkMBRyvBlZy0Ao/XlFmjM04WgPAvtH4BCXKqpZmAVVAuSxYWAAgbybwLFCnWDsbLqrNYEqrgqSMwtexI16wL7inY/G4eiK9WhanoWKsr5QbWLBSbDUa6jrM8H1ntXaHtXhl4dWT2yPUq0npqiMrEl1KCpYahR71z0tvaeKDzgPUzrpcWiQx6bwHKubcan1v4iHpUbC8fh7jnJgUW0/5PhAhrhSANf0d5MNLIl767SQqDe0FiTmIAgU3FvcRepLfAW/Ex+ApZVA66zuObNVsNkAX1G/zJmj7KcCbEProi6s3+keMC/7FcAJH7y69xTZL/abr5CbfGlbXI5QlF0FIIugXQDlYSFxFzlAvA2V4py5igUVFnFxpv1P5SSlO21tYooBThlqKyMoKMCGU6gg6HT4zw7tF2IfD4llpshTdcxbMFb7Jstja418IooFe/Zit96n/mb+yDbsxW+9T/zN/ZFFA4/ZTEW96l/mb+yB/wDS1f71L/M39kUUDo7K1z9ql/mb+yMPZiv96n/mb+yKKBGfglUG10+Lf2xYThT97VfievlFFAKnCanVPif7Y/8A6Y3VfifyiigJeGPpqvxPj4SVR4DVIuCnqW/tiigTKfZ+qNb0/i39sn0uBVSLZqfLm3Uf4YooBKnBat96fxbn/wBsEOz1bcNT2PNvP7vUTkUBnDexdZnVS9PvGxOZiddz7k9S4dwX2QWlTyqqr1Ny3U6RRQD1MI52Kj4/lIOIwtRrC676b+HhFFA2k5FFA//Z'
    elif autor_choosed=="Roosevelt":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Stalin":
        img_link='data:image/jpeg;base64,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'
    elif autor_choosed=="Truman":
        img_link='data:image/jpeg;base64,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'
        
    print(card_value)
    return fig4,fig,card_value,img_link



# Running App
if __name__=='__main__':
    app.run_server(port=8051)
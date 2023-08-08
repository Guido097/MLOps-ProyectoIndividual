from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import JSONResponse
import json
import pandas as pd
import numpy as np
from dateutil.parser import parse
import ast
import csv
import codecs
import pickle
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



app = FastAPI()

rows = []
with open('steam_games.json') as d: 
    for line in d.readlines():
            rows.append(ast.literal_eval(line))
df = pd.DataFrame(rows)
df.dropna(subset=["release_date"], inplace=True)
df.drop_duplicates(subset=['id'])
df.dropna(how="all")
df["metascore"] = df["metascore"].replace({" 'NA'": 0, pd.NA: 0})
df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
df["release_date"] = df["release_date"].apply(lambda x: parse(str(x)).strftime('%Y-%m-%d') if pd.notnull(x) else None)
df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
df["release_year"] = df["release_date"].dt.year
df.dropna(subset=["genres"], inplace=True)
df.dropna(subset=["specs"], inplace=True)
df.dropna(subset=["sentiment"], inplace=True)
df.dropna(subset=["app_name"], inplace=True)
df.drop("reviews_url", axis=1, inplace=True)
df.drop("url", axis=1, inplace=True)
df["release_year"] = df["release_year"].astype(int)
df["genres"] = df["genres"].astype(str)
df["app_name"] = df["app_name"].astype(str)

# Dividir los géneros y convertirlos en una lista de géneros únicos
df['genres']  = df['genres'].str.replace('[','',regex=True).replace(']','',regex=True).replace("'","",regex=True)
df['genres'] = df['genres'].str.split(', ')
df = df.explode('genres', ignore_index=True)
# Aplicar Label Encoder a la columna 'Genres'
label_encoder_genres = LabelEncoder()
df['encoded_genres'] = label_encoder_genres.fit_transform(df['genres'])

unique_release_years = df["release_year"].unique()
unique_release_years_str = [str(year) for year in unique_release_years]

@app.get("/")
async def root():
    return { ''' Hola! Este es un proyecto individual para la carrera de Data Science de Henry. Te recomiendo leer el README'''}

@app.get('/genre/')
def genero(year : str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado["genres"] = df_filtrado["genres"].str.replace(r'\[|\]', '').str.replace("'", "").str.split(", ")
        lista_generos = [genero for sublist in df_filtrado["genres"] for genero in sublist]
        generos_lanzados = pd.Series(lista_generos).value_counts()
        top_5_generos = generos_lanzados.head(5)
        return top_5_generos.index.tolist()
    else:
        return {"Error!: The introduced year is not in database"}

@app.get('/games/')
def juegos(year : str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado = df_filtrado.drop_duplicates(subset=['app_name'])
        df_filtrado["app_name"] = df_filtrado["app_name"].str.replace(r'\[|\]', '').str.replace("'", "").str.split(", ")
        juegos_lanzados = df_filtrado["app_name"].tolist()
        return juegos_lanzados
    else:
        return {"Error!: The introduced year is not in database"}

@app.get('/specs/')
def specs(year: str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado = df_filtrado.drop_duplicates(subset=['app_name'])
        lista_specs = [spec for sublist in df_filtrado["specs"] for spec in sublist]
        freq_specs = pd.Series(lista_specs).value_counts()
        top_5_specs = freq_specs.head(5)
        resultado_dict = top_5_specs.to_dict()
        return resultado_dict
    else:
        return {"Error!: The introduced year is not in database"}

@app.get('/earlyaccess/')
def earlyacces(year: str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado = df_filtrado.drop_duplicates(subset=['app_name'])
        juegos_early_access = df_filtrado["early_access"].sum()
        if juegos_early_access == 0:
            return "That year were not released games with early access"
        else:
            return f"That year, {juegos_early_access} were released with early access."
    else:
        return {"Error!: The introduced year is not in database"}


@app.get('/sentiment/')
def sentiment(year: str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado = df_filtrado.drop_duplicates(subset=['app_name'])
        sentiment_counts = df_filtrado["sentiment"].value_counts()
        filtered_sentiments = sentiment_counts.index[~sentiment_counts.index.str.contains("user reviews")]
        filtered_counts = sentiment_counts.loc[filtered_sentiments]
        resultado_dict = filtered_counts.to_dict()
        return resultado_dict
    else:
        return {"Error!: The introduced year is not in database"}

@app.get('/metascore/')
def metascore(year: str):
    if int(year) in unique_release_years:
        df_filtrado = df[df["release_year"] == int(year)]
        df_filtrado = df_filtrado.drop_duplicates(subset=['app_name'])
        df_filtrado["metascore"] = pd.to_numeric(df_filtrado["metascore"])
        top_5_juegos = df_filtrado.nlargest(5, "metascore")
        resultados = top_5_juegos.set_index("app_name")["metascore"].to_dict()
        return resultados
    else:
        return {"Error!: The introduced year is not in database"}





#Limpio la columna Metascore y normalizo
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['metascore'].fillna(0, inplace=True)
df['metascore'] = df['metascore'].astype(int)
#Limpio la columna Price y normalizo
df['price'] = df['price'].replace(['Free To Play', 'Free', 'Free HITMAN™ Holiday Pack','Free to Play','Play for Free!','Free Mod','Free Demo', 'nan'], '0')
# Convertir los valores a números (opcional)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
#Relleno los nulos con 0
df['price'].fillna(0, inplace=True)


#Saco outliers
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
df_outliers = df[(df['price'] >= lower_limit) & (df['price'] <= upper_limit)]

#Creo el modelo de ML para que el RMSE quede cargado en la variable "rmse"
X = df_outliers[['encoded_genres', 'early_access', 'release_year', 'metascore']]
y = df_outliers['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear y ajustar el modelo
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
# Predecir el precio en el conjunto de prueba
y_pred = model.predict(X_test)
force_row_wise = True
# Calcular la raíz del error cuadrado medio (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


label_encoder = joblib.load('label_encoder.pkl')


@app.get('/predict/')
def predict(genre, early_access : bool, metascore:int, year:int):
    # Obtener el valor codificado del género usando el LabelEncoder
    genre_encoded = label_encoder.transform([genre])[0]
        
    # Crear un DataFrame con las características ingresadas
    data = pd.DataFrame({
        "early_access": [early_access],
        "metascore": [metascore],
        "release_year": [year],
        "genres_encoded": [genre_encoded],
    })
    
    # Realizar la predicción del precio utilizando el modelo entrenado
    price_predicted = model.predict(data)[0]

    return {'Precio': price_predicted, "RMSE del modelo": rmse}
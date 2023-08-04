from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import JSONResponse
import json
import pandas as pd
from dateutil.parser import parse
import ast
import csv
import codecs

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/')
def genero(year : str):
    df_filtrado = df[df["release_year"] == int(year)]
    df_filtrado["genres"] = df_filtrado["genres"].str.replace(r'\[|\]', '').str.replace("'", "").str.split(", ")
    lista_generos = [genero for sublist in df_filtrado["genres"] for genero in sublist]
    generos_lanzados = pd.Series(lista_generos).value_counts()
    top_5_generos = generos_lanzados.head(5)
    return top_5_generos.index.tolist()

@app.get('/')
def juegos(year : str):
    df_filtrado = df[df["release_year"] == int(year)]
    df_filtrado["app_name"] = df_filtrado["app_name"].str.replace(r'\[|\]', '').str.replace("'", "").str.split(", ")
    juegos_lanzados = df_filtrado["app_name"].tolist()
    return juegos_lanzados

@app.get('/')
def specs(year: str):
    df_filtrado = df[df["release_year"] == int(year)]
    lista_specs = [spec for sublist in df_filtrado["specs"] for spec in sublist]
    freq_specs = pd.Series(lista_specs).value_counts()
    top_5_specs = freq_specs.head(5)
    resultado_dict = top_5_specs.to_dict()
    return resultado_dict

@app.get('/')
def earlyacces(year: str):
    '''df["release_date"] = pd.to_datetime(df["release_date"])'''
    df_filtrado = df[df["release_year"] == int(year)]
    juegos_early_access = df_filtrado["early_access"].sum()
    return print('En el año', year, 'fueron lanzados', juegos_early_access, 'con early access.')

@app.get('/')
def sentiment(year: str):
    df_filtrado = df[df["release_year"] == int(year)]
    sentiment_counts = df_filtrado["sentiment"].value_counts()
    resultado_dict = sentiment_counts.to_dict()
    return resultado_dict

@app.get('/')
def metascore(year: str):
    df_filtrado = df[df["release_year"] == int(year)]
    df_filtrado["metascore"] = pd.to_numeric(df_filtrado["metascore"])
    top_5_juegos = df_filtrado.nlargest(5, "metascore")
    resultados = top_5_juegos.set_index("app_name")["metascore"].to_dict()
    return resultados
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
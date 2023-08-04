from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import JSONResponse
import json
import pandas as pd
from dateutil.parser import parse
import ast
import csv
import codecs

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
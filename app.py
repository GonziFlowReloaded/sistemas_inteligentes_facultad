#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import io
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from rna.procesamiento.prueba_manual import (
    load_resources,
    preprocess_text,
    predict_class,
    MAX_SEQUENCE_LENGTH,
    classify_comment
)

app = FastAPI()

# montar directorio de plantillas y estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# cargar modelo al arrancar
@app.on_event("startup")
async def startup_event():
    global sentiment_model, word2idx, idx2word, labels
    sentiment_model, word2idx, idx2word, labels = load_resources()

# ruta principal: formulario de carga de CSV
@app.get("/")
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# procesar CSV y generar datos para dos gráficos separados
@app.post("/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode('utf-8')))

    # Conteo de Pedidos vs Reclamos\    
    class_counts = {"Pedido": 0, "Reclamo": 0}
    # Lista de etiquetas de sentimiento para los clasificados como "Otro"
    sentiment_list = []

    for row in reader:
        texto = row.get("comentario", "")
        cls = classify_comment(texto)
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            seq = preprocess_text(texto, word2idx, MAX_SEQUENCE_LENGTH)
            label, conf, _ = predict_class(sentiment_model, seq, labels)
            sentiment_list.append(label)

    # Conteo de sentimientos
    sentiment_counts = {}
    for s in sentiment_list:
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    # Preparar datos para la plantilla
    class_labels = list(class_counts.keys())
    class_values = list(class_counts.values())
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_values = [sentiment_counts[k] for k in sentiment_labels]
    total = sum(class_values) + sum(sentiment_values)

    return templates.TemplateResponse("report.html", {
        "request": request,
        "class_labels": class_labels,
        "class_values": class_values,
        "sentiment_labels": sentiment_labels,
        "sentiment_values": sentiment_values,
        "total": total
    })

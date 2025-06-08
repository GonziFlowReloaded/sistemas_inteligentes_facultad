#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Rutas a archivos necesarios ---
MODEL_PATH    = 'rna/procesamiento/final_model.keras'
WORD2IDX_PATH = 'rna/procesamiento/word2idx.json'
IDX2WORD_PATH = 'rna/procesamiento/idx2word.json' # Necesitaremos este para mapear de vuelta
LABELS_PATH   = 'rna/procesamiento/labels.json'   # Asumiendo que guardaste tus etiquetas

# --- Hiperparámetros (deben coincidir con el entrenamiento) ---
MAX_SEQUENCE_LENGTH = 100 # debe coincidir con el padding usado en el entrenamiento

def load_resources():
    """Carga el modelo, word2idx, idx2word, y etiquetas."""
    print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"[INFO] Cargando word2idx desde: {WORD2IDX_PATH}")
    with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    # Es útil tener idx2word para depuración o si se necesita reconstruir
    # Si no lo tienes, puedes generarlo a partir de word2idx
    idx2word = {int(idx): word for word, idx in word2idx.items()}

    print(f"[INFO] Cargando etiquetas desde: {LABELS_PATH}")
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    return model, word2idx, idx2word, labels

def preprocess_text(text, word2idx, max_len):
    """Tokeniza, convierte a índices y paddea el texto."""
    words = text.lower().split()
    # Convertir palabras a índices. Usar 0 para OOV (Out Of Vocabulary) o palabras desconocidas.
    # Asumimos que 0 fue el índice reservado para el padding/OOV durante el preprocesamiento.
    indexed_text = [word2idx.get(word, 0) for word in words]
    # Paddea la secuencia para que tenga la misma longitud que en el entrenamiento
    padded_sequence = pad_sequences([indexed_text], maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

def predict_class(model, preprocessed_input, labels):
    """Realiza la predicción y devuelve la clase más probable."""
    predictions = model.predict(preprocessed_input)
    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    # Mapear el índice a la etiqueta de la clase
    predicted_label = labels[str(predicted_class_idx)] # Asumiendo que las claves en labels.json son strings de los índices
    confidence = predictions[0][predicted_class_idx]
    return predicted_label, confidence, predictions[0]

def classify_comment(texto):
    """
    Clasifica un comentario como 'Pedido', 'Reclamo' u 'Otro' usando palabras clave simples.
    """
    texto_lower = texto.lower()
    palabras_pedido = [
        "quiero", "me gustaría", "solicito", "necesito", "podría", "quisiera", "por favor",
        "exijo", "demando", "pido", "solicitud", "esperaría", "sería bueno", "sería ideal", "me encantaría",
        "hagan", "haga", "hagan algo", "deberían", "debería", "propongo", "propongan", "proponga"
    ]
    palabras_reclamo = [
        "reclamo", "queja", "problema", "inconveniente", "molestia", "error", "fallo",
        "corrupción", "corrupto", "ladrones", "mentira", "mentiroso", "engaño", "engañoso", "vergüenza",
        "indignante", "injusticia", "basta", "no hacen nada", "no sirve", "no funciona", "estafa", "roban",
        "desastre", "fracaso", "decepción", "decepcionante", "abandonados", "abandono", "impresentable"
    ]
    for palabra in palabras_pedido:
        if palabra in texto_lower:
            return "Pedido"
    for palabra in palabras_reclamo:
        if palabra in texto_lower:
            return "Reclamo"
    return "Otro"

def main():
    # 1) Cargar recursos
    model, word2idx, idx2word, labels = load_resources()
    num_classes = len(labels)
    print(f"[INFO] Modelo cargado con {num_classes} clases.")
    model.summary()
    print("-" * 50)

    while True:
        text_input = input("\nIntroduce un texto para clasificar (o 'salir' para terminar): ")
        if text_input.lower() == 'salir':
            break

        # 2) Preprocesar el texto
        preprocessed_input = preprocess_text(text_input, word2idx, MAX_SEQUENCE_LENGTH)
        print(f"[DEBUG] Secuencia preprocesada: {preprocessed_input}")

        # 3) Realizar la predicción
        predicted_label, confidence, all_predictions = predict_class(model, preprocessed_input, labels)

        # 4) Mostrar resultados
        print(f"\n[RESULTADO] Texto: '{text_input}'")
        print(f"            Clase predicha: '{predicted_label}'")
        print(f"            Confianza: {confidence:.4f}")
        print("\n[TODAS LAS PROBABILIDADES POR CLASE]:")
        for i, prob in enumerate(all_predictions):
            print(f"  - {labels[str(i)]}: {prob:.4f}")
        print("-" * 50)

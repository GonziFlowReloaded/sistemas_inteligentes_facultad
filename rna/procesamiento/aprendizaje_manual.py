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
LABELS_PATH   = 'rna/procesamiento/labels.json'

# --- Hiperparámetros (deben coincidir con el entrenamiento) ---
MAX_SEQUENCE_LENGTH = 100
FINE_TUNE_EPOCHS = 2   # Pequeña cantidad para no sobreajustar

def load_resources():
    print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"[INFO] Cargando word2idx desde: {WORD2IDX_PATH}")
    with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    print(f"[INFO] Cargando etiquetas desde: {LABELS_PATH}")
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return model, word2idx, labels

def preprocess_text(text, word2idx, max_len):
    words = text.lower().split()
    indexed_text = [word2idx.get(word, 0) for word in words]
    padded_sequence = pad_sequences([indexed_text], maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

def predict_class(model, preprocessed_input, labels):
    predictions = model.predict(preprocessed_input)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[str(predicted_class_idx)]
    confidence = predictions[0][predicted_class_idx]
    return predicted_label, predicted_class_idx, confidence, predictions[0]

def one_hot(idx, num_classes):
    arr = np.zeros(num_classes)
    arr[idx] = 1
    return arr

def main():
    # 1) Cargar recursos
    model, word2idx, labels = load_resources()
    num_classes = len(labels)
    labels_inv = {v: int(k) for k, v in labels.items()}  # Para mapear texto->índice

    print(f"[INFO] Modelo cargado con {num_classes} clases.\n")

    while True:
        text_input = input("\nIntroduce un texto para clasificar (o 'salir' para terminar): ")
        if text_input.lower() == 'salir':
            break

        preprocessed_input = preprocess_text(text_input, word2idx, MAX_SEQUENCE_LENGTH)
        predicted_label, predicted_idx, confidence, all_probs = predict_class(model, preprocessed_input, labels)

        print(f"\n[RESULTADO] Texto: '{text_input}'")
        print(f"  Clase predicha: '{predicted_label}'")
        print(f"  Confianza: {confidence:.4f}")
        print("[PROBABILIDADES]:")
        for i, prob in enumerate(all_probs):
            print(f"  - {labels[str(i)]}: {prob:.4f}")

        correction = input(f"\n¿La predicción es correcta? [s/n]: ").strip().lower()
        if correction == 's':
            print("¡Perfecto! No se hacen cambios.")
            continue

        print("\nEtiquetas disponibles:")
        for k, v in labels.items():
            print(f"  - {v}")

        true_label = input("¿Cuál es la clase correcta? (Escribí exactamente como aparece): ").strip()
        if true_label not in labels_inv:
            print("Clase inválida. Se omite el aprendizaje.")
            continue

        true_idx = labels_inv[true_label]
        y_true = one_hot(true_idx, num_classes)

        # -- Entrenamiento puntual con el nuevo dato corregido --
        print(f"[APRENDIZAJE] Ajustando modelo con el nuevo ejemplo corregido...")
        model.fit(
            preprocessed_input,
            np.array([y_true]),
            epochs=FINE_TUNE_EPOCHS,
            verbose=2
        )
        print("[APRENDIZAJE] ¡Modelo actualizado!")

        # -- Guardar el modelo actualizado --
        save = input("¿Guardar el modelo actualizado? [s/n]: ").strip().lower()
        if save == 's':
            model.save(MODEL_PATH)
            print(f"[INFO] Modelo guardado en: {MODEL_PATH}")

if __name__ == '__main__':
    main()

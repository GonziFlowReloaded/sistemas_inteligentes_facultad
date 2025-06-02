#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

# -------- GPU CONFIGURATION (CUDA) --------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"[GPU] {len(gpus)} GPU(s) físicas encontradas, {len(logical_gpus)} dispositivo(s) lógico(s).")
    except RuntimeError as e:
        print(f"[GPU] Error al configurar GPU: {e}")
else:
    print("[GPU] No se detectó GPU. Usando CPU.")

# ------- Rutas a datos preprocesados -------
WORD2IDX_PATH   = 'word2idx.json'
X_TRAIN_PATH    = 'X_train_padded.npy'
X_TEST_PATH     = 'X_test_padded.npy'
Y_TRAIN_PATH    = 'y_train.npy'
Y_TEST_PATH     = 'y_test.npy'

# ------ Hiperparámetros ------
EMBEDDING_DIM         = 200
LSTM_UNITS            = 64
DROPOUT_RATE_BEFORE   = 0.3
DROPOUT_RATE_AFTER    = 0.3
BATCH_SIZE            = 32
EPOCHS                = 20
MAX_SEQUENCE_LENGTH   = 100  # debe coincidir con el padding
PATIENCE_EARLYSTOP    = 3
VALIDATION_SPLIT      = 0.1

def load_data():
    X_train = np.load(X_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test  = np.load(Y_TEST_PATH)
    return X_train, X_test, y_train, y_test

def get_vocab_size():
    with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    return max(int(idx) for idx in word2idx.values()) + 1

def build_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=EMBEDDING_DIM,
                  input_length=MAX_SEQUENCE_LENGTH),
        Dropout(DROPOUT_RATE_BEFORE),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
        Dropout(DROPOUT_RATE_AFTER),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=['accuracy']
    )
    return model

def main():
    # 1) Cargar datos
    X_train, X_test, y_train, y_test = load_data()
    num_classes = y_train.shape[1]
    vocab_size  = get_vocab_size()
    print(f"\n[VOCAB] {vocab_size} palabras (incluyendo OOV y padding).")
    print(f"[DATA] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[DATA] X_test:  {X_test.shape},  y_test:  {y_test.shape}\n")

    # 2) Calcular class weights (opcional, si dataset desbalanceado)
    y_true_labels = y_train.argmax(axis=1)
    clase_pesos = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_true_labels),
        y=y_true_labels
    )
    class_weights_dict = {i: peso for i, peso in enumerate(clase_pesos)}
    print(f"[CLASSES] class_weight → {class_weights_dict}\n")

    # 3) Construir modelo
    model = build_model(vocab_size, num_classes)
    model.summary()

    # 4) Callbacks
    checkpoint = ModelCheckpoint(
        'best_model.keras',  # guardamos en formato .keras nativo
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE_EARLYSTOP,
        restore_best_weights=True,
        verbose=1
    )

    # 5) Entrenar explícitamente en GPU si está disponible
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"[INFO] Entrenando en: {device}\n")
    with tf.device(device):
        history = model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights_dict,
            callbacks=[checkpoint, early_stop],
            verbose=2
        )

        # 6) Evaluación final
        loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)
        print(f"\n[Test] Loss: {loss:.4f}  |  Accuracy: {acc:.4f}\n")

        # 7) Guardar modelo final
        model.save('final_model.keras')
        print("[INFO] Modelo final guardado como ‟final_model.keras”.\n")

if __name__ == '__main__':
    main()

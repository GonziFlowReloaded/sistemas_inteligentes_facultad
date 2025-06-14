import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import time

# Carga del dataset
df = pd.read_csv(r'tp2\train_and_test2.csv')

# Normalizar nombres de columnas
df.columns = df.columns.str.lower().str.strip()
# Renombrar columna de supervivencia si tiene nombre atípico
if '2urvived' in df.columns:
    df = df.rename(columns={'2urvived': 'survived'})

# Eliminar columnas irrelevantes
drop_cols = [c for c in ['passengerid','name','ticket','cabin'] if c in df.columns]
df = df.drop(drop_cols, axis=1)
# Eliminar columnas residuales de dummy (ej. 'zero', 'zero.1', etc.)
zero_cols = [c for c in df.columns if c.startswith('zero')]
df = df.drop(zero_cols, axis=1)

# Imputar datos faltantes
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Separar características y objetivo
X = df.drop('survived', axis=1)
y = df['survived']

# Codificar 'sex' si es texto
if X['sex'].dtype == object:
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})

# One-hot encoding para 'embarked'
X = pd.get_dummies(X, columns=['embarked'], drop_first=True)

# Normalizar variables numéricas
numeric_cols = ['age', 'fare', 'sibsp', 'parch', 'sex']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Balanceo por oversampling de la clase minoritaria
df_bal = pd.concat([X, y], axis=1)
df_majority = df_bal[df_bal.survived == 0]
df_minority = df_bal[df_bal.survived == 1]
df_min_up = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_min_up])
X_bal = df_balanced.drop('survived', axis=1)
y_bal = df_balanced['survived']

# División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# Definición de tres modelos MLPClassifier
models = [
    {
        'name': 'Modelo_1',
        'clf': MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam',
                             learning_rate_init=0.001, max_iter=50, batch_size=32, random_state=42),
        'params': {
            'Capas ocultas': '[16]',
            'Activación': 'relu',
            'Optimizador': 'adam',
            'LR': 0.001,
            'Épocas': 50,
            'Batch size': 32,
            'Dropout': 'No'
        }
    },
    {
        'name': 'Modelo_2',
        'clf': MLPClassifier(hidden_layer_sizes=(32,16), activation='tanh', solver='sgd',
                             learning_rate_init=0.01, max_iter=100, batch_size=16, random_state=42),
        'params': {
            'Capas ocultas': '[32,16]',
            'Activación': 'tanh',
            'Optimizador': 'sgd',
            'LR': 0.01,
            'Épocas': 100,
            'Batch size': 16,
            'Dropout': 'No'
        }
    },
    {
        'name': 'Modelo_3',
        'clf': MLPClassifier(hidden_layer_sizes=(64,32,16), activation='relu', solver='adam',
                             learning_rate_init=0.0005, max_iter=50, batch_size=64, random_state=42),
        'params': {
            'Capas ocultas': '[64,32,16]',
            'Activación': 'relu',
            'Optimizador': 'adam',
            'LR': 0.0005,
            'Épocas': 50,
            'Batch size': 64,
            'Dropout': 'No'
        }
    }
]

# Entrenamiento y evaluación
results = []
for m in models:
    start = time.time()
    m['clf'].fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = m['clf'].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    row = {
        'Modelo': m['name'],
        **m['params'],
        'Tiempo (s)': round(elapsed, 2),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'Confusión [TN, FP, FN, TP]': cm.flatten().tolist()
        }
    results.append(row)

# Mostrar resultados

df_results = pd.DataFrame(results)

print("\nResultados de los modelos:")
df_results.to_string(index=False)


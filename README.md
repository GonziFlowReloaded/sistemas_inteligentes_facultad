# Integrador SI - Análisis y Clasificación de Comentarios

## Descripción General

Este proyecto integra procesamiento de lenguaje natural, aprendizaje automático y visualización para analizar y clasificar comentarios ciudadanos. Permite identificar automáticamente si un comentario es un pedido, un reclamo o pertenece a otra categoría, y además realiza análisis de sentimiento sobre los comentarios.

Incluye:
- Un agente experto basado en reglas para clasificación básica.
- Un modelo de red neuronal entrenado para análisis de sentimiento.
- Un backend web (FastAPI) para cargar archivos CSV y visualizar resultados.
- Scripts y notebooks para preprocesamiento y experimentación con modelos (incluyendo datasets tipo Titanic).

## Estructura de Carpetas

- `main.py`, `agente_experto.py`, `comentarios.py`: Lógica principal y agente experto basado en reglas.
- `app.py`: Backend web con FastAPI para cargar y analizar comentarios vía CSV.
- `rna/procesamiento/`: Preprocesamiento, entrenamiento y prueba de modelos de sentimiento (Keras/Tensorflow).
    - `procesamiento_csv.ipynb`: Limpieza y preparación de datos.
    - `prueba_manual.py`, `aprendizaje_manual.py`: Prueba y ajuste manual del modelo de sentimiento.
    - Modelos y recursos: `final_model.keras`, `word2idx.json`, `labels.json`, etc.
- `titanic/`: Ejercicios y experimentos de aprendizaje automático con el dataset Titanic.
    - `rna.py`, `rna2.ipynb`: Modelos MLP para clasificación de supervivencia.
    - `datasets/`: Datasets y scripts de preprocesamiento.
- `templates/`: Plantillas HTML para la interfaz web.
- `static/`: Archivos estáticos para la web.
- `docs/`: Documentación adicional (ej: infografía).

## Instalación de Dependencias

Instala las dependencias con:
```bash
pip install -r requirements.txt
```

## Ejecución del Agente Experto (Reglas)

```bash
python main.py
```
Esto clasificará los comentarios de `comentarios.py` como Pedido/Reclamo u Otro.

## Ejecución del Backend Web

```bash
uvicorn app:app --reload
```
Luego accede a `http://localhost:8000` para cargar un archivo CSV con comentarios y ver los análisis.

El CSV debe tener columnas: `usuario,comentario`.

## Entrenamiento y Prueba del Modelo de Sentimiento

1. Preprocesa los datos:
   - Ejecuta el notebook `rna/procesamiento/procesamiento_csv.ipynb` para limpiar y vectorizar los comentarios.
2. Entrena el modelo (requiere scripts/notebooks adicionales, no incluidos aquí).
3. Prueba el modelo manualmente:
   ```bash
   python rna/procesamiento/prueba_manual.py
   ```
4. Ajusta el modelo con ejemplos corregidos:
   ```bash
   python rna/procesamiento/aprendizaje_manual.py
   ```

## Experimentos con Titanic

En la carpeta `titanic/` hay notebooks y scripts para experimentar con modelos de redes neuronales sobre el dataset Titanic, incluyendo preprocesamiento y comparación de arquitecturas.

## Requisitos
- Python 3.10+
- Ver dependencias en `requirements.txt`

## Créditos
- Proyecto académico para la materia Sistemas de Información.
- Autor: José Gonzalo Scali

---

# Agente Experto para Clasificación de Comentarios

Este proyecto implementa un agente experto en Python para clasificar comentarios de usuarios como "Pedido/Reclamo" u "Otro", utilizando una lista de palabras clave. Es un trabajo práctico para la materia Sistemas Inteligentes.

## Descripción
El sistema toma una lista de comentarios de usuarios y, mediante un agente experto, determina si cada comentario corresponde a un pedido/reclamo o a otro tipo de mensaje. La lógica se basa en la presencia de palabras clave asociadas a pedidos o reclamos.

## Estructura del Proyecto
- `main.py`: Script principal. Ejecuta la clasificación y muestra los resultados.
- `agente_experto.py`: Implementa la clase `AgenteExperto` con la lógica de clasificación.
- `comentarios.py`: Contiene la lista de comentarios de ejemplo (diccionarios con usuario y comentario).
- `README.md`: Este archivo.

## Requisitos
- Python 3.10 o superior

## Instalación
1. Clona este repositorio o descarga los archivos.
2. Asegúrate de tener Python instalado (https://www.python.org/downloads/).

## Ejecución
Desde la terminal, navega a la carpeta del proyecto y ejecuta:

```
python main.py
```

Esto mostrará por pantalla cada comentario, el usuario y la clasificación asignada ("Pedido/Reclamo" u "Otro").

## Ejemplo de Salida
```
Usuario: mariadelcerro88
Comentario: Esperamos que esta vez sí se cumplan las promesas, ya pasaron varios años con lo mismo.
Clasificación: Otro
----------------------------------------
Usuario: correntinorabioso
Comentario: Ya basta de chamuyo, hagan algo concreto. Estamos cansados de promesas.
Clasificación: Pedido/Reclamo
----------------------------------------
...etc
```

## Personalización
Puedes modificar la lista de palabras clave en `agente_experto.py` para ajustar la lógica de clasificación, o agregar/quitar comentarios en `comentarios.py`.

## Contacto
Para dudas o sugerencias, contacta a: [Tu Nombre o Email]

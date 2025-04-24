from comentarios import comentarios
from agente_experto import AgenteExperto

if __name__ == "__main__":
    # Extraer solo los textos de los comentarios
    textos = [c["comentario"] for c in comentarios]
    
    # Instanciar el agente experto
    agente = AgenteExperto()
    
    # Clasificar los textos
    resultados = agente.agente_experto(textos)
    
    # Mostrar resultados junto al usuario y comentario
    for comentario, es_pedido_o_reclamo in zip(comentarios, resultados):
        usuario = comentario["usuario"]
        texto = comentario["comentario"]
        clasificacion = "Pedido/Reclamo" if es_pedido_o_reclamo else "Otro"
        print(f"Usuario: {usuario}\nComentario: {texto}\nClasificación: {clasificacion}\n{'-'*40}")

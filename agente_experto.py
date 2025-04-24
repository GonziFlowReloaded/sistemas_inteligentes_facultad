# agente_experto.py

class AgenteExperto:
    """
    Clase que representa un agente experto para clasificar textos.
    """
    def __init__(self):
        pass

    def agente_experto(self, lista_textos):
        """
        Función principal que recibe una lista de textos y los clasifica.
        """
        resultados = []
        for texto in lista_textos:
            texto_preprocesado = self.preprocesar_texto(texto)
            clasificacion = self.clasificar_texto(texto_preprocesado)
            resultados.append(clasificacion)
        return resultados

    def preprocesar_texto(self, texto):
        """
        Preprocesa el texto recibido (limpieza, normalización, etc).
        """
        # TODO: Implementar preprocesamiento
        return texto

    def clasificar_texto(self, texto):
        """
        Clasifica el texto recibido según la lógica del agente experto.
        Retorna True si el texto es un pedido o un reclamo, False en caso contrario.
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
        for palabra in palabras_pedido + palabras_reclamo:
            if palabra in texto_lower:
                return True
        return False

    def mostrar_resultados(self, resultados):
        """
        Muestra o devuelve los resultados de la clasificación.
        """
        # TODO: Implementar presentación de resultados
        pass

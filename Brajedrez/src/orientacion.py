from src.nucleo import CLASES

def esta_girado(etiquetas: list[int], margen: float = 1.0) -> bool:
    """etiquetas de las 64 casillas tal como se ven (fila 0 = arriba). Solo se gira si hay pruebas claras:
    peones blancos, de media, más de `margen` filas por encima de los negros (o, sin peones, el rey blanco
    3 filas o más por encima del negro). En las posiciones de ONCE la diferencia real va de 1 a 4,4 filas."""
    def fila_media(clase: str):
        filas = [i // 8 for i, e in enumerate(etiquetas) if CLASES[e] == clase]
        return sum(filas) / len(filas) if filas else None
    pb, pn = fila_media("P"), fila_media("p")
    if pb is not None and pn is not None:
        return pb < pn - margen
    rb, rn = fila_media("K"), fila_media("k")
    return rb is not None and rn is not None and rb <= rn - 3

def enderezar(etiquetas: list[int]) -> list[int]:
    """Gira 180° un tablero visto desde las negras."""
    return etiquetas[::-1]

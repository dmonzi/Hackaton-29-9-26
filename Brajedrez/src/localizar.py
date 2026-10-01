import cv2
import numpy as np
from src.nucleo import ordenar_esquinas

def quitar_marco(recorte: np.ndarray) -> np.ndarray:
    """Quita el margen blanco y el marco negro impreso que rodean el tablero en los diagramas de libro.
    Sin esto, las casillas del borde llevan un trozo de marco y toda la cuadrícula queda desplazada."""
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if recorte.ndim == 3 else recorte
    def corte(perfil: np.ndarray) -> int:  # medias de gris desde el borde hacia dentro
        k = max(3, int(0.06 * len(perfil)))
        oscuras = [i for i in range(k) if perfil[i] < 100]
        if oscuras:  # marco negro: se corta justo después, más una línea de transición
            return min(oscuras[-1] + 2, k)
        i = 0
        while i < k and perfil[i] > 240:  # solo margen blanco
            i += 1
        return i
    filas, cols = gris.mean(1), gris.mean(0)
    a, b, i, d = corte(filas), corte(filas[::-1]), corte(cols), corte(cols[::-1])
    alto, ancho = gris.shape
    return recorte[a:alto - b, i:ancho - d]

def localizar_tablero(imagen: np.ndarray, salida: int = 400, en_perspectiva: bool = False):
    """Tablero recortado y recto (salida x salida) o None si no hay tablero.
    Por defecto supone captura o escaneo (tablero recto): recorta el rectángulo del marco.
    Con en_perspectiva=True (foto con el móvil) corrige la perspectiva del cuadrilátero."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    bordes = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0), 50, 150), np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.25 * gris.size or not 0.8 < w / h < 1.25:
            continue
        if not en_perspectiva:
            return cv2.resize(quitar_marco(imagen[y:y + h, x:x + w]), (salida, salida), interpolation=cv2.INTER_AREA)
        aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(aprox) == 4:
            origen = ordenar_esquinas(aprox.reshape(4, 2).astype(np.float32))
            destino = np.float32([[0, 0], [salida, 0], [salida, salida], [0, salida]])
            return cv2.warpPerspective(imagen, cv2.getPerspectiveTransform(origen, destino), (salida, salida))
    if 0.9 < ancho / alto < 1.1 and gris.std() > 20:  # la imagen ya es solo el tablero
        return cv2.resize(imagen, (salida, salida), interpolation=cv2.INTER_AREA)
    return None

def localizar_tableros(imagen: np.ndarray, salida: int = 400, area_min: float = 0.02) -> list[np.ndarray]:
    """Todos los tableros de una imagen (p. ej., una página de libro con varios diagramas), de arriba abajo.
    Si no encuentra ninguno, prueba con localizar_tablero() (imagen que ya es solo el tablero)."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0), 50, 150), np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= area_min * gris.size and 0.8 < w / h < 1.25 and cv2.contourArea(c) > 0.6 * w * h:
            cajas.append((y, x, w, h))
    cajas.sort()
    tableros = [cv2.resize(quitar_marco(imagen[y:y + h, x:x + w]), (salida, salida), interpolation=cv2.INTER_AREA)
                for y, x, w, h in cajas]
    if tableros:
        return tableros
    unico = localizar_tablero(imagen, salida)
    return [unico] if unico is not None else []

"""Utilidades comunes: clases, FEN <-> etiquetas y corte de casillas."""
import cv2
import numpy as np

CLASES = ["vacia"] + list("PNBRQKpnbrqk")  # 13 clases

def leer_imagen(ruta) -> np.ndarray | None:
    """Como cv2.imread, pero admite rutas con tildes o «º» en Windows."""
    try:
        return cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None

def fen_a_etiquetas(nombre_archivo: str) -> list[int]:
    """'1B1B1K2-3p1N2-...-1B6.jpeg' -> 64 etiquetas (a8..h8, a7..h1)."""
    colocacion = nombre_archivo.rsplit(".", 1)[0].replace("-", "/")
    etiquetas = []
    for fila in colocacion.split("/"):
        for c in fila:
            etiquetas += [0] * int(c) if c.isdigit() else [CLASES.index(c)]
    assert len(etiquetas) == 64, colocacion
    return etiquetas

def etiquetas_a_fen(etiquetas: list[int]) -> str:
    filas = []
    for r in range(8):
        fila, vacias = "", 0
        for e in etiquetas[r * 8:(r + 1) * 8]:
            if e == 0:
                vacias += 1
            else:
                fila += (str(vacias) if vacias else "") + CLASES[e]
                vacias = 0
        filas.append(fila + (str(vacias) if vacias else ""))
    return "/".join(filas)

def cortar_casillas(tablero: np.ndarray, lado: int = 32) -> np.ndarray:
    """Tablero ya recortado y recto -> array (64, lado, lado) en gris."""
    gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY) if tablero.ndim == 3 else tablero
    gris = cv2.resize(gris, (lado * 8, lado * 8), interpolation=cv2.INTER_AREA)
    return gris.reshape(8, lado, 8, lado).swapaxes(1, 2).reshape(64, lado, lado)

def ordenar_esquinas(p: np.ndarray) -> np.ndarray:
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])

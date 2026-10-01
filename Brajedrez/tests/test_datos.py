import os
import random
import cv2
import numpy as np
from src.ajustar import casillas_once
from src.evaluar import evaluar_once, evaluar_sinteticos
from src.nucleo import cortar_casillas, etiquetas_a_fen
from src.preparar_datos import lista_usados, preparar

def _vecino_mas_cercano(X, y):
    """Clasificador de sustitución (sin PyTorch) para comprobar que etiquetas y casillas van alineadas."""
    base = X.reshape(len(y), -1).astype(np.float32)
    def clasificar(tablero):
        c = cortar_casillas(tablero).reshape(64, -1).astype(np.float32)
        d = ((c[:, None, :] - base[None, :, :]) ** 2).sum(-1)
        return y[d.argmin(1)].tolist(), [1.0] * 64
    return clasificar

def test_etiquetas_de_once_alineadas_y_orientacion():
    X, y = casillas_once()
    assert X.shape == (640, 32, 32)
    resultados = evaluar_once(_vecino_mas_cercano(X, y))
    assert all(r["exacto"] and r["orientacion_ok"] for r in resultados)

def test_preparar_y_evaluar_con_dataset_de_juguete(tmp_path):
    X, y = casillas_once()
    carpeta = tmp_path / "datos"
    carpeta.mkdir()
    rng = random.Random(0)
    for _ in range(40):
        idx = [rng.randrange(len(y)) for _ in range(64)]
        img = cv2.resize(X[idx].reshape(8, 8, 32, 32).swapaxes(1, 2).reshape(256, 256), (400, 400),
                         interpolation=cv2.INTER_NEAREST)
        nombre = etiquetas_a_fen([int(y[i]) for i in idx]).replace("/", "-") + ".png"
        cv2.imwrite(str(carpeta / nombre), img)
    (carpeta / "portada.png").write_bytes(cv2.imencode(".png", np.zeros((10, 10), np.uint8))[1].tobytes())
    salida = str(tmp_path / "casillas.npz")
    preparar(str(carpeta), 30, salida=salida)
    usados = set(open(lista_usados(salida), encoding="utf-8").read().split())
    assert len(usados) == 30 and "portada.png" not in usados
    d = np.load(salida)
    r = evaluar_sinteticos(_vecino_mas_cercano(d["X"], d["y"]), str(carpeta), 10, usados)
    # Con etiquetas desalineadas el acierto caería muy por debajo; no es 100 % porque hay casillas no vistas
    assert r["tableros"] == 10 and r["acierto_casilla"] > 0.9

import csv
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.localizar import localizar_tablero
from src.marcas import casillas_resaltadas, flechas
from src.nucleo import fen_a_etiquetas, leer_imagen
from src.orientacion import esta_girado

FIX = Path(__file__).parent / "fixtures"
FILAS = list(csv.DictReader(open(FIX / "ejemplos_once.csv", encoding="utf-8")))

def _esperadas(texto):
    return {c: v.split() for c, v in (p.split(":") for p in texto.split(";"))} if texto else {}

@pytest.mark.parametrize("fila", FILAS, ids=[f["archivo"] for f in FILAS])
def test_localiza_resaltados_y_flechas(fila):
    tablero = localizar_tablero(leer_imagen(str(FIX / "img" / fila["archivo"])))
    assert tablero is not None
    girado = fila["girado"] == "1"
    assert casillas_resaltadas(tablero, girado) == _esperadas(fila["resaltadas"])
    esperadas = sorted(tuple(x.split("-")) for x in fila["flechas"].split()) if fila["flechas"] else []
    assert flechas(tablero, girado) == esperadas

@pytest.mark.parametrize("fila", FILAS, ids=[f["archivo"] for f in FILAS])
def test_orientacion(fila):
    normal = fen_a_etiquetas(fila["fen"] + ".x")
    visto = normal[::-1] if fila["girado"] == "1" else normal
    assert esta_girado(visto) == (fila["girado"] == "1")

def test_sin_tablero():
    assert localizar_tablero(np.full((600, 600, 3), 250, np.uint8)) is None

def test_madera_sin_flechas_falsas():
    # las vetas de la madera son naranjas como las flechas de chess.com
    tablero = localizar_tablero(leer_imagen(str(FIX / "img" / "madera.jpeg")))
    assert flechas(tablero) == [] and casillas_resaltadas(tablero) == {}

"""De un tablero recortado a la respuesta completa: FEN, audio, braille, marcas y dudas."""
from typing import Callable
import numpy as np
from src.formatos import a_audio, a_braille
from src.marcas import casillas_resaltadas, flechas
from src.nucleo import etiquetas_a_fen
from src.orientacion import enderezar, esta_girado
from src.validar import problemas

UMBRAL_DUDA = 0.8
Clasificador = Callable[[np.ndarray], tuple[list[int], list[float]]]

def _lista(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]

def analizar_tablero(tablero: np.ndarray, clasificador: Clasificador | None = None,
                     fen: str | None = None, girado: bool = False) -> dict:
    """clasificador(tablero) -> (etiquetas, confianzas) tal como se ven. Con fen se salta el modelo (pruebas)."""
    dudas = []
    if fen is None:
        if clasificador is None:
            return {"error": "Falta el modelo entrenado (modelos/casillas_v2.pt): todavía no puedo leer las piezas."}
        etiquetas, confianzas = clasificador(tablero)
        girado = esta_girado(etiquetas)
        if girado:
            etiquetas, confianzas = enderezar(etiquetas), confianzas[::-1]
        fen = etiquetas_a_fen(etiquetas)
        dudas = ["ABCDEFGH"[i % 8] + str(8 - i // 8) for i, c in enumerate(confianzas) if c < UMBRAL_DUDA]
    avisos = problemas(fen)
    if avisos:
        return {"fen": fen, "error": "No he reconocido bien la posición. " + " ".join(avisos)}
    resaltadas, fl = casillas_resaltadas(tablero, girado), flechas(tablero, girado)
    audio = a_audio(fen, resaltadas, fl)
    if dudas:
        audio += "\nNo estoy seguro de " + ("la casilla " if len(dudas) == 1 else "las casillas ") + _lista(dudas) + "."
    return {"fen": fen, "girado": girado, "audio": audio, "braille": a_braille(fen, resaltadas=resaltadas, flechas=fl),
            "resaltadas": resaltadas, "flechas": fl, "dudas": dudas}

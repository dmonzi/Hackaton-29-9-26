"""Clasifica las 64 casillas con el modelo entrenado (necesita PyTorch; sin probar en el entorno de origen)."""
import numpy as np
import torch
from src.entrenar import RedCasillas
from src.nucleo import cortar_casillas

def cargar_modelo(ruta: str = "modelos/casillas_v2.pt") -> RedCasillas:
    modelo = RedCasillas()
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    return modelo.eval()

def clasificar(tablero: np.ndarray, modelo: RedCasillas) -> tuple[list[int], list[float]]:
    """Etiquetas y confianzas de las 64 casillas tal como se ven (fila 0 = arriba)."""
    x = torch.from_numpy(cortar_casillas(tablero)).unsqueeze(1).float() / 255
    with torch.no_grad():
        confianza, clase = torch.softmax(modelo(x), dim=1).max(1)
    return clase.tolist(), confianza.tolist()

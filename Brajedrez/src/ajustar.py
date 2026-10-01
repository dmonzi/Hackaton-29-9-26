"""Ajuste fino con los diagramas reales de ONCE, mezclados con casillas sintéticas.

    python -m src.ajustar --reservar tablero4.png tablero6.png diagrama23.png   # para medir con 3 no vistos
    python -m src.ajustar                                                        # modelo final, con los 10
    python -m src.ajustar --base modelos/casillas_v2.pt --dificiles --rep-once 80 --salida modelos/casillas_v3.pt
        # además repasa las casillas de entrenamiento en las que el modelo base falla o duda
"""
import argparse
import csv
import os
from pathlib import Path
import cv2
import numpy as np
from src.estilo_libro import estilo_libro
from src.localizar import localizar_tablero
from src.nucleo import cortar_casillas, fen_a_etiquetas, leer_imagen

def casillas_once(carpeta: str = "tests/fixtures", reservar: set[str] = frozenset()) -> tuple[np.ndarray, np.ndarray]:
    """Casillas de los ejemplos tal como se ven en la imagen, con su etiqueta (girada si hace falta)."""
    X, y = [], []
    for fila in csv.DictReader(open(Path(carpeta) / "ejemplos_once.csv", encoding="utf-8")):
        if fila["archivo"] in reservar:
            continue
        tablero = localizar_tablero(leer_imagen(str(Path(carpeta) / "img" / fila["archivo"])))
        etiquetas = fen_a_etiquetas(fila["fen"] + ".x")
        if fila["girado"] == "1":
            etiquetas = etiquetas[::-1]
        X += list(cortar_casillas(tablero))
        y += etiquetas
    return np.array(X, np.uint8), np.array(y, np.int64)

def casillas_dificiles(modelo: str, carpeta: str = "datos/train", usados: str = "datos/casillas_tableros.txt",
                       umbral: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Casillas de los tableros de entrenamiento en las que el modelo falla o no llega al umbral de confianza,
    más su copia en estilo libro. Se guardan en datos/dificiles_<modelo>.npz para no repetir la búsqueda."""
    cache = Path("datos") / f"dificiles_{Path(modelo).stem}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    import torch
    from src.entrenar import dispositivo
    from src.reconocer import cargar_modelo
    dev = dispositivo()
    red = cargar_modelo(modelo).to(dev)
    X, y = [], []
    for nombre in Path(usados).read_text(encoding="utf-8").split():
        imagen = leer_imagen(os.path.join(carpeta, nombre))
        if imagen is None:
            continue
        casillas, esperadas = cortar_casillas(imagen), np.array(fen_a_etiquetas(nombre))
        with torch.no_grad():
            confianza, clase = torch.softmax(red(torch.from_numpy(casillas).unsqueeze(1).float().to(dev) / 255), 1).max(1)
        for i in ((confianza.cpu().numpy() < umbral) | (clase.cpu().numpy() != esperadas)).nonzero()[0]:
            X += [casillas[i], estilo_libro(casillas[i], oscura=(i // 8 + i % 8) % 2 == 1)]
            y += [esperadas[i]] * 2
    X, y = np.array(X, np.uint8), np.array(y, np.int64)
    np.savez_compressed(cache, X=X, y=y)
    print(f"{len(y)} casillas difíciles para {modelo} -> {cache}")
    return X, y

def mezclar(reservar: set[str], sinteticas: str = "datos/casillas.npz", n_sinteticas: int = 30000,
            repeticiones: int = 20, salida: str = "datos/mezcla.npz",
            dificiles: tuple[np.ndarray, np.ndarray] | None = None, rep_dificiles: int = 4) -> str:
    Xo, yo = casillas_once(reservar=reservar)
    d = np.load(sinteticas)
    idx = np.random.default_rng(0).permutation(len(d["y"]))[:n_sinteticas]
    Xd, yd = dificiles if dificiles is not None else (np.empty((0, *Xo.shape[1:]), np.uint8), np.empty(0, np.int64))
    X = np.concatenate([np.repeat(Xo, repeticiones, axis=0), np.repeat(Xd, rep_dificiles, axis=0), d["X"][idx]])
    y = np.concatenate([np.repeat(yo, repeticiones), np.repeat(yd, rep_dificiles), d["y"][idx]])
    np.savez_compressed(salida, X=X, y=y)
    print(f"{len(yo)} casillas de ONCE × {repeticiones} ({len(yo) * repeticiones / len(y):.0%} de la mezcla)"
          + (f" + {len(yd)} difíciles × {rep_dificiles}" if len(yd) else "") + f" + {len(idx)} sintéticas -> {salida}")
    return salida

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reservar", nargs="*", default=[], help="ejemplos que NO se usan, para medir después")
    p.add_argument("--base", default="modelos/casillas.pt")
    p.add_argument("--salida", default="modelos/casillas_v2.pt")
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rep-once", type=int, default=20, help="veces que se repite cada casilla de ONCE")
    p.add_argument("--dificiles", action="store_true", help="añade las casillas en las que --base falla o duda")
    p.add_argument("--rep-dificiles", type=int, default=4)
    a = p.parse_args()
    from src.entrenar import entrenar
    dificiles = casillas_dificiles(a.base) if a.dificiles else None
    entrenar(mezclar(set(a.reservar), repeticiones=a.rep_once, dificiles=dificiles, rep_dificiles=a.rep_dificiles),
             epocas=a.epocas, salida=a.salida, pesos_iniciales=a.base, lr=a.lr)

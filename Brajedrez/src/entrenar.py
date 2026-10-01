"""Modelo y entrenamiento del clasificador de casillas (13 clases), en CPU.

    python -m src.entrenar --npz datos/casillas.npz --epocas 8 --salida modelos/casillas.pt

Usa la tarjeta gráfica NVIDIA si PyTorch la detecta (torch.cuda.is_available()); si no, la CPU.

AVISO: escrito sin poder ejecutarlo (no había PyTorch donde se preparó). Prueba antes con 1 época.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset

class RedCasillas(nn.Module):
    def __init__(self, n_clases: int = 13):
        super().__init__()
        def bloque(e, s):
            return nn.Sequential(nn.Conv2d(e, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU(), nn.MaxPool2d(2))
        self.red = nn.Sequential(bloque(1, 32), bloque(32, 64), bloque(64, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_clases))

    def forward(self, x):
        return self.red(x)

def binarizar(x):  # simula la impresión en blanco y negro de un libro
    return (x > x.mean()).float()

AUMENTOS = T.Compose([
    T.RandomAffine(degrees=3, translate=(0.06, 0.06), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.4, contrast=0.5),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.3),
    T.RandomApply([T.Lambda(binarizar)], p=0.2),
])

class Casillas(Dataset):
    """Guarda las casillas en uint8 (poca memoria) y las pasa a float al pedirlas."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, indices: list[int], aumentar: bool):
        self.X, self.y, self.indices, self.aumentar = X, y, indices, aumentar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = self.X[j].unsqueeze(0).float() / 255
        return (AUMENTOS(x) if self.aumentar else x), self.y[j]

def dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # gráficas Intel Arc
        return torch.device("xpu")
    return torch.device("cpu")

def entrenar(npz: str = "datos/casillas.npz", epocas: int = 8, salida: str = "modelos/casillas.pt",
             pesos_iniciales: str | None = None, lr: float = 2e-3, lote: int = 256,
             trabajadores: int | None = None) -> float:
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    dev = dispositivo()
    if trabajadores is None:  # con gráfica, los aumentos en CPU pasan a ser el cuello de botella
        trabajadores = 4 if dev.type != "cpu" else 0
    d = np.load(npz)
    X, y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    orden = torch.randperm(len(y), generator=torch.Generator().manual_seed(0)).tolist()
    n_val = max(1, len(y) // 10)
    dl_tr = DataLoader(Casillas(X, y, orden[n_val:], True), batch_size=lote, shuffle=True,
                       num_workers=trabajadores, persistent_workers=trabajadores > 0, pin_memory=dev.type == "cuda")
    dl_va = DataLoader(Casillas(X, y, orden[:n_val], False), batch_size=1024)
    modelo = RedCasillas()
    if pesos_iniciales:
        modelo.load_state_dict(torch.load(pesos_iniciales, map_location="cpu"))
    modelo.to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * len(dl_tr))
    nombre = torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type
    print(f"{len(y) - n_val} casillas de entrenamiento, {n_val} de validación, {epocas} épocas, en {nombre}")
    mejor = 0.0
    for epoca in range(epocas):
        t0 = time.time()
        modelo.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            nn.functional.cross_entropy(modelo(xb), yb, label_smoothing=0.05).backward()
            opt.step()
            sched.step()
        modelo.eval()
        ok = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                ok += (modelo(xb).argmax(1) == yb).sum().item()
        acierto = ok / n_val
        print(f"época {epoca + 1}/{epocas}: acierto por casilla en validación {acierto:.4f} ({time.time() - t0:.0f} s)")
        if acierto > mejor:
            mejor = acierto
            torch.save(modelo.state_dict(), salida)
    registro = os.path.join(os.path.dirname(salida) or ".", "entrenamiento.json")
    historial = json.load(open(registro, encoding="utf-8")) if os.path.exists(registro) else {}
    historial[salida] = {"acierto_validacion": round(mejor, 4), "epocas": epocas, "casillas": len(y),
                         "desde": pesos_iniciales, "datos": npz}
    json.dump(historial, open(registro, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Mejor acierto {mejor:.4f}; modelo guardado en {salida}")
    return mejor

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="datos/casillas.npz")
    p.add_argument("--epocas", type=int, default=8)
    p.add_argument("--salida", default="modelos/casillas.pt")
    p.add_argument("--pesos", default=None, help="modelo de partida (para el ajuste fino)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--trabajadores", type=int, default=None, help="procesos para los aumentos (0 = ninguno)")
    a = p.parse_args()
    entrenar(a.npz, a.epocas, a.salida, a.pesos, a.lr, trabajadores=a.trabajadores)

"""Regla R3: detectar posiciones imposibles antes de describir."""

def problemas(fen: str) -> list[str]:
    colocacion = fen.split()[0]
    filas = colocacion.split("/")
    avisos = []
    if colocacion.count("K") != 1 or colocacion.count("k") != 1:
        avisos.append("Cada bando debe tener exactamente un rey.")
    if any(c in "Pp" for c in filas[0] + filas[-1]):
        avisos.append("Hay un peón en la fila 1 o en la fila 8.")
    if sum(c.isupper() for c in colocacion) > 16 or sum(c.islower() for c in colocacion) > 16:
        avisos.append("Un bando tiene más de 16 piezas.")
    return avisos

"""Salidas en los formatos de ONCE: audio (texto para lector de pantalla) y braille (signografía B8)."""
FILA_BRAILLE = {1: "⠂", 2: "⠆", 3: "⠒", 4: "⠲", 5: "⠢", 6: "⠖", 7: "⠶", 8: "⠦"}
BRAILLE_A_FILA = {v: k for k, v in FILA_BRAILLE.items()}
LETRA = {"K": "R", "Q": "D", "R": "T", "N": "C", "B": "A", "P": "P"}
NOMBRE = {"K": ("Rey", "Rey"), "Q": ("Dama", "Damas"), "R": ("Torre", "Torres"),
          "B": ("Alfil", "Alfiles"), "N": ("Caballo", "Caballos"), "P": ("Peón", "Peones")}
ORDEN_AUDIO = "KQRBNP"    # Rey, Dama, Torres, Alfiles, Caballos, Peones (ejemplos de audio de ONCE)
ORDEN_BRAILLE = "KQRNBP"  # R, D, T, C, A, P (ejemplos braille de ONCE)
FLECHA = "¬⠒⠕¬"

def piezas(fen: str) -> dict:
    """{'Blancas': {'K': [('e', 1)], ...}, 'Negras': {...}}, casillas por columna y fila."""
    res = {"Blancas": {k: [] for k in "KQRBNP"}, "Negras": {k: [] for k in "KQRBNP"}}
    for i, fila in enumerate(fen.split()[0].split("/")):
        col = 0
        for c in fila:
            if c.isdigit():
                col += int(c)
            else:
                res["Blancas" if c.isupper() else "Negras"][c.upper()].append(("abcdefgh"[col], 8 - i))
                col += 1
    for bando in res.values():
        for lista in bando.values():
            lista.sort()
    return res

def _lista(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]

def a_audio(fen: str, resaltadas: dict | None = None, flechas: list | None = None) -> str:
    """Formato de los ejemplos de audio de ONCE: 'Blancas: Rey en E2, Torres en C1 y H1, ...'."""
    lineas = []
    for bando, p in piezas(fen).items():
        grupos = []
        for t in ORDEN_AUDIO:
            if p[t]:
                nombre = NOMBRE[t][0] if len(p[t]) == 1 else NOMBRE[t][1]
                grupos.append(f"{nombre} en " + _lista([f"{c.upper()}{f}" for c, f in p[t]]))
        lineas.append(f"{bando}: " + ", ".join(grupos) + ".")
    for color, casillas in (resaltadas or {}).items():
        lineas.append(f"Casillas resaltadas en {color}: " + _lista([c.upper() for c in casillas]) + ".")
    if flechas:
        lineas.append("Flechas: " + "; ".join(f"de {o.upper()} a {d.upper()}" for o, d in flechas) + ".")
    return "\n".join(lineas)

def _b(casilla: str) -> str:
    return casilla[0] + FILA_BRAILLE[int(casilla[1])]

def a_braille(fen: str, peon_con_letra: bool = False, resaltadas: dict | None = None,
              flechas: list | None = None) -> str:
    """Formato de los ejemplos braille de ONCE (Documento técnico B8): 'Blancas: Re⠂ Dd⠂ ...'."""
    lineas = ["Tablero:"]
    for bando, p in piezas(fen).items():
        items = []
        for t in ORDEN_BRAILLE:
            letra = "" if t == "P" and not peon_con_letra else LETRA[t]
            items += [f"{letra}{c}{FILA_BRAILLE[f]}" for c, f in p[t]]
        lineas.append(f"{bando}: " + " ".join(items))
    notas = []
    colores = list((resaltadas or {}).items())
    if colores:
        txt = f"Resaltadas en {colores[0][0]} las casillas " + _lista([_b(c) for c in colores[0][1]])
        for color, cs in colores[1:]:
            txt += f" y, en {color}, " + ("la casilla " if len(cs) == 1 else "las casillas ") + _lista([_b(c) for c in cs])
        notas.append(txt)
    notas += [f"{_b(o)}{FLECHA}{_b(d)}" for o, d in (flechas or [])]
    if notas:
        lineas.append(")" + "\n".join(notas) + "(")
    return "\n".join(lineas)

def braille_a_fen(blancas: str, negras: str) -> str:
    """Convierte las líneas braille de ONCE a colocación FEN (para usar sus ejemplos como tests)."""
    tablero = [["" for _ in range(8)] for _ in range(8)]
    inv = {v: k for k, v in LETRA.items()}
    for linea, blanco in ((blancas, True), (negras, False)):
        for item in linea.split():
            letra, col, fila = (item[0], item[1], item[2]) if len(item) == 3 else ("P", item[0], item[1])
            t = inv[letra]
            tablero[8 - BRAILLE_A_FILA[fila]]["abcdefgh".index(col)] = t if blanco else t.lower()
    filas = []
    for f in tablero:
        s, v = "", 0
        for x in f:
            if x:
                s += (str(v) if v else "") + x
                v = 0
            else:
                v += 1
        filas.append(s + (str(v) if v else ""))
    return "/".join(filas)

# Evidencia: resultados en diagramas que el modelo no ha visto

Para medir con honestidad, el ajuste fino con los ejemplos de ONCE se hizo dos veces:

1. **Modelo de medida** (`modelos/casillas_medida.pt`): ajustado con 7 de los 10 ejemplos de ONCE. Se reservaron `tablero4.png`, `tablero6.png` y `diagrama23.png`, que el modelo no ve nunca.
2. **Modelo final** (`modelos/casillas_v2.pt`): ajustado con los 10 ejemplos. Es el que usa la web.

Los números de abajo salen de `modelos/metricas.json`, generado con `src/evaluar.py`.

## Diagramas de ONCE reservados (modelo de medida)

| Diagrama | Tipo | Resultado |
| --- | --- | --- |
| tablero4.png | Captura digital | Exacto (64/64 casillas) |
| tablero6.png | Captura digital | Exacto (64/64 casillas) |
| diagrama23.png | Diagrama de libro | 63/64 casillas. Falla f5 (ve dama negra en vez de blanca) con confianza 0,69, por debajo del umbral de 0,8, así que la herramienta **avisa**: «No estoy seguro de la casilla F5». |

Es decir: no inventa nada sin avisar. El único error en un diagrama nuevo se comunica como duda.

## Tableros sintéticos no usados al entrenar (500 tableros)

| Modelo | Acierto por casilla | Tableros exactos |
| --- | --- | --- |
| casillas.pt (base, solo sintéticos) | 99,90 % | 96,8 % |
| casillas_medida.pt (ajuste con 7 de ONCE) | 99,86 % | 95,6 % |
| casillas_v2.pt (final, ajuste con 10 de ONCE) | 99,88 % | 96,0 % |

## Efecto del ajuste fino con ONCE

| Modelo | Ejemplos de ONCE exactos |
| --- | --- |
| casillas.pt (base) | 6 de 10: las 6 capturas digitales; fallan los 4 diagramas de libro |
| casillas_v2.pt (final) | 10 de 10 (los ha visto al ajustar: no es una medida independiente) |

La medida independiente de los diagramas de libro es la de la reserva (arriba).

## Cómo reproducirlo

    python -m src.ajustar --reservar tablero4.png tablero6.png diagrama23.png --salida modelos/casillas_medida.pt
    python -m src.evaluar --modelo modelos/casillas_medida.pt --solo tablero4.png tablero6.png diagrama23.png
    python -m src.evaluar --modelo modelos/casillas_v2.pt

Requiere el dataset en `datos/train` y `datos/casillas.npz` (ver README).

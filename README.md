# Brajedrez

Cada diagrama de ajedrez, en voz y en braille. Proyecto para el reto de ONCE del Hackatón.

Brajedrez lee un diagrama de ajedrez (captura de pantalla, escaneo de libro, PDF o Word) y se lo cuenta a una persona ciega: dónde está cada pieza, qué casillas están resaltadas y qué flechas hay. Lo entrega como texto para lector de pantalla y voz, y como transcripción braille en el formato de ONCE. Si duda de una casilla, lo dice.

Qué es, por qué y cómo funciona: [Brajedrez/README.md](Brajedrez/README.md).

## Requisitos

- Python 3.10 o superior.
- Unos 2 GB libres para las dependencias (PyTorch).
- No hace falta tarjeta gráfica: el reconocimiento funciona en CPU. La gráfica solo sirve para reentrenar.

## Instalación

Desde la carpeta raíz del repositorio.

Windows (PowerShell):

    python -m venv .venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\.venv\Scripts\Activate.ps1
    cd Brajedrez
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Linux o macOS:

    python3 -m venv .venv
    source .venv/bin/activate
    cd Brajedrez
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Los siguientes comandos se ejecutan dentro de `Brajedrez/` con el entorno activado.

## Uso

### Web

    python -m src.web

Abre http://localhost:8000, sube uno o varios diagramas (PNG, JPG, PDF o Word) y pulsa «Analizar diagramas». Para cada diagrama verás:

- La descripción en texto, preparada para lector de pantalla, y un lector de voz integrado para escucharla.
- Avisos de las casillas de las que no está seguro.
- La transcripción braille en el formato de ONCE, con botones para copiarla o descargarla en .txt.
- La posición en notación FEN.

Opciones:

| Opción | Para qué |
| --- | --- |
| `--host 0.0.0.0` | Abrirla desde el móvil u otro equipo de la misma red: http://IP-del-portátil:8000 |
| `--puerto 8080` | Usar otro puerto (por defecto, 8000) |
| `--modelo ruta.pt` | Usar otro modelo (por defecto, `modelos/casillas_v2.pt`) |
| `--prueba` | Modo demostración sin modelo: solo reconoce los 10 diagramas de ejemplo de ONCE |

### Línea de comandos

    python -m src.cli describir tests/fixtures/img/diagrama23.png

Escribe la descripción, los avisos de duda y el braille. Por ejemplo:

    Blancas: Rey en H3, Dama en F5, Torre en F1, Caballo en E2, Peones en A6, C4, D5, G3 y G4.
    Negras: Rey en H7, Dama en G6, Torre en E7, Alfil en E3, Peones en A7, C5, E4, G5 y H6.
    No estoy seguro de la casilla F5.

    Tablero:
    Blancas: Rh⠒ Df⠢ Tf⠂ Ce⠆ a⠖ c⠲ d⠢ g⠒ g⠲
    Negras: Rh⠶ Dg⠖ Te⠶ Ae⠒ a⠶ c⠢ e⠲ g⠢ h⠖

Opciones:

| Opción | Para qué |
| --- | --- |
| `--json` | Salida completa en JSON: FEN, audio, braille, resaltados, flechas y dudas |
| `--foto` | Foto hecha con el móvil: corrige la perspectiva |
| `--modelo ruta.pt` | Usar otro modelo |

En `Brajedrez/tests/fixtures/img/` hay diagramas de ejemplo para probar.

### Tests

    python -m pytest -q

41 pruebas: formatos de audio y braille idénticos a los de ONCE, detección de resaltados y flechas, orientación y la web.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `Brajedrez/src/` | Código: visión, clasificador, formatos de audio y braille, web y línea de comandos |
| `Brajedrez/modelos/` | Modelo final (`casillas_v2.pt`), modelo base, modelo de medida y sus métricas |
| `Brajedrez/tests/` | Pruebas y diagramas de ejemplo de ONCE |
| `Brajedrez/docs/` | Detalle técnico, evidencias de resultados y material del reto |

## Reentrenar el modelo

No hace falta para usar Brajedrez. Para reentrenarlo se necesita el dataset *chess-positions* de Kaggle en `Brajedrez/datos/train` (no está en el repositorio). Pasos y comandos en [Brajedrez/docs/LEEME_tecnico.md](Brajedrez/docs/LEEME_tecnico.md). Con una gráfica NVIDIA, instala PyTorch con CUDA y el entrenamiento la usa sola:

    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130

## Problemas frecuentes

- **«No se puede cargar el archivo Activate.ps1»**: ejecuta antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` en esa misma ventana.
- **«Falta el modelo entrenado»**: ejecuta los comandos desde `Brajedrez/`, donde está la carpeta `modelos/`.
- **El puerto 8000 está ocupado**: usa `--puerto 8080`.

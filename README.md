# Detección de code-switching español–quechua

Proyecto de Ciencia de Datos y Machine Learning (UMSS, II/2026), práctica del segundo parcial. Dado
un audio, decide **tramo por tramo** (tramas de 10 ms) si se habla español o quechua y marca dónde
cambia el idioma.

Incluye:

- **Modelo profundo (supervisado):** CNN + BiLSTM sobre log-mel, entrenado con PyTorch.
- **Modelo no supervisado:** agrupamiento (K-means y GMM, k = 2) de ventanas de 500 ms descritas con
  MFCC.
- **Línea base:** predecir siempre español.
- Un corpus **sintético** de code-switching, armado pegando fragmentos monolingües de Common Voice
  (no existe un corpus público de code-switching español–quechua con etiquetas por trama).
- Una aplicación web para entrenar, ver los resultados y analizar audios propios.

## Resultado

Partición de prueba (450 frases sintéticas con hablantes que no están en el entrenamiento), cifras de
`resultados/metricas.json`. Los cambios se evalúan con tolerancia de ±200 ms.

| Modelo | Exactitud | F1 macro | F1 de cambios | Falsos cambios en control |
|---|---:|---:|---:|---:|
| Profundo (CNN + BiLSTM) | 86.5 % | 86.5 % | 58.7 % | 44.4 % |
| No supervisado (K-means) | 53.1 % | 52.3 % | 17.5 % | 90.0 % |
| Línea base (siempre español) | 49.4 % | 33.0 % | 0.0 % | 0.0 % |

El no supervisado queda prácticamente al azar (ARI con el idioma 0.0035 en las ventanas de prueba). Los
resultados describen el corpus sintético, **no** el habla real de Cochabamba: el quechua es el de
Puno (variedad Collao), cada unión entre idiomas es también un cambio de hablante, y no hubo
grabaciones reales en esta corrida.

El informe final está en `docs/INFORME FINAL - DETECCION DE CODE-SWITCHING.docx`. La decisión de usar
un corpus sintético y sus consecuencias están en `docs/adr/0001-datos-sinteticos-y-fuentes.md`, y
todas las cifras medidas, en `docs/resultados.md`.

## Requisitos

- **Python 3.12** en un entorno virtual `.venv`, creado con [uv](https://docs.astral.sh/uv/)
  (`pip install uv`). PyTorch se instala desde el índice de CPU: no hace falta GPU.
- **Node.js / npm** es opcional: solo sirve para los atajos `npm run dev` y `npm run setup`. Sin Node
  todo funciona con el comando de Python equivalente.
- Unos 15 GB de memoria y espacio en disco para los clips y el corpus sintético (la síntesis ocupa
  unos 2 GB).

## Datos

Los audios de origen **no vienen en el repositorio** (pesan gigas). Se descargan de Mozilla Data
Collective (Common Voice, licencia CC0) y se dejan **sin descomprimir** en `datos/crudo/`:

| Qué descargar | Archivo en `datos/crudo/` |
|---|---|
| Quechua de Puno, `qxp`, Common Voice 27.0 (2026-09-11) | `1789492511207-cv-corpus-27.0-2026-09-11-qxp.tar.gz` |
| Español de México, segmento femenino (`es-MX_female`) | `1783027036623-es-MX_female.tar.gz` |
| Español de México, segmento masculino (`es-MX_male`) | `1783028556514-es-MX_male.tar.gz` |

Los números del prefijo de cada archivo los pone la plataforma; `src/preparar.py` lee los `.tar.gz`
de esa carpeta y no necesita los nombres exactos. Los archivos de español no traen
`validated.tsv`: el script usa la unión de `train.tsv`, `dev.tsv` y `test.tsv`.

`datos/clips/` (clips ya preparados) y `datos/sintetico/` (frases sintéticas) se generan solos y no se
versionan.

## Cómo correr

### 1. Crear el entorno (una sola vez)

```bash
npm run setup
```

Sin Node, es lo mismo que:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy scipy librosa soundfile scikit-learn joblib matplotlib pypandoc_binary pytest
```

### 2. Pipeline completo (scripts independientes, en este orden)

```bash
.venv/bin/python src/preparar.py             # clips de Common Voice: mono 16 kHz, división por hablante
.venv/bin/python src/sintetizar.py           # corpus sintético de code-switching
.venv/bin/python src/modelo_profundo.py      # entrena CNN + BiLSTM
.venv/bin/python src/modelo_no_supervisado.py
.venv/bin/python src/evaluar.py              # métricas y figuras en resultados/
.venv/bin/python src/informe.py              # genera el .docx del informe
```

Tiempos medidos en la corrida de referencia (Intel Core Ultra 7 265U, solo CPU), de
`resultados/tiempos.json`:

| Paso | Tiempo |
|---|---:|
| `preparar.py` | 66 s |
| `sintetizar.py` | 177 s |
| `modelo_profundo.py` | **8283 s (2.3 h)** |
| `modelo_no_supervisado.py` | 82 s |
| `evaluar.py` | 63 s |

El entrenamiento profundo es el paso lento: cerca de 2.3 h en CPU (21 épocas con parada temprana, de
unos 5 a 9 min cada una). Todo usa la semilla 2026, así que repetir la corrida debería dar métricas muy
parecidas, aunque no necesariamente idénticas: las versiones de los paquetes no están fijadas y los
cálculos en CPU pueden variar con el número de hilos. Los tiempos sí cambian.

> El repositorio ya trae los modelos y los resultados de la corrida de referencia en `modelos/` y
> `resultados/`, así que la aplicación funciona sin entrenar.

## Uso de la aplicación

```bash
npm run dev
```

Sin Node: `.venv/bin/python src/servidor.py`. Se abre el navegador en `http://127.0.0.1:8000/` (si el
puerto está ocupado usa el siguiente libre, hasta el 8010; la terminal muestra la dirección). Para
detenerla: `Ctrl+C`. Solo escucha en `127.0.0.1` y solo ejecuta los pasos de su lista cerrada. Con
`--sin-navegador` no abre el navegador.

| Vista | Qué hace |
|---|---|
| **Datos** | Clips por idioma y partición, y las frases sintéticas de prueba. |
| **Entrenar** | Corre los pasos del pipeline en orden o uno solo, con su estado, su tiempo y la salida de la consola en vivo. |
| **Resultados** | Comparación de los tres modelos, métricas por tipo de frase y por duración, agrupamiento y gráficos. |
| **Probar** | Elige una frase de prueba, sube un audio (WAV, MP3, M4A, OGG o FLAC, entre 1 s y 5 min) o graba con el micrófono. Cada modelo dibuja su línea de tiempo con español, quechua, silencios y cambios. |

El navegador decodifica cualquier audio y lo envía como WAV mono de 16 kHz. Un audio vacío, de menos
de 1 s, de más de 5 min, en silencio o ilegible devuelve un mensaje claro en español. La detección de
habla es relativa a la trama más fuerte, así que un audio con solo ruido no se rechaza: se analiza
igual y sus tramos no significan nada.

> La grabación con micrófono desde la aplicación no se verificó con un micrófono real; la subida de
> archivos y las frases de prueba sí.

## Agregar grabaciones reales

Con grabaciones reales de habla mezclada, `src/evaluar.py` mide los modelos en habla de verdad. En
`datos/reales/` pon pares con el mismo nombre:

- `frase1.wav`: el audio (cualquier duración válida).
- `frase1.txt`: las etiquetas de Audacity (menú *Archivo > Exportar > Exportar etiquetas*), una por
  línea con tres columnas separadas por tabulador: inicio en segundos, fin en segundos y el idioma.

```
0.00	2.35	es
2.35	3.80	qu
3.80	6.10	es
```

El idioma es `es`, `español` o `espanol` para español, y `qu` o `quechua` para quechua (sin importar
mayúsculas). Después corre de nuevo `evaluar.py` (o el paso **Evaluar** de la aplicación): la
sección «Grabaciones reales» de los resultados pasa de vacía a mostrar las métricas. Los archivos con
formato inválido se omiten y se listan con el motivo.

## Estructura

```
datos/          crudo/ (tar de Common Voice), clips/, sintetico/ y reales/ (opcional)
src/            preparar, sintetizar, modelo_profundo, modelo_no_supervisado, evaluar, informe,
                servidor.py + app.html (la aplicación) y módulos de apoyo
tests/          pruebas de cada módulo
modelos/        pesos entrenados
resultados/     métricas, tiempos y figuras (incluye las capturas de la aplicación)
docs/           especificación, ADR, resultados.md e informe (contenido.md y el .docx)
```

## Verificar

```bash
npm test
```

(equivale a `.venv/bin/python -m pytest -q`).

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `.venv/bin/python: No such file or directory` | Falta el entorno: `npm run setup` (o los comandos de uv del paso 1). |
| `npm: command not found` | Usar el comando de Python equivalente; Node no es necesario. |
| `No module named 'torch'` (o `librosa`, `sklearn`) | Se está usando el Python del sistema; correr con `.venv/bin/python`. |
| `preparar.py` dice `Falta el audio de ... en datos/crudo/` | Dejar los tres `.tar.gz` sin descomprimir en `datos/crudo/` (sección **Datos**). |
| Un script o la aplicación dice que los modelos no están entrenados | Correr el paso que nombra el mensaje (por ejemplo `modelo_profundo.py`), o **Entrenar todo** en la aplicación. |
| `Primero corre la evaluación (src/evaluar.py).` | `informe.py` necesita `resultados/metricas.json`; correr `evaluar.py` antes. |
| `Los puertos 8000-8010 están ocupados.` | Cerrar otras instancias de la aplicación (`Ctrl+C`) y volver a correrla. |
| La grabación con micrófono no funciona | El navegador debe tener permiso de micrófono; es la parte sin verificar. Subir un archivo de audio es la alternativa. |
| El entrenamiento profundo tarda horas | Es lo esperado en CPU (unas 2.3 h). Dejarlo terminar: el modelo solo se guarda al final. |

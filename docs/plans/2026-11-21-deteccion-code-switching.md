# Detección de code-switching español–quechua — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototipo completo (corpus sintético, modelo profundo CNN+BiLSTM, modelo no supervisado K-means/GMM,
evaluación, aplicación web local e informe `.docx`) que identifica español o quechua tramo por tramo y marca los
puntos de cambio.

**Architecture:** Pipeline de scripts independientes en `segundo parcial/src/` (preparar → sintetizar → entrenar
profundo → entrenar no supervisado → evaluar → informe) que se comunican por archivos en `datos/`, `modelos/` y
`resultados/`. Un servidor HTTP de la biblioteca estándar (mismo esquema que `src/servidor.py` del primer parcial)
corre esos scripts y expone la inferencia a una página HTML.

**Tech Stack:** Python 3.12, numpy, scipy, librosa, soundfile, PyTorch CPU, scikit-learn, joblib, matplotlib,
pypandoc_binary (trae pandoc embebido; pandoc no está instalado en el sistema), pytest. HTML/JS sin framework.

**Spec:** `segundo parcial/docs/specs/2026-11-21-deteccion-code-switching-design.md`

## Global Constraints

- Todas las rutas de este plan son relativas a `segundo parcial/` salvo que digan "raíz" (el repo `Machine learning/`). Los comandos `npm`, `.venv/bin/python` y `pytest` se corren desde `segundo parcial/`; los `git` desde la raíz.
- Python 3.12 en `segundo parcial/.venv`, creado con `uv`. Torch desde el índice CPU: `--index-url https://download.pytorch.org/whl/cpu`.
- Antes de instalar, verificar que cada paquete existe en PyPI con ese nombre exacto (`uv pip install --dry-run`); no inventar paquetes.
- Idioma de código, comentarios, mensajes de interfaz e informe: español. Commits en inglés, sin trailers de Claude.
- Audio interno: mono, 16 000 Hz, float32 en [-1, 1].
- Tramas de 10 ms (salto 160 muestras), ventana de 25 ms (400 muestras), `center=True`; número de tramas `1 + n_muestras // 160`.
- Log-mel de 64 bandas, normalizado por frase (media 0, desviación 1 por banda).
- Etiquetas por trama: `ES = 0`, `QU = 1`, `IGNORAR = -100` (no-habla).
- Fragmentos de 0,5 a 3 s; frases de 4 a 12 s; mixtas con 2 a 5 tramos de idioma; proporciones 60 % mixta / 20 % monolingüe / 20 % control.
- Fundido cruzado de 20 a 50 ms en cada unión; pausa aleatoria de 0 a 150 ms; el mismo tratamiento en los tres tipos de frase.
- División por hablante 80/10/10; ningún hablante en dos particiones; frases sintéticas solo con clips de una misma partición.
- Post-proceso: mediana de 9 tramas de salida y duración mínima de tramo de 200 ms (20 tramas). Tolerancia de puntos de cambio ±200 ms.
- No supervisado: ventanas de 500 ms (50 tramas) con salto de 250 ms (25 tramas); 20 MFCC + deltas, media y desviación → 80 dimensiones; k = 2.
- Semilla global `SEMILLA = 2026`.
- Entrenamiento del modelo profundo: menos de una hora en CPU (14 núcleos, 15 GB RAM).
- Servidor solo en `127.0.0.1`; solo ejecuta los pasos de su lista cerrada `PASOS`.
- Las cifras del informe salen de `resultados/`, nunca se escriben a mano.

## Review Focus

1. Audio subido en cualquier formato (mp3, m4a, estéreo, 44,1 kHz) o grabado desde el micrófono → se analiza igual: el navegador lo decodifica y lo envía como WAV mono 16 kHz (Task 9).
2. Audio inválido, vacío, de menos de 1 s, de más de 300 s o sin habla → mensaje claro en español con HTTP 400, nunca un traceback (Tasks 7 y 9).
3. Modelos sin entrenar o datos crudos ausentes → la app y los scripts lo dicen y nombran el paso a correr (Tasks 2, 5, 6 y 9).
4. Frases sin ningún cambio de idioma (monolingüe, control) → las métricas de puntos de cambio no dividen por cero y siguen la convención fija de Task 4.
5. Petición a `/audio/...` o `/resultados/...` con `..` en la ruta → 404, nunca un archivo fuera de la carpeta servida (Task 9).

---

### Task 1: Proyecto, configuración y utilidades de audio

**Files:**
- Create: `package.json`, `.gitignore`, `src/config.py`, `src/audio.py`, `tests/conftest.py`, `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `src/config.py`: `SR = 16000`, `HOP = 160`, `VENTANA = 400`, `N_MELS = 64`, `ES = 0`, `QU = 1`, `IGNORAR = -100`, `IDIOMAS = {'es': ES, 'qu': QU}`, `NOMBRES = {ES: 'Español', QU: 'Quechua'}`, `SEMILLA = 2026`, rutas `RAIZ`, `DATOS`, `CRUDO`, `CLIPS`, `SINTETICO`, `REALES`, `MODELOS`, `RESULTADOS`, `FIGURAS` (todas `pathlib.Path`, `RAIZ` = carpeta `segundo parcial/`).
  - `src/audio.py`:
    - `class ErrorAudio(ValueError)`
    - `n_tramas(n_muestras: int) -> int`
    - `cargar_audio(ruta: str | Path) -> np.ndarray` — mono, 16 kHz, float32; `ErrorAudio("No se pudo leer el audio: <nombre>")` si falla la lectura o queda vacío.
    - `normalizar_volumen(y: np.ndarray, objetivo_dbfs: float = -20.0) -> np.ndarray` — por RMS, recortado a [-1, 1]; silencio total se devuelve igual.
    - `recortar_silencios(y: np.ndarray, top_db: float = 30.0) -> np.ndarray`
    - `mascara_habla(y: np.ndarray, umbral_db: float = -35.0) -> np.ndarray` — bool, longitud `n_tramas(len(y))`; habla si el RMS de la trama supera `umbral_db` respecto al RMS máximo del audio. Audio todo cero → todo `False`.
    - `pausas(y: np.ndarray) -> np.ndarray` — índices de muestra (int) del centro de cada racha de tramas de no-habla interior (no las de los extremos).
- `tests/conftest.py` agrega `src/` a `sys.path` y define fixtures `tono(dur_s, f=220.0)` y `silencio(dur_s)` que devuelven arrays float32 a 16 kHz.

- [ ] **Step 1: Crear `package.json` y `.gitignore`**

`package.json` con `"dev": ".venv/bin/python src/servidor.py"`, `"test": ".venv/bin/python -m pytest -q"` y `"setup"` que crea `.venv` con `uv venv --python 3.12 .venv` e instala con `uv pip install --python .venv/bin/python` primero `torch --index-url https://download.pytorch.org/whl/cpu` y luego `numpy scipy librosa soundfile scikit-learn joblib matplotlib pypandoc_binary pytest`. `.gitignore`: `.venv/`, `datos/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`.

- [ ] **Step 2: Instalar el entorno**

Run: `npm run setup && .venv/bin/python -c "import torch, librosa, sklearn, soundfile, pypandoc; print(torch.__version__, librosa.__version__)"`
Expected: imprime dos versiones, sin errores.

- [ ] **Step 3: Escribir los tests que fallan**

```python
# tests/test_audio.py
import numpy as np, soundfile as sf, pytest
from audio import n_tramas, cargar_audio, normalizar_volumen, mascara_habla, pausas, ErrorAudio
from config import SR

def test_n_tramas():
    assert n_tramas(16000) == 101
    assert n_tramas(0) == 1

def test_cargar_audio_estereo_44k(tmp_path, tono):
    y = np.stack([tono(1.0), tono(1.0)], axis=1)
    sf.write(tmp_path / 'a.wav', y, 44100)
    x = cargar_audio(tmp_path / 'a.wav')
    assert x.ndim == 1 and x.dtype == np.float32
    assert abs(len(x) - SR) < 200  # ~1 s remuestreado a 16 kHz

def test_cargar_audio_invalido(tmp_path):
    (tmp_path / 'malo.wav').write_bytes(b'no es audio')
    with pytest.raises(ErrorAudio):
        cargar_audio(tmp_path / 'malo.wav')

def test_normalizar_volumen(tono):
    y = normalizar_volumen(0.01 * tono(1.0))
    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)))
    assert abs(rms_db - (-20.0)) < 0.5
    assert np.all(normalizar_volumen(np.zeros(1600, np.float32)) == 0)

def test_mascara_y_pausas(tono, silencio):
    y = np.concatenate([tono(1.0), silencio(0.5), tono(1.0)])
    m = mascara_habla(y)
    assert len(m) == n_tramas(len(y))
    assert m[50] and not m[125] and m[200]
    p = pausas(y)
    assert len(p) == 1 and abs(p[0] - int(1.25 * SR)) < 0.05 * SR
    assert not mascara_habla(np.zeros(SR, np.float32)).any()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_audio.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'audio'`

- [ ] **Step 5: Implementar `src/config.py` y `src/audio.py`**

`cargar_audio` con `librosa.load(ruta, sr=SR, mono=True)` envuelto para convertir cualquier excepción de lectura en `ErrorAudio`. `mascara_habla` con `librosa.feature.rms(frame_length=VENTANA, hop_length=HOP, center=True)`.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_audio.py -q`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add "segundo parcial/package.json" "segundo parcial/.gitignore" "segundo parcial/src/config.py" "segundo parcial/src/audio.py" "segundo parcial/tests"
git commit -m "feat: add project setup and audio utilities for code-switching"
```

---

### Task 2: Preparación de clips de Common Voice

**Files:**
- Create: `src/preparar.py`, `tests/test_preparar.py`

**Interfaces:**
- Consumes: `cargar_audio`, `normalizar_volumen`, `recortar_silencios`, `ErrorAudio` (Task 1); `CRUDO`, `CLIPS`, `SR`, `SEMILLA`.
- Produces:
  - `locale_de_tar(ruta_tar: Path) -> str` — locale leído de la estructura interna `cv-corpus-*/<locale>/...` (p. ej. `'qxp'`, `'es'`).
  - `IDIOMA_DE_LOCALE = {'qxp': 'qu', 'es': 'es'}` — cualquier otro locale es un error.
  - `leer_validados(ruta_tar: Path) -> list[dict]` — filas de `validated.tsv` con claves `hablante` (`client_id`), `archivo` (`path`), `genero` (`'M'` si `gender` empieza con `male`, `'F'` si con `female`, `''` en otro caso).
  - `dividir_hablantes(hablantes: list[str], semilla: int = SEMILLA) -> dict[str, str]` — valores `'entrenamiento'`, `'validacion'`, `'prueba'` en proporción 80/10/10; con 3 o más hablantes cada partición recibe al menos uno.
  - `seleccionar(filas: list[dict], max_por_hablante: int, horas_max: float, duracion_media_s: float, semilla: int = SEMILLA) -> list[dict]` — tope por hablante, luego muestreo hasta el tope de horas estimado.
  - `preparar(rutas_tar: list[Path], horas_max: float = 12.0, max_por_hablante: int = 200) -> list[dict]` — extrae solo los clips seleccionados, los decodifica con un `multiprocessing.Pool`, normaliza volumen, recorta silencios, descarta los de menos de 1 s y escribe `datos/clips/<idioma>/<particion>/<id>.wav`. Devuelve filas `{ruta, idioma, hablante, genero, particion, duracion}`.
  - Script: `python src/preparar.py` busca `*.tar.gz` y `*.tar` en `datos/crudo/`, agrupa por idioma, llama a `preparar` y escribe `datos/clips/indice.csv` y `resultados/datos.json` (clips, horas y hablantes por idioma y partición).
  - Si falta el idioma `qu` o `es` en `datos/crudo/`, termina con código 1 y el mensaje `Falta el audio de <Quechua|Español> en datos/crudo/. Archivos encontrados: <lista>. Ver README, sección Datos.`

- [ ] **Step 1: Escribir los tests que fallan**

Fixture `tar_falso(tmp_path, locale, hablantes)` que arma un `.tar.gz` con `cv-corpus-26.0/<locale>/validated.tsv` (columnas `client_id path sentence gender`) y clips `.wav` de tono de 1,5 s en `clips/`.

```python
def test_locale_y_validados(tar_falso):
    t = tar_falso('qxp', {'h1': 3, 'h2': 2})
    assert locale_de_tar(t) == 'qxp'
    filas = leer_validados(t)
    assert len(filas) == 5 and {f['hablante'] for f in filas} == {'h1', 'h2'}

def test_dividir_hablantes_disjunto():
    p = dividir_hablantes([f'h{i}' for i in range(50)])
    assert sorted(set(p.values())) == ['entrenamiento', 'prueba', 'validacion']
    assert list(p.values()).count('entrenamiento') == 40

def test_seleccionar_tope_por_hablante():
    filas = [{'hablante': 'a', 'archivo': f'{i}.mp3', 'genero': ''} for i in range(500)]
    assert len(seleccionar(filas, max_por_hablante=200, horas_max=100, duracion_media_s=4)) == 200

def test_preparar_escribe_clips(tar_falso, tmp_path, monkeypatch):
    monkeypatch.setattr('preparar.CLIPS', tmp_path / 'clips')
    t = tar_falso('es', {f'h{i}': 2 for i in range(10)})
    filas = preparar([t])
    assert filas and all(f['idioma'] == 'es' for f in filas)
    por_hablante = {}
    for f in filas:
        por_hablante.setdefault(f['hablante'], set()).add(f['particion'])
    assert all(len(v) == 1 for v in por_hablante.values())

def test_falta_idioma(tmp_path, monkeypatch, tar_falso, capsys):
    monkeypatch.setattr('preparar.CRUDO', tar_falso('es', {'h1': 1}).parent)
    with pytest.raises(SystemExit) as e:
        preparar_main()
    assert e.value.code == 1 and 'Falta el audio de Quechua' in capsys.readouterr().err
```

(`preparar_main` es la función `main()` del script, importada como `from preparar import main as preparar_main`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preparar.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'preparar'`

- [ ] **Step 3: Implementar `src/preparar.py` con las firmas de Interfaces**

Leer el tar con `tarfile` sin extraerlo entero: primero `validated.tsv`, después solo los miembros seleccionados, a una carpeta temporal. `<id>` del wav = primeros 8 caracteres del hablante + índice.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_preparar.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/preparar.py" "segundo parcial/tests/test_preparar.py"
git commit -m "feat: extract Common Voice clips with speaker-disjoint splits"
```

---

### Task 3: Características y síntesis del corpus

**Files:**
- Create: `src/caracteristicas.py`, `src/sintetizar.py`, `tests/test_caracteristicas.py`, `tests/test_sintetizar.py`

**Interfaces:**
- Consumes: Task 1 completo; `datos/clips/indice.csv` (Task 2).
- Produces:
  - `src/caracteristicas.py`:
    - `logmel(y: np.ndarray) -> np.ndarray` — float32 `(n_tramas(len(y)), 64)`, `log(mel + 1e-6)` normalizado por banda.
    - `mfcc_ventanas(y: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(X (N, 80) float32, centros (N,) int en tramas)`. 20 MFCC + deltas; resta de la media cepstral sobre las tramas de habla de toda la frase; por ventana de 50 tramas con salto 25, media y desviación sobre sus tramas de habla; se omiten ventanas con menos de 50 % de habla. Sin ventanas válidas → `(np.zeros((0, 80)), np.zeros(0, int))`.
  - `src/sintetizar.py`:
    - `fragmentar(y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]` — trozos de 0,5 a 3 s cortados solo en `pausas(y)`; el trozo sin pausa válida dentro del rango se descarta.
    - `Banco` = `dict[str, dict[str, list[tuple[np.ndarray, str]]]]` (idioma → hablante → lista de `(fragmento, genero)`).
    - `construir_banco(filas: list[dict], rng) -> Banco` — filas de `indice.csv` de una sola partición.
    - `componer(tipo: str, banco: Banco, rng) -> dict` con `tipo ∈ {'mixta', 'monolingue', 'control'}`. Devuelve `{'audio': np.ndarray, 'etiquetas': np.ndarray int16 (n_tramas,), 'meta': {'tipo', 'duracion', 'fragmentos': [{'idioma', 'hablante', 'genero', 'inicio', 'fin'}]}}`.
      - Un "tramo" son 1 o 2 fragmentos consecutivos del mismo idioma y hablante.
      - mixta: 2 a 5 tramos que alternan idioma, cada tramo de un hablante al azar.
      - monolingüe: un idioma, un hablante.
      - control: un idioma, 2 o más tramos, hablantes distintos en tramos consecutivos.
      - Si la duración cae fuera de 4–12 s, se vuelve a sortear (muestreo por rechazo).
      - Uniones: pausa de 0 a 150 ms y fundido cruzado de 20 a 50 ms. Las tramas reciben el idioma de su fragmento; las de no-habla (`mascara_habla` sobre el audio final) reciben `IGNORAR`.
      - Aumento sobre la frase completa: ruido gaussiano a SNR aleatoria de 20 a 40 dB y ganancia aleatoria de ±6 dB.
    - `TIPOS = {'mixta': 0.6, 'monolingue': 0.2, 'control': 0.2}`
    - `generar_particion(particion: str, n_frases: int, semilla: int) -> None` — escribe `datos/sintetico/<particion>/<id>.wav`, `<id>.npz` (claves `logmel`, `etiquetas`) y una línea por frase en `datos/sintetico/<particion>/meta.jsonl` (`{'id', 'tipo', 'duracion', 'fragmentos'}`).
    - Script: `python src/sintetizar.py [--horas-entrenamiento 9 --horas-validacion 1 --horas-prueba 1]`, con una semilla distinta por partición derivada de `SEMILLA`. Si falta `datos/clips/indice.csv`, termina con código 1 y el mensaje `Primero corre la preparación de datos (src/preparar.py).`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_caracteristicas.py
def test_logmel_forma(tono):
    y = tono(2.0)
    x = logmel(y)
    assert x.shape == (n_tramas(len(y)), 64) and x.dtype == np.float32

def test_mfcc_ventanas(tono, silencio):
    y = np.concatenate([tono(2.0), silencio(1.0)])
    X, c = mfcc_ventanas(y, mascara_habla(y))
    assert X.shape[1] == 80 and len(X) == len(c) and c.max() < 200 + 25
    X0, c0 = mfcc_ventanas(silencio(1.0), mascara_habla(silencio(1.0)))
    assert X0.shape == (0, 80)
```

```python
# tests/test_sintetizar.py — fixture `banco_falso`: 2 idiomas × 4 hablantes × 6 fragmentos de tono de 1–2 s
#                                                  (frecuencia distinta por hablante)
def test_fragmentar_duraciones(tono, silencio):
    y = np.concatenate([tono(1.2), silencio(0.3), tono(0.8), silencio(0.3), tono(4.0)])
    for f in fragmentar(y, np.random.default_rng(0)):
        assert 0.5 * SR <= len(f) <= 3.0 * SR

@pytest.mark.parametrize('tipo', ['mixta', 'monolingue', 'control'])
def test_componer_reglas(tipo, banco_falso):
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = componer(tipo, banco_falso, rng)
        m = f['meta']
        assert 4.0 <= m['duracion'] <= 12.0
        assert len(f['etiquetas']) == n_tramas(len(f['audio']))
        idiomas = [fr['idioma'] for fr in m['fragmentos']]
        cambios = sum(a != b for a, b in zip(idiomas, idiomas[1:]))
        if tipo == 'mixta':
            assert 1 <= cambios <= 4
        else:
            assert cambios == 0
        if tipo == 'control':
            assert len({fr['hablante'] for fr in m['fragmentos']}) >= 2
        if tipo == 'monolingue':
            assert len({fr['hablante'] for fr in m['fragmentos']}) == 1
        assert set(np.unique(f['etiquetas'])) <= {ES, QU, IGNORAR}

def test_componer_determinista(banco_falso):
    a = componer('mixta', banco_falso, np.random.default_rng(7))
    b = componer('mixta', banco_falso, np.random.default_rng(7))
    assert np.array_equal(a['audio'], b['audio'])

def test_generar_particion_proporciones(tmp_path, monkeypatch, indice_falso):
    monkeypatch.setattr('sintetizar.SINTETICO', tmp_path)
    generar_particion('entrenamiento', 100, semilla=3)
    tipos = [json.loads(l)['tipo'] for l in (tmp_path / 'entrenamiento' / 'meta.jsonl').open()]
    assert len(tipos) == 100 and 50 <= tipos.count('mixta') <= 70
    assert len(list((tmp_path / 'entrenamiento').glob('*.npz'))) == 100
```

(`indice_falso` escribe clips de tono y un `indice.csv` en `tmp_path` y parchea `sintetizar.CLIPS`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_caracteristicas.py tests/test_sintetizar.py -q`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/caracteristicas.py` y `src/sintetizar.py` con las firmas de Interfaces**

`logmel` con `librosa.feature.melspectrogram(sr=SR, n_fft=512, win_length=VENTANA, hop_length=HOP, n_mels=N_MELS, center=True)`; `mfcc_ventanas` con `librosa.feature.mfcc(n_mfcc=20, ...)` y `librosa.feature.delta`, con los mismos parámetros de trama.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_caracteristicas.py tests/test_sintetizar.py -q`
Expected: todos pasan (`7 passed`)

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/caracteristicas.py" "segundo parcial/src/sintetizar.py" "segundo parcial/tests/test_caracteristicas.py" "segundo parcial/tests/test_sintetizar.py"
git commit -m "feat: synthesize code-switched corpus with speaker-change controls"
```

---

### Task 4: Post-proceso y métricas

**Files:**
- Create: `src/postproceso.py`, `src/metricas.py`, `tests/test_postproceso.py`, `tests/test_metricas.py`

**Interfaces:**
- Consumes: `ES`, `QU`, `IGNORAR`, `HOP`, `SR`.
- Produces:
  - `src/postproceso.py`:
    - `suavizar(etiquetas: np.ndarray, mediana: int = 37, min_tramas: int = 20) -> np.ndarray` — mediana solo sobre tramas de habla (las `IGNORAR` quedan `IGNORAR`). Después, los tramos de menos de `min_tramas` tramas de habla toman el idioma del tramo vecino más largo. 37 tramas de 10 ms equivalen a las 9 salidas de 40 ms de las restricciones globales.
    - `tramos(etiquetas: np.ndarray) -> list[tuple[float, float, int]]` — `(inicio_s, fin_s, idioma)`. Tramas de habla consecutivas con el mismo idioma, aunque las separen tramas `IGNORAR`, forman un solo tramo.
    - `puntos_cambio(tramos: list[tuple[float, float, int]]) -> list[float]` — en cada par de tramos consecutivos con idioma distinto, el punto medio entre el fin de uno y el inicio del otro.
  - `src/metricas.py`:
    - `metricas_trama(ref: np.ndarray, pred: np.ndarray) -> dict` — `{'exactitud', 'f1_macro', 'f1_es', 'f1_qu', 'matriz': [[int, int], [int, int]], 'n': int}`; ignora tramas con `ref == IGNORAR`; una `pred == IGNORAR` sobre habla cuenta como error.
    - `contar_cambios(ref: list[float], pred: list[float], tolerancia: float = 0.2) -> tuple[int, int, int]` — `(vp, fp, fn)` con emparejamiento uno a uno por cercanía.
    - `prf(vp: int, fp: int, fn: int) -> dict` — `{'precision', 'exhaustividad', 'f1'}`. Convención fija: sin referencia ni predicción → `1.0, 1.0, 1.0`; `vp + fp == 0` → precisión `1.0`; `vp + fn == 0` → exhaustividad `1.0`; F1 `0.0` si precisión + exhaustividad = 0.
    - `tasa_falsos_cambios(predicciones: list[list[float]]) -> float` — fracción de frases (de control) con al menos un cambio predicho; lista vacía → `0.0`.
    - `mapear_grupos(grupos: np.ndarray, ref: np.ndarray) -> dict[int, int]` — grupo → idioma, con `scipy.optimize.linear_sum_assignment` sobre la matriz de coincidencias, ignorando `IGNORAR`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_postproceso.py
def test_suavizar_quita_tramo_corto():
    e = np.array([ES] * 100 + [QU] * 5 + [ES] * 100)
    assert (suavizar(e) == ES).all()

def test_suavizar_respeta_no_habla():
    e = np.array([ES] * 50 + [IGNORAR] * 30 + [QU] * 50)
    s = suavizar(e)
    assert (s[50:80] == IGNORAR).all() and s[0] == ES and s[-1] == QU

def test_tramos_y_cambios():
    e = np.array([ES] * 100 + [IGNORAR] * 20 + [QU] * 100)
    t = tramos(e)
    assert [x[2] for x in t] == [ES, QU]
    assert abs(t[0][0] - 0.0) < 1e-9 and abs(t[0][1] - 1.0) < 1e-9
    assert puntos_cambio(t) == pytest.approx([1.1])

def test_tramos_unen_por_silencio():
    e = np.array([ES] * 50 + [IGNORAR] * 20 + [ES] * 50)
    assert len(tramos(e)) == 1 and puntos_cambio(tramos(e)) == []
```

```python
# tests/test_metricas.py
def test_metricas_trama_ignora():
    ref = np.array([ES, ES, QU, QU, IGNORAR])
    pred = np.array([ES, QU, QU, QU, ES])
    m = metricas_trama(ref, pred)
    assert m['n'] == 4 and m['exactitud'] == 0.75 and m['matriz'] == [[1, 1], [0, 2]]

def test_contar_cambios_tolerancia():
    assert contar_cambios([1.0, 3.0], [1.15, 2.5, 3.1]) == (2, 1, 0)

@pytest.mark.parametrize('vp,fp,fn,esperado', [
    (0, 0, 0, (1.0, 1.0, 1.0)),
    (0, 2, 0, (0.0, 1.0, 0.0)),
    (0, 0, 3, (1.0, 0.0, 0.0)),
    (1, 1, 1, (0.5, 0.5, 0.5)),
])
def test_prf_convencion(vp, fp, fn, esperado):
    r = prf(vp, fp, fn)
    assert (r['precision'], r['exhaustividad'], r['f1']) == pytest.approx(esperado)

def test_tasa_falsos_cambios():
    assert tasa_falsos_cambios([[], [2.0], [], [1.0, 3.0]]) == 0.5
    assert tasa_falsos_cambios([]) == 0.0

def test_mapear_grupos_invertido():
    assert mapear_grupos(np.array([1, 1, 0, 0]), np.array([ES, ES, QU, QU])) == {1: ES, 0: QU}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_postproceso.py tests/test_metricas.py -q`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/postproceso.py` y `src/metricas.py` con las firmas de Interfaces**

`f1` con `sklearn.metrics.f1_score(labels=[ES, QU])` sobre las tramas válidas.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_postproceso.py tests/test_metricas.py -q`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/postproceso.py" "segundo parcial/src/metricas.py" "segundo parcial/tests/test_postproceso.py" "segundo parcial/tests/test_metricas.py"
git commit -m "feat: add smoothing, segment extraction and switch-point metrics"
```

---

### Task 5: Modelo profundo CNN + BiLSTM

**Files:**
- Create: `src/modelo_profundo.py`, `tests/test_modelo_profundo.py`

**Interfaces:**
- Consumes: `logmel`, `mascara_habla`, `metricas_trama`, `SINTETICO`, `MODELOS`, `RESULTADOS`, `IGNORAR`, `SEMILLA`.
- Produces:
  - `class ErrorModelo(FileNotFoundError)` — también la usa Task 6 (se importa desde aquí).
  - `class CRNN(torch.nn.Module)`: `__init__(self, n_mels: int = 64, canales: tuple = (32, 64, 64), oculto: int = 128)`; `forward(x: Tensor (B, T, 64)) -> Tensor (B, T, 2)`. Tres bloques Conv2d 3×3 + BatchNorm + ReLU; los dos primeros reducen el tiempo a la mitad (reducción total ×4) y los tres reducen la frecuencia a la mitad; BiLSTM de 2 capas con `oculto` unidades por dirección; Linear a 2 clases. Antes de devolver, la salida se repite ×4 en el tiempo y se recorta o rellena a T.
  - `class FrasesDataset(torch.utils.data.Dataset)` sobre `datos/sintetico/<particion>/*.npz`; `__getitem__` → `(logmel Tensor (T, 64), etiquetas LongTensor (T,))`; con `aumentar=True` aplica SpecAugment en numpy (2 máscaras de tiempo de hasta 20 tramas y 2 de frecuencia de hasta 8 bandas).
  - `rellenar(lote) -> tuple[Tensor, Tensor]` — collate: ceros en el logmel e `IGNORAR` en las etiquetas.
  - `entrenar(dir_sintetico: Path = SINTETICO, epocas: int = 30, paciencia: int = 5, lote: int = 32, lr: float = 1e-3, semilla: int = SEMILLA) -> dict` — `CrossEntropyLoss(ignore_index=IGNORAR)`, Adam, parada temprana por F1 macro por trama en validación. Guarda `modelos/profundo.pt` (`{'estado': state_dict, 'config': {...}}`) y `resultados/entrenamiento_profundo.json` (`{'epocas': [{'epoca', 'perdida_entrenamiento', 'perdida_validacion', 'f1_validacion', 'segundos'}], 'mejor_epoca', 'parametros'}`). Llama a `torch.set_num_threads(os.cpu_count())`.
  - `cargar_profundo(ruta: Path = MODELOS / 'profundo.pt') -> CRNN` — modo eval; si no existe, `ErrorModelo('El modelo profundo no está entrenado. Corre el paso "Entrenar modelo profundo".')`.
  - `predecir_profundo(modelo: CRNN, y: np.ndarray) -> np.ndarray` — int `(n_tramas(len(y)),)`, argmax por trama, `IGNORAR` donde `mascara_habla` es `False`, sin suavizar.
  - Script: `python src/modelo_profundo.py [--epocas N]` imprime una línea por época (pérdidas, F1 de validación, segundos). Si falta `datos/sintetico/entrenamiento`, termina con código 1 y el mensaje `Primero genera el corpus sintético (src/sintetizar.py).`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_crnn_forma_salida():
    m = CRNN()
    for T in (37, 400, 1201):
        assert m(torch.zeros(2, T, 64)).shape == (2, T, 2)

def test_crnn_tamano():
    assert 1e5 < sum(p.numel() for p in CRNN().parameters()) < 2e6

def test_rellenar_ignora_relleno():
    x, y = rellenar([(torch.zeros(10, 64), torch.zeros(10, dtype=torch.long)),
                     (torch.zeros(7, 64), torch.ones(7, dtype=torch.long))])
    assert x.shape == (2, 10, 64) and (y[1, 7:] == IGNORAR).all()

def test_entrenar_aprende(tmp_path, corpus_falso, monkeypatch):
    # corpus_falso: 40 frases de entrenamiento y 10 de validación donde ES = tono grave y QU = tono agudo
    monkeypatch.setattr('modelo_profundo.MODELOS', tmp_path)
    monkeypatch.setattr('modelo_profundo.RESULTADOS', tmp_path)
    h = entrenar(corpus_falso, epocas=8, paciencia=8, lote=8)
    assert max(e['f1_validacion'] for e in h['epocas']) > 0.9
    assert (tmp_path / 'profundo.pt').exists()

def test_cargar_sin_entrenar(tmp_path):
    with pytest.raises(ErrorModelo, match='no está entrenado'):
        cargar_profundo(tmp_path / 'no.pt')

def test_predecir_ignora_silencio(tono, silencio):
    y = np.concatenate([tono(1.0), silencio(1.0)])
    p = predecir_profundo(CRNN().eval(), y)
    assert len(p) == n_tramas(len(y)) and (p[150:190] == IGNORAR).all()
```

(`corpus_falso` va en `tests/conftest.py` y lo reutiliza Task 6: escribe los `.npz` y `meta.jsonl` con el mismo formato de Task 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_modelo_profundo.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'modelo_profundo'`

- [ ] **Step 3: Implementar `src/modelo_profundo.py` con las firmas de Interfaces**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_modelo_profundo.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/modelo_profundo.py" "segundo parcial/tests/test_modelo_profundo.py" "segundo parcial/tests/conftest.py"
git commit -m "feat: add CNN-BiLSTM frame-level language classifier"
```

---

### Task 6: Modelo no supervisado

**Files:**
- Create: `src/modelo_no_supervisado.py`, `tests/test_modelo_no_supervisado.py`

**Interfaces:**
- Consumes: `mfcc_ventanas`, `mascara_habla`, `cargar_audio`, `mapear_grupos`, `metricas_trama`, `ErrorModelo` (de `modelo_profundo`), rutas de `config`.
- Produces:
  - `ventanas_de_particion(dir_particion: Path, max_ventanas: int | None, semilla: int) -> dict` — recorre los `.wav` y `meta.jsonl` de la partición. Devuelve `{'X': (N, 80), 'idioma': (N,)` (mayoría de las tramas de habla de la ventana)`, 'genero': (N,) str, 'hablante': (N,) str, 'frase': (N,) str}`.
  - `entrenar_no_supervisado(dir_sintetico: Path = SINTETICO, max_ventanas: int = 200_000, semilla: int = SEMILLA) -> dict` — `StandardScaler`, `PCA(n_components=0.95)`, `KMeans(2, n_init=10)` y `GaussianMixture(2, covariance_type='full')` ajustados **sin** `idioma` sobre entrenamiento. Barrido k = 2..6 con `silhouette_score` sobre una submuestra de 10 000. Mapeo grupo → idioma con `mapear_grupos` sobre validación; `mejor` = el algoritmo con mayor F1 macro en validación. Guarda `modelos/no_supervisado.joblib` (`{'escalador', 'pca', 'kmeans', 'gmm', 'mapa': {'kmeans': {...}, 'gmm': {...}}, 'mejor': 'kmeans' | 'gmm'}`) y `resultados/no_supervisado.json` (`{'silueta_por_k': {k: valor}, 'validacion': {algoritmo: metricas_trama}, 'ari_idioma': {algoritmo: v}, 'ari_genero': {algoritmo: v}, 'ari_hablante': {algoritmo: v}, 'mejor'}`). Los ARI contra género y hablante sirven para ver si los grupos siguieron al idioma o al hablante.
  - `cargar_no_supervisado(ruta: Path = MODELOS / 'no_supervisado.joblib') -> dict` — si no existe, `ErrorModelo('El modelo no supervisado no está entrenado. Corre el paso "Entrenar modelo no supervisado".')`.
  - `predecir_no_supervisado(modelo: dict, y: np.ndarray, algoritmo: str | None = None) -> np.ndarray` — int `(n_tramas,)`. Cada trama de habla toma el idioma (ya mapeado) de la ventana cuyo centro está más cerca; no-habla → `IGNORAR`; `algoritmo=None` usa `modelo['mejor']`.
  - `grupos_de_ventanas(modelo: dict, X: np.ndarray, algoritmo: str) -> np.ndarray` — grupos crudos, sin mapear (para ARI, NMI y pureza en Task 7).
  - Script: `python src/modelo_no_supervisado.py`, con el mismo mensaje de error que Task 5 si falta el corpus.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_entrenar_separa_tonos(tmp_path, corpus_falso, monkeypatch):
    monkeypatch.setattr('modelo_no_supervisado.MODELOS', tmp_path)
    monkeypatch.setattr('modelo_no_supervisado.RESULTADOS', tmp_path)
    r = entrenar_no_supervisado(corpus_falso, max_ventanas=5000)
    assert r['validacion'][r['mejor']]['exactitud'] > 0.9
    assert set(r['silueta_por_k']) == {2, 3, 4, 5, 6}

def test_entrenar_no_usa_etiquetas(corpus_falso, corpus_falso_invertido, tmp_path, monkeypatch):
    # mismo audio con los idiomas invertidos en las etiquetas → mismos grupos crudos
    monkeypatch.setattr('modelo_no_supervisado.MODELOS', tmp_path)
    monkeypatch.setattr('modelo_no_supervisado.RESULTADOS', tmp_path)
    entrenar_no_supervisado(corpus_falso, max_ventanas=5000)
    a = cargar_no_supervisado(tmp_path / 'no_supervisado.joblib')['kmeans'].cluster_centers_
    entrenar_no_supervisado(corpus_falso_invertido, max_ventanas=5000)
    b = cargar_no_supervisado(tmp_path / 'no_supervisado.joblib')['kmeans'].cluster_centers_
    assert np.allclose(a, b)

def test_predecir_longitud_e_ignorar(modelo_ns_entrenado, tono, silencio):
    y = np.concatenate([tono(2.0, f=200), silencio(1.0)])
    p = predecir_no_supervisado(modelo_ns_entrenado, y)
    assert len(p) == n_tramas(len(y)) and (p[220:290] == IGNORAR).all()
    assert set(np.unique(p[:180])) <= {ES, QU}

def test_cargar_sin_entrenar(tmp_path):
    with pytest.raises(ErrorModelo, match='no está entrenado'):
        cargar_no_supervisado(tmp_path / 'no.joblib')
```

Fixtures nuevos en `tests/conftest.py`: `corpus_falso_invertido` (copia de `corpus_falso` con `ES` y `QU` intercambiados en los `.npz` y en `meta.jsonl`) y `modelo_ns_entrenado` (alcance de sesión: entrena sobre `corpus_falso` en una carpeta temporal y devuelve `cargar_no_supervisado(...)`; lo reutiliza Task 7).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_modelo_no_supervisado.py -q`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/modelo_no_supervisado.py` con las firmas de Interfaces**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_modelo_no_supervisado.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/modelo_no_supervisado.py" "segundo parcial/tests/test_modelo_no_supervisado.py" "segundo parcial/tests/conftest.py"
git commit -m "feat: add unsupervised K-means/GMM language clustering"
```

---

### Task 7: Inferencia y evaluación

**Files:**
- Create: `src/inferencia.py`, `src/evaluar.py`, `tests/test_inferencia.py`, `tests/test_evaluar.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 5 y 6.
- Produces:
  - `src/inferencia.py`:
    - `class ErrorEntrada(ValueError)`
    - `DURACION_MIN_S = 1.0`, `DURACION_MAX_S = 300.0`
    - `validar_audio(y: np.ndarray) -> None` — lanza `ErrorEntrada` con `'El audio dura menos de 1 segundo.'`, `'El audio dura más de 5 minutos; recórtalo.'` o `'No se detectó habla en el audio.'`.
    - `analizar_senal(y: np.ndarray, profundo: CRNN, no_supervisado: dict) -> dict` — `{'duracion': float, 'modelos': {'profundo': {'tramos': [{'inicio', 'fin', 'idioma': 'es' | 'qu'}], 'cambios': [float]}, 'no_supervisado': {..., 'algoritmo': str}}}`. Aplica `suavizar` a ambas predicciones.
    - `analizar(ruta: Path) -> dict` — `cargar_audio` + `normalizar_volumen` + `validar_audio` + `analizar_senal` con los modelos cargados una sola vez (caché a nivel de módulo). Propaga `ErrorAudio`, `ErrorEntrada` y `ErrorModelo`.
  - `src/evaluar.py`:
    - `leer_etiquetas_audacity(ruta: Path) -> list[tuple[float, float, int]]` — líneas `inicio<TAB>fin<TAB>etiqueta`; etiqueta sin distinguir mayúsculas en `{'es', 'español', 'espanol'}` → `ES` y `{'qu', 'quechua'}` → `QU`. Ignora líneas vacías; cualquier otra línea inválida → `ValueError('Línea <n> de <archivo>: <motivo>')`.
    - `etiquetas_desde_tramos(tramos, n: int, mascara: np.ndarray) -> np.ndarray` — int `(n,)`; fuera de un tramo o sin habla → `IGNORAR`.
    - `cambios_de_referencia(meta: dict) -> list[float]` — fronteras entre fragmentos consecutivos de idioma distinto en la metadata de una frase sintética.
    - `evaluar_particion(dir_particion: Path, predictores: dict[str, Callable[[np.ndarray], np.ndarray]]) -> dict` — por modelo: `{'trama': metricas_trama global, 'cambios': prf global (vp, fp y fn sumados sobre las frases), 'por_tipo': {tipo: {'trama', 'cambios'}}, 'por_duracion': {'<1s' | '1-2s' | '>=2s': exactitud de las tramas de fragmentos de esa duración}, 'falsos_cambios_control': tasa_falsos_cambios}`. Cada predictor recibe el audio y devuelve etiquetas por trama ya suavizadas. Las frases se recorren en el orden de `meta.jsonl`.
    - `evaluar_reales(dir_reales: Path, predictores) -> dict | None` — pares `<nombre>.wav` + `<nombre>.txt`; `None` si la carpeta no existe o está vacía.
    - Script: `python src/evaluar.py` evalúa sobre la partición de prueba los tres predictores (`'profundo'`, `'no_supervisado'` y `'linea_base'`, que devuelve la clase mayoritaria de las tramas de entrenamiento). Agrega, solo para el no supervisado, `ari`, `nmi` y `pureza` a nivel de ventana. Escribe `resultados/metricas.json` (`{'prueba': {...}, 'reales': {...} | None, 'no_supervisado_ventanas': {...}, 'generado': fecha ISO}`) y las figuras en `resultados/figuras/`: `curvas_entrenamiento.png`, `matriz_profundo.png`, `matriz_no_supervisado.png`, `silueta_k.png`, `ejemplos_lineas_tiempo.png` (3 frases de prueba: una mixta, una monolingüe y una de control, con la verdad y ambos modelos). Si falta un modelo, termina con código 1 y el mensaje de su `ErrorModelo`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_inferencia.py
def test_validar_corto(tono):
    with pytest.raises(ErrorEntrada, match='menos de 1 segundo'):
        validar_audio(tono(0.5))

def test_validar_largo():
    with pytest.raises(ErrorEntrada, match='más de 5 minutos'):
        validar_audio(np.zeros(int(301 * SR), np.float32) + 0.1)

def test_validar_sin_habla(silencio):
    with pytest.raises(ErrorEntrada, match='No se detectó habla'):
        validar_audio(silencio(3.0))

def test_analizar_senal_forma(tono, modelo_ns_entrenado):
    r = analizar_senal(tono(3.0), CRNN().eval(), modelo_ns_entrenado)
    assert r['duracion'] == pytest.approx(3.0, abs=0.01)
    for nombre in ('profundo', 'no_supervisado'):
        for t in r['modelos'][nombre]['tramos']:
            assert t['idioma'] in ('es', 'qu') and 0 <= t['inicio'] < t['fin'] <= 3.01
```

```python
# tests/test_evaluar.py
def test_leer_audacity(tmp_path):
    (tmp_path / 'a.txt').write_text('0.0\t1.5\tEspañol\n1.5\t3.2\tqu\n\n')
    assert leer_etiquetas_audacity(tmp_path / 'a.txt') == [(0.0, 1.5, ES), (1.5, 3.2, QU)]

def test_leer_audacity_invalido(tmp_path):
    (tmp_path / 'b.txt').write_text('0.0\t1.5\taymara\n')
    with pytest.raises(ValueError, match='Línea 1'):
        leer_etiquetas_audacity(tmp_path / 'b.txt')

def test_cambios_de_referencia():
    meta = {'fragmentos': [{'idioma': 'es', 'inicio': 0, 'fin': 2.0},
                           {'idioma': 'es', 'inicio': 2.0, 'fin': 3.0},
                           {'idioma': 'qu', 'inicio': 3.05, 'fin': 5.0}]}
    assert cambios_de_referencia(meta) == pytest.approx([3.025])

def test_evaluar_particion_oraculo(corpus_falso):
    # un predictor que devuelve exactamente las etiquetas guardadas → exactitud 1 y cero falsos cambios
    r = evaluar_particion(corpus_falso / 'validacion', {'oraculo': oraculo_de(corpus_falso / 'validacion')})
    assert r['oraculo']['trama']['exactitud'] == 1.0
    assert r['oraculo']['falsos_cambios_control'] == 0.0

def test_evaluar_reales_vacio(tmp_path):
    assert evaluar_reales(tmp_path / 'no_existe', {}) is None
```

(`oraculo_de(dir)` es un helper del test: devuelve una función que, en la n-ésima llamada, carga el `.npz` de la n-ésima frase de `meta.jsonl` y devuelve sus etiquetas.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inferencia.py tests/test_evaluar.py -q`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/inferencia.py` y `src/evaluar.py` con las firmas de Interfaces**

Las figuras con matplotlib, backend `Agg`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: toda la suite pasa.

- [ ] **Step 5: Commit**

```bash
git add "segundo parcial/src/inferencia.py" "segundo parcial/src/evaluar.py" "segundo parcial/tests/test_inferencia.py" "segundo parcial/tests/test_evaluar.py"
git commit -m "feat: add inference API and evaluation with real-recording support"
```

---

### Task 8: Corrida real con los datos de Common Voice

Requiere que el usuario haya puesto en `datos/crudo/` el `.tar.gz` de Puno Quechua y los de Mexican Spanish (Male y Female). Si no están, parar y pedírselos.

**Files:**
- Create: `resultados/` (salidas de la corrida), `modelos/profundo.pt`, `modelos/no_supervisado.joblib`, `docs/adr/0001-datos-sinteticos-y-fuentes.md`

**Interfaces:**
- Consumes: scripts de Tasks 2, 3, 5, 6 y 7.
- Produces: `resultados/datos.json`, `resultados/entrenamiento_profundo.json`, `resultados/no_supervisado.json`, `resultados/metricas.json`, `resultados/figuras/*.png`, los dos modelos, y `resultados/tiempos.json` (`{paso: segundos}`, lo usa Task 9 para estimar duraciones).

- [ ] **Step 1: Preparar los clips**

Run: `.venv/bin/python src/preparar.py`
Expected: `datos/clips/indice.csv` existe; `resultados/datos.json` muestra los dos idiomas con hablantes en las tres particiones.

- [ ] **Step 2: Sintetizar el corpus**

Run: `.venv/bin/python src/sintetizar.py`
Expected: unas 4 000 frases de entrenamiento y unas 450 de validación y de prueba; en `meta.jsonl` las proporciones de tipos quedan cerca de 60/20/20.

- [ ] **Step 3: Escuchar 5 frases mixtas y 5 de control**

Verificación manual: que las uniones no se oigan como cortes. Si se oyen clics, ajustar el fundido dentro de 20–50 ms y repetir el Step 2.

- [ ] **Step 4: Entrenar el modelo profundo**

Run: `.venv/bin/python src/modelo_profundo.py`
Expected: termina en menos de 60 minutos y el F1 de validación sube respecto a la época 1. Si una época tarda más de 3 minutos, reducir horas de entrenamiento o canales y anotarlo en el ADR.

- [ ] **Step 5: Entrenar el no supervisado y evaluar**

Run: `.venv/bin/python src/modelo_no_supervisado.py && .venv/bin/python src/evaluar.py`
Expected: existen `resultados/metricas.json` y las 5 figuras. El F1 del modelo profundo supera a la línea base.

- [ ] **Step 5b: Guardar los tiempos medidos**

Cada script de los Steps 1, 2, 4 y 5 imprime al final `Duración total: <segundos> s`; con esos valores, escribir `resultados/tiempos.json` (`{'preparar', 'sintetizar', 'profundo', 'no_supervisado', 'evaluar'}`). El servidor de Task 9 lo reescribe tras cada corrida exitosa.

- [ ] **Step 6: Registrar las decisiones en el ADR**

`docs/adr/0001-datos-sinteticos-y-fuentes.md`: corpus sintético y su porqué, fuentes y licencias (CC0), volumen final y cualquier ajuste hecho en los Steps 3–4. En prosa, sin código.

- [ ] **Step 7: Commit**

```bash
git add "segundo parcial/resultados" "segundo parcial/modelos" "segundo parcial/docs/adr"
git commit -m "chore: add outputs and models from the reference run"
```

---

### Task 9: Aplicación web

**Antes de escribir la interfaz, cargar la skill `rodrigo-frontend` y seguir su flujo.**

**Files:**
- Create: `src/servidor.py`, `src/app.html`, `tests/test_servidor.py`

**Interfaces:**
- Consumes: `analizar`, `ErrorEntrada`, `ErrorAudio`, `ErrorModelo`; `resultados/*.json`; `datos/sintetico/prueba/`.
- Produces: el servidor, con la misma estructura que raíz `src/servidor.py` (copiar y adaptar las clases `Corrida` y `Manejador`, el flujo SSE y `PUERTOS = range(8000, 8011)`). Lista `PASOS` con ids `preparar`, `sintetizar`, `profundo`, `no_supervisado`, `evaluar`, `informe`, en ese orden y con `segundos` tomados de `resultados/tiempos.json`.
  - `GET /` → `app.html`
  - `GET /api/estado` → `{'datos_crudos': bool, 'clips': bool, 'corpus': bool, 'modelos': {'profundo': bool, 'no_supervisado': bool}, 'salidas': {...}}`
  - `GET /api/eventos`, `POST /api/correr`, `POST /api/cancelar` → como en el primer parcial.
  - `GET /api/frases?tipo=<mixta|monolingue|control>` → hasta 20 frases de prueba `{'id', 'tipo', 'duracion'}`.
  - `GET /api/frase/<id>` → `analizar` de esa frase más `'verdad': {'tramos', 'cambios'}` leído de su metadata.
  - `POST /api/analizar` (cuerpo `audio/wav`, máximo 20 MB) → JSON de `analizar`. Errores: `ErrorEntrada` o `ErrorAudio` → 400 `{'error': mensaje}`; `ErrorModelo` → 409 `{'error': mensaje}`; cuerpo de más de 20 MB → 413 `{'error': 'El archivo supera 20 MB.'}`.
  - `GET /audio/prueba/<id>.wav` y `GET /resultados/<ruta>` → archivos, resueltos con `Path.resolve()` y verificados con `is_relative_to` de su carpeta; si no, 404.
- `app.html`: vistas **Datos**, **Entrenar**, **Resultados** y **Probar** (§7 del spec). En Probar, el archivo elegido o la grabación (`MediaRecorder`) se decodifica con `AudioContext.decodeAudioData`, se remuestrea a 16 kHz mono con `OfflineAudioContext` y se codifica como WAV PCM16 en JS antes del `POST`. Línea de tiempo por modelo (y la verdad cuando exista), con español y quechua en colores distinguibles también para daltónicos, la no-habla en gris, los cambios marcados y un cursor sincronizado con el `<audio>`. El texto de error del servidor se muestra tal cual.

- [ ] **Step 1: Escribir los tests que fallan**

Fixture `servidor_vivo` que levanta `ThreadingHTTPServer` en un puerto libre con `inferencia.analizar` parcheado para devolver un resultado fijo o lanzar el error pedido. Helpers del test: `post(url, cuerpo, tipo, error=None)` y `get(url)` con `urllib.request`, que devuelven la respuesta también en estados 4xx; `wav_de(y)` codifica a WAV con `soundfile` en memoria.

```python
def test_analizar_audio_invalido(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', b'basura', 'audio/wav', error=ErrorAudio('No se pudo leer el audio: x'))
    assert r.status == 400 and 'No se pudo leer' in json.loads(r.read())['error']

def test_analizar_sin_modelo(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', wav_de(tono(2.0)), 'audio/wav', error=ErrorModelo('no está entrenado'))
    assert r.status == 409

def test_analizar_demasiado_grande(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', b'0' * (20 * 1024 * 1024 + 1), 'audio/wav')
    assert r.status == 413

def test_analizar_ok(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', wav_de(tono(2.0)), 'audio/wav')
    assert r.status == 200 and 'modelos' in json.loads(r.read())

@pytest.mark.parametrize('ruta', ['/resultados/../src/servidor.py', '/audio/prueba/..%2F..%2Fsrc%2Fconfig.py'])
def test_sin_salir_de_carpeta(servidor_vivo, ruta):
    assert get(servidor_vivo + ruta).status == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_servidor.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'servidor'`

- [ ] **Step 3: Implementar `src/servidor.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_servidor.py -q`
Expected: `6 passed`

- [ ] **Step 5: Implementar `src/app.html` siguiendo `rodrigo-frontend`**

- [ ] **Step 6: Probar en el navegador de punta a punta**

Run: `npm run dev`, luego, con Playwright: abrir la URL; en Probar elegir una frase mixta y comprobar que las tres líneas de tiempo se dibujan y el cursor avanza; subir un `.mp3` de Common Voice; subir un `.txt` renombrado a `.wav` y comprobar el mensaje de error; revisar que la consola no tenga errores. Grabar con el micrófono queda como verificación manual del usuario.
Expected: todo lo anterior se cumple; capturas guardadas en `resultados/figuras/app_*.png` para el informe.

- [ ] **Step 7: Commit**

```bash
git add "segundo parcial/src/servidor.py" "segundo parcial/src/app.html" "segundo parcial/tests/test_servidor.py" "segundo parcial/resultados/figuras"
git commit -m "feat: add local web app to train models and analyze audio"
```

---

### Task 10: Informe y README

**Files:**
- Create: `docs/informe/contenido.md`, `src/informe.py`, `tests/test_informe.py`, `README.md`
- Output: `docs/INFORME FINAL - DETECCION DE CODE-SWITCHING.docx`

**Interfaces:**
- Consumes: `resultados/*.json`, `resultados/figuras/*.png`.
- Produces:
  - `docs/informe/contenido.md` — prosa del informe en español con la estructura I–VII del PDF de la práctica y la carátula de `docs/Propuestas machine.pdf` (raíz). Todas las cifras van como marcadores `$nombre` de `string.Template` (p. ej. `$f1_profundo`, `$horas_qu_entrenamiento`). Las figuras se insertan como imágenes Markdown que apuntan a `resultados/figuras/`. Incluye las limitaciones de §9 del spec y una Bibliografía con fuentes reales y verificables (Common Voice, Puno Quechua en Common Voice, SpecAugment, CRNN para identificación de idioma, K-means, GMM).
  - `valores(resultados: Path = RESULTADOS) -> dict[str, str]` — todas las cifras ya formateadas (porcentajes con 1 decimal, horas con 1 decimal).
  - `generar(salida: Path) -> Path` — sustituye con `Template.substitute` (falla si falta una cifra) y convierte con `pypandoc.convert_text(..., 'docx', format='md', outputfile=...)`, con tabla de contenido.
  - Script: `python src/informe.py` genera el `.docx`. Si falta `resultados/metricas.json`, termina con código 1 y el mensaje `Primero corre la evaluación (src/evaluar.py).`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_valores_cubren_marcadores():
    texto = (RAIZ / 'docs/informe/contenido.md').read_text()
    faltan = {m.group('named') or m.group('braced') for m in Template.pattern.finditer(texto)
              if m.group('named') or m.group('braced')} - set(valores())
    assert not faltan, faltan

def test_generar_docx(tmp_path):
    salida = generar(tmp_path / 'informe.docx')
    assert salida.exists() and salida.stat().st_size > 50_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_informe.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'informe'`

- [ ] **Step 3: Escribir `docs/informe/contenido.md` e implementar `src/informe.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_informe.py -q`
Expected: `2 passed`

- [ ] **Step 5: Generar el informe y revisarlo**

Run: `.venv/bin/python src/informe.py`
Expected: existe el `.docx`. Abrirlo (convertirlo a PDF con `pypandoc` o LibreOffice si está instalado) y revisar que las 7 secciones, las figuras y las tablas aparezcan y que no quede ningún `$marcador` sin sustituir.

- [ ] **Step 6: Escribir `README.md`**

Con las mismas secciones que el README raíz: qué hace, resultado (tabla con las cifras de `metricas.json`), requisitos, sección **Datos** (qué descargar de Mozilla Data Collective y dónde ponerlo), cómo correr, uso de la app, cómo agregar grabaciones reales en `datos/reales/` (formato de etiquetas de Audacity) y solución de problemas.

- [ ] **Step 7: Suite completa y commit**

Run: `.venv/bin/python -m pytest -q`
Expected: toda la suite pasa.

```bash
git add "segundo parcial/docs/informe" "segundo parcial/src/informe.py" "segundo parcial/tests/test_informe.py" "segundo parcial/README.md" "segundo parcial/docs/INFORME FINAL - DETECCION DE CODE-SWITCHING.docx"
git commit -m "docs: add final report generator, report and README"
```

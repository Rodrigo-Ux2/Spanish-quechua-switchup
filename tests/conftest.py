import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

SR = 16000


@pytest.fixture
def tono():
    def _tono(dur_s, f=220.0):
        t = np.arange(int(dur_s * SR)) / SR
        return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return _tono


@pytest.fixture
def silencio():
    return lambda dur_s: np.zeros(int(dur_s * SR), np.float32)


@pytest.fixture
def banco_falso(tono):
    """2 idiomas x 4 hablantes x 6 fragmentos de tono de 1 a 2 s (frecuencia distinta por hablante)."""
    banco = {}
    for i, idioma in enumerate(('es', 'qu')):
        banco[idioma] = {}
        for h in range(4):
            f = 200.0 + 100 * (i * 4 + h)
            banco[idioma][f'{idioma}{h}'] = [
                (tono(1.0 + 0.2 * k, f), 'M' if h % 2 else 'F') for k in range(6)]
    return banco


@pytest.fixture
def indice_falso(tmp_path, tono, silencio, monkeypatch):
    """Clips de tono con pausas y un indice.csv en tmp_path; parchea sintetizar.CLIPS."""
    import soundfile as sf
    clips = tmp_path / 'clips'
    filas = ['ruta,idioma,hablante,genero,particion,duracion']
    for i, idioma in enumerate(('es', 'qu')):
        (clips / idioma / 'entrenamiento').mkdir(parents=True)
        for h in range(4):
            f = 200.0 + 100 * (i * 4 + h)
            for n in range(3):
                y = np.concatenate([tono(1.5, f), silencio(0.3), tono(1.5, f),
                                    silencio(0.3), tono(1.5, f)])
                ruta = f'{idioma}/entrenamiento/{idioma}{h}_{n}.wav'
                sf.write(clips / ruta, y, SR)
                filas.append(f'{ruta},{idioma},{idioma}{h},M,entrenamiento,{len(y) / SR}')
    (clips / 'indice.csv').write_text('\n'.join(filas) + '\n', encoding='utf-8')
    monkeypatch.setattr('sintetizar.CLIPS', clips)
    return clips


@pytest.fixture(scope='session')
def corpus_falso(tmp_path_factory):
    """Corpus sintético mínimo con el formato de generar_particion: 40 frases de
    entrenamiento y 10 de validación; ES = tono estable, QU = tono con vibrato. Solo lectura."""
    import json

    import soundfile as sf

    from caracteristicas import logmel
    from sintetizar import componer

    def tono(dur_s, f, vibrato):
        # QU: tono con vibrato (FM de ±150 Hz a 5 Hz); ES: tono estable. La variabilidad
        # trama a trama sobrevive a la resta de media cepstral por frase, a diferencia de
        # una frecuencia constante
        t = np.arange(int(dur_s * SR)) / SR
        fase = 2 * np.pi * f * t - (150 / 5) * np.cos(2 * np.pi * 5 * t) * vibrato
        return (0.5 * np.sin(fase)).astype(np.float32)

    banco = {idioma: {f'{idioma}{h}': [(tono(0.5 + 0.1 * k, 200.0 + 20 * (i * 4 + h), i == 1),
                                         'M' if h % 2 else 'F') for k in range(6)]
                      for h in range(4)}
             for i, idioma in enumerate(('es', 'qu'))}
    raiz = tmp_path_factory.mktemp('sintetico')
    rng = np.random.default_rng(0)
    patron = ['mixta', 'mixta', 'mixta', 'monolingue', 'control']
    for particion, n in (('entrenamiento', 40), ('validacion', 10)):
        destino = raiz / particion
        destino.mkdir()
        with open(destino / 'meta.jsonl', 'w', encoding='utf-8') as meta:
            for i in range(n):
                id_ = f'{particion[:3]}_{i:05d}'
                frase = componer(patron[i % len(patron)], banco, rng)
                sf.write(destino / f'{id_}.wav', frase['audio'], SR, subtype='PCM_16')
                np.savez_compressed(destino / f'{id_}.npz', logmel=logmel(frase['audio']),
                                    etiquetas=frase['etiquetas'])
                meta.write(json.dumps({'id': id_, **frase['meta']}, ensure_ascii=False) + '\n')
    return raiz


@pytest.fixture(scope='session')
def corpus_falso_invertido(corpus_falso, tmp_path_factory):
    """Copia de corpus_falso con ES y QU intercambiados en los .npz y en meta.jsonl."""
    import json
    import shutil

    raiz = tmp_path_factory.mktemp('sintetico_invertido') / 'corpus'
    shutil.copytree(corpus_falso, raiz)
    for npz in raiz.glob('*/*.npz'):
        datos = dict(np.load(npz))
        e = datos['etiquetas']
        datos['etiquetas'] = np.where(e == 0, 1, np.where(e == 1, 0, e)).astype(e.dtype)
        np.savez_compressed(npz, **datos)
    cambio = {'es': 'qu', 'qu': 'es'}
    for meta in raiz.glob('*/meta.jsonl'):
        filas = [json.loads(linea) for linea in meta.read_text(encoding='utf-8').splitlines()]
        for fila in filas:
            for frag in fila['fragmentos']:
                frag['idioma'] = cambio[frag['idioma']]
        meta.write_text('\n'.join(json.dumps(f, ensure_ascii=False) for f in filas) + '\n',
                        encoding='utf-8')
    return raiz


@pytest.fixture(scope='session')
def modelo_ns_entrenado(corpus_falso, tmp_path_factory):
    """Modelo no supervisado entrenado sobre corpus_falso en carpetas temporales."""
    from modelo_no_supervisado import cargar_no_supervisado, entrenar_no_supervisado

    destino = tmp_path_factory.mktemp('modelo_ns')
    entrenar_no_supervisado(corpus_falso, max_ventanas=5000, dir_modelos=destino,
                            dir_resultados=destino)
    return cargar_no_supervisado(destino / 'no_supervisado.joblib')

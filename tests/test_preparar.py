import io
import tarfile

import pytest
import soundfile as sf

from preparar import (dividir_hablantes, leer_validados, locale_de_tar,
                      preparar, seleccionar)
from preparar import main as preparar_main

CABECERA = 'client_id\tpath\tsentence\tgender\n'


def _agregar(tar, nombre, datos: bytes):
    info = tarfile.TarInfo(nombre)
    info.size = len(datos)
    tar.addfile(info, io.BytesIO(datos))


def _wav(tono):
    b = io.BytesIO()
    sf.write(b, tono(1.5), 16000, format='WAV')
    return b.getvalue()


@pytest.fixture
def tar_falso(tmp_path, tono):
    """Arma un .tar.gz con estructura cv-corpus-26.0/<locale>/validated.tsv."""
    def _crear(locale, hablantes):
        filas, clips = [], []
        for h, n in hablantes.items():
            for i in range(n):
                filas.append(f'{h}\t{h}_{i}.wav\tfrase\tfemale_feminine\n')
                clips.append(f'{h}_{i}.wav')
        destino = tmp_path / 'crudo'
        destino.mkdir(exist_ok=True)
        ruta = destino / f'{locale}.tar.gz'
        base = f'cv-corpus-26.0/{locale}'
        with tarfile.open(ruta, 'w:gz') as t:
            _agregar(t, f'{base}/validated.tsv', (CABECERA + ''.join(filas)).encode())
            for c in clips:
                _agregar(t, f'{base}/clips/{c}', _wav(tono))
        return ruta
    return _crear


@pytest.fixture
def tar_segmento(tmp_path):
    """Layout de segmento: es-MX_female/{train,dev}.tsv, sin validated.tsv."""
    cab = 'client_id\tpath\tsentence_id\tsentence\tgender\tlocale\n'
    train = cab + 'a\t1.mp3\ts\tx\tfemale_feminine\tes\n' + 'b\t2.mp3\ts\ty\tfemale_feminine\tes\n'
    dev = cab + 'b\t2.mp3\ts\ty\tfemale_feminine\tes\n' + 'c\t3.mp3\ts\tz\tmale_masculine\tes\n'
    ruta = tmp_path / 'es-MX_female.tar.gz'
    with tarfile.open(ruta, 'w:gz') as t:
        _agregar(t, 'es-MX_female/train.tsv', train.encode())
        _agregar(t, 'es-MX_female/dev.tsv', dev.encode())
        _agregar(t, 'es-MX_female/clips/1.mp3', b'x')
    return ruta


def test_locale_y_validados(tar_falso):
    t = tar_falso('qxp', {'h1': 3, 'h2': 2})
    assert locale_de_tar(t) == 'qxp'
    filas = leer_validados(t)
    assert len(filas) == 5 and {f['hablante'] for f in filas} == {'h1', 'h2'}
    assert {f['genero'] for f in filas} == {'F'}


def test_segmento_sin_validated(tar_segmento):
    assert locale_de_tar(tar_segmento) == 'es'
    filas = leer_validados(tar_segmento)
    assert sorted(f['archivo'] for f in filas) == ['1.mp3', '2.mp3', '3.mp3']
    assert {f['archivo']: f['genero'] for f in filas}['3.mp3'] == 'M'


def test_dividir_hablantes_disjunto():
    p = dividir_hablantes([f'h{i}' for i in range(50)])
    assert sorted(set(p.values())) == ['entrenamiento', 'prueba', 'validacion']
    assert list(p.values()).count('entrenamiento') == 40


def test_dividir_pocos_hablantes():
    p = dividir_hablantes(['a', 'b', 'c'])
    assert sorted(p.values()) == ['entrenamiento', 'prueba', 'validacion']


def test_seleccionar_tope_por_hablante():
    filas = [{'hablante': 'a', 'archivo': f'{i}.mp3', 'genero': ''} for i in range(500)]
    assert len(seleccionar(filas, max_por_hablante=200, horas_max=100, duracion_media_s=4)) == 200


def test_seleccionar_tope_de_horas():
    filas = [{'hablante': f'h{i}', 'archivo': f'{i}.mp3', 'genero': ''} for i in range(100)]
    assert len(seleccionar(filas, max_por_hablante=200, horas_max=1, duracion_media_s=360)) == 10


def test_preparar_escribe_clips(tar_falso, tmp_path, monkeypatch):
    monkeypatch.setattr('preparar.CLIPS', tmp_path / 'clips')
    t = tar_falso('es', {f'h{i}': 2 for i in range(10)})
    filas = preparar([t])
    assert filas and all(f['idioma'] == 'es' for f in filas)
    por_hablante = {}
    for f in filas:
        por_hablante.setdefault(f['hablante'], set()).add(f['particion'])
    assert all(len(v) == 1 for v in por_hablante.values())
    assert all((tmp_path / 'clips' / f['ruta']).exists() for f in filas)


def test_falta_idioma(tmp_path, monkeypatch, tar_falso, capsys):
    monkeypatch.setattr('preparar.CRUDO', tar_falso('es', {'h1': 1}).parent)
    with pytest.raises(SystemExit) as e:
        preparar_main()
    assert e.value.code == 1 and 'Falta el audio de Quechua' in capsys.readouterr().err

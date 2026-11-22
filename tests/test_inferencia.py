import numpy as np
import pytest

import inferencia
from inferencia import ErrorEntrada, analizar, analizar_senal, validar_audio
from modelo_profundo import CRNN, ErrorModelo

SR = 16000


def test_validar_corto(tono):
    with pytest.raises(ErrorEntrada, match='menos de 1 segundo'):
        validar_audio(tono(0.5))


def test_validar_largo():
    with pytest.raises(ErrorEntrada, match='más de 5 minutos'):
        validar_audio(np.zeros(int(301 * SR), np.float32) + 0.1)


def test_validar_sin_habla(silencio):
    with pytest.raises(ErrorEntrada, match='No se detectó habla'):
        validar_audio(silencio(3.0))


def test_validar_ok(tono):
    validar_audio(tono(3.0))


def test_analizar_senal_forma(tono, modelo_ns_entrenado):
    r = analizar_senal(tono(3.0), CRNN().eval(), modelo_ns_entrenado)
    assert r['duracion'] == pytest.approx(3.0, abs=0.01)
    assert r['modelos']['no_supervisado']['algoritmo'] == modelo_ns_entrenado['mejor']
    for nombre in ('profundo', 'no_supervisado'):
        assert isinstance(r['modelos'][nombre]['cambios'], list)
        for t in r['modelos'][nombre]['tramos']:
            assert t['idioma'] in ('es', 'qu') and 0 <= t['inicio'] < t['fin'] <= 3.01


def test_analizar_modelo_ausente_no_se_cachea(tono, tmp_path, monkeypatch):
    import soundfile as sf
    sf.write(tmp_path / 'a.wav', tono(2.0), SR)
    monkeypatch.setattr(inferencia, '_modelos', None)
    monkeypatch.setattr(inferencia, 'MODELOS', tmp_path)
    with pytest.raises(ErrorModelo):
        analizar(tmp_path / 'a.wav')
    assert inferencia._modelos is None

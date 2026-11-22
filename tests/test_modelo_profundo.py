import numpy as np
import pytest
import torch

from audio import n_tramas
from config import IGNORAR
from modelo_profundo import (CRNN, ErrorModelo, cargar_profundo, entrenar,
                             predecir_profundo, rellenar)


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
    assert (tmp_path / 'entrenamiento_profundo.json').exists()
    assert isinstance(cargar_profundo(tmp_path / 'profundo.pt'), CRNN)


def test_cargar_sin_entrenar(tmp_path):
    with pytest.raises(ErrorModelo, match='no está entrenado'):
        cargar_profundo(tmp_path / 'no.pt')


def test_predecir_ignora_silencio(tono, silencio):
    y = np.concatenate([tono(1.0), silencio(1.0)])
    p = predecir_profundo(CRNN().eval(), y)
    assert len(p) == n_tramas(len(y)) and (p[150:190] == IGNORAR).all()

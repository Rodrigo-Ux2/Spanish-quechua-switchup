import numpy as np
import pytest

from config import ES, QU, IGNORAR
from metricas import (metricas_trama, contar_cambios, prf,
                      tasa_falsos_cambios, mapear_grupos)


def test_metricas_trama_ignora():
    ref = np.array([ES, ES, QU, QU, IGNORAR])
    pred = np.array([ES, QU, QU, QU, ES])
    m = metricas_trama(ref, pred)
    assert m['n'] == 4 and m['exactitud'] == 0.75 and m['matriz'] == [[1, 1], [0, 2]]


def test_metricas_trama_sin_tramas_validas():
    m = metricas_trama(np.full(3, IGNORAR), np.array([ES, QU, ES]))
    assert m['n'] == 0 and m['exactitud'] == 0.0 and m['f1_macro'] == 0.0
    assert m['matriz'] == [[0, 0], [0, 0]]


def test_metricas_trama_pred_ignorar_es_error():
    m = metricas_trama(np.array([ES, QU]), np.array([ES, IGNORAR]))
    assert m['n'] == 2 and m['exactitud'] == 0.5
    assert m['matriz'] == [[1, 0], [1, 0]]


def test_contar_cambios_tolerancia():
    assert contar_cambios([1.0, 3.0], [1.15, 2.5, 3.1]) == (2, 1, 0)


def test_contar_cambios_uno_a_uno():
    # Dos predicciones cerca de una sola referencia: una es falso positivo.
    assert contar_cambios([1.0], [0.95, 1.05]) == (1, 1, 0)
    assert contar_cambios([], []) == (0, 0, 0)


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


def test_mapear_grupos_ignora_y_un_solo_grupo():
    g = np.array([5, 5, 5, 5])
    r = np.array([ES, ES, IGNORAR, QU])
    assert set(mapear_grupos(g, r)) == {5}

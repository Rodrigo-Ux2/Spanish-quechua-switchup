import numpy as np
import pytest

from audio import n_tramas
from config import ES, IGNORAR, QU
from modelo_no_supervisado import (cargar_no_supervisado, entrenar_no_supervisado,
                                   predecir_no_supervisado)
from modelo_profundo import ErrorModelo


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


def test_completar_mapa_y_predecir_con_grupo_ausente(modelo_ns_entrenado, tono):
    from modelo_no_supervisado import _completar_mapa
    assert _completar_mapa({}) == {0: ES, 1: QU}
    assert _completar_mapa({0: QU}) == {0: QU, 1: ES}
    assert _completar_mapa({1: ES}) == {0: QU, 1: ES}
    assert _completar_mapa({0: ES, 1: ES}) == {0: ES, 1: ES}
    # un modelo cuyo mapa vino de una validación con un solo grupo no debe fallar
    modelo = {**modelo_ns_entrenado,
              'mapa': {a: _completar_mapa({0: QU}) for a in ('kmeans', 'gmm')}}
    p = predecir_no_supervisado(modelo, tono(2.0, f=200))
    assert set(np.unique(p)) <= {ES, QU}

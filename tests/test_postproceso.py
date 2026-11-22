import numpy as np
import pytest

from config import ES, QU, IGNORAR
from postproceso import suavizar, tramos, puntos_cambio


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


def test_entrada_vacia_o_sin_habla():
    for e in (np.array([], dtype=int), np.full(30, IGNORAR)):
        assert (suavizar(e) == e).all()
        assert tramos(e) == []


def test_suavizar_un_solo_tramo_corto_se_deja():
    e = np.array([ES] * 5)
    assert (suavizar(e) == ES).all()


def test_suavizar_fusiona_tramo_corto_con_vecino_mas_largo():
    # La mediana no lo quita (ventana 3); el tramo de 5 tramas se absorbe.
    e = np.array([ES] * 50 + [QU] * 5 + [ES] * 10 + [QU] * 50)
    s = suavizar(e, mediana=3, min_tramas=20)
    assert [x[2] for x in tramos(s)] == [ES, QU]
    assert (s[:65] == ES).all() and (s[65:] == QU).all()

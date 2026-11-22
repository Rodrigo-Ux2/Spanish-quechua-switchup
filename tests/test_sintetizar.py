import json

import numpy as np
import pytest

from audio import n_tramas
from config import ES, HOP, IDIOMAS, IGNORAR, QU, SR
from sintetizar import _aumentar, componer, fragmentar, generar_particion, main


def test_fragmentar_duraciones(tono, silencio):
    y = np.concatenate([tono(1.2), silencio(0.3), tono(0.8), silencio(0.3), tono(4.0)])
    fragmentos = fragmentar(y, np.random.default_rng(0))
    assert fragmentos
    for f in fragmentos:
        assert 0.5 * SR <= len(f) <= 3.0 * SR


@pytest.mark.parametrize('tipo', ['mixta', 'monolingue', 'control'])
def test_componer_reglas(tipo, banco_falso):
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = componer(tipo, banco_falso, rng)
        m = f['meta']
        assert 4.0 <= m['duracion'] <= 12.0
        assert len(f['etiquetas']) == n_tramas(len(f['audio']))
        assert f['etiquetas'].dtype == np.int16
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
        for a, b in zip(m['fragmentos'], m['fragmentos'][1:]):
            assert a['inicio'] < b['inicio'] and a['fin'] <= m['duracion'] + 1e-6


@pytest.mark.parametrize('tipo', ['mixta', 'monolingue', 'control'])
def test_componer_determinista(tipo, banco_falso):
    a = componer(tipo, banco_falso, np.random.default_rng(7))
    b = componer(tipo, banco_falso, np.random.default_rng(7))
    assert np.array_equal(a['audio'], b['audio'])
    assert np.array_equal(a['etiquetas'], b['etiquetas'])


def test_etiquetas_siguen_el_idioma_de_cada_fragmento(banco_falso):
    rng = np.random.default_rng(5)
    for _ in range(10):
        f = componer('mixta', banco_falso, rng)
        et = f['etiquetas']
        for fr in f['meta']['fragmentos']:
            esperado = IDIOMAS[fr['idioma']]
            # centro y puntos a 100 ms de cada borde (fuera de la zona de unión)
            for t in ((fr['inicio'] + fr['fin']) / 2, fr['inicio'] + 0.1, fr['fin'] - 0.1):
                assert et[int(t * SR / HOP)] == esperado
        assert (et == ES).any() and (et == QU).any()


def test_aumento_snr_ganancia_y_recorte(tono):
    rng = np.random.default_rng(0)
    x = tono(2.0, 300.0) * 0.6  # amplitud 0.3: sin recorte para ganancias de hasta +6 dB
    for _ in range(20):
        y = _aumentar(x, rng)
        a = float(np.dot(y, x) / np.dot(x, x))  # ganancia efectiva
        residuo = y - a * x
        snr = 10 * np.log10(a**2 * np.sum(x**2) / np.sum(residuo**2))
        assert 19.0 <= snr <= 41.0
        assert 10 ** (-6.1 / 20) <= a <= 10 ** (6.1 / 20)
    # amplitud 4: ni con -6 dB de ganancia baja de 1, así que el recorte siempre actúa
    fuerte = _aumentar(tono(1.0) * 8.0, rng)
    assert np.isclose(np.abs(fuerte).max(), 1.0)
    assert fuerte.max() <= 1.0 and fuerte.min() >= -1.0


def test_componer_banco_insuficiente(tono):
    corto = {i: {f'{i}{h}': [(tono(0.5), 'M')] for h in range(2)} for i in ('es', 'qu')}
    with pytest.raises(ValueError, match='banco de fragmentos'):
        componer('mixta', corto, np.random.default_rng(0))


def test_generar_particion_falta_un_idioma(tmp_path, monkeypatch, indice_falso):
    monkeypatch.setattr('sintetizar.SINTETICO', tmp_path / 'sint')
    ruta = indice_falso / 'indice.csv'
    lineas = ruta.read_text(encoding='utf-8').splitlines()
    ruta.write_text('\n'.join(l for l in lineas if not l.startswith('qu/')) + '\n',
                    encoding='utf-8')
    with pytest.raises(ValueError, match='faltan: qu'):
        generar_particion('entrenamiento', 5, semilla=1)


def test_main_sin_indice_sale_con_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('sintetizar.CLIPS', tmp_path)
    monkeypatch.setattr('sys.argv', ['sintetizar.py'])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert 'Primero corre la preparación de datos (src/preparar.py).' in err


def test_generar_particion_proporciones(tmp_path, monkeypatch, indice_falso):
    monkeypatch.setattr('sintetizar.SINTETICO', tmp_path)
    generar_particion('entrenamiento', 100, semilla=3)
    tipos = [json.loads(l)['tipo'] for l in (tmp_path / 'entrenamiento' / 'meta.jsonl').read_text().splitlines()]
    assert len(tipos) == 100 and 50 <= tipos.count('mixta') <= 70
    assert len(list((tmp_path / 'entrenamiento').glob('*.npz'))) == 100
    z = np.load(next((tmp_path / 'entrenamiento').glob('*.npz')))
    assert z['logmel'].shape[1] == 64 and len(z['logmel']) == len(z['etiquetas'])

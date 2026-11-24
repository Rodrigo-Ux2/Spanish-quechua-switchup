import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import soundfile as sf

import inferencia
import servidor
from audio import ErrorAudio
from modelo_profundo import ErrorModelo

SR = 16000
RESULTADO_FIJO = {'duracion': 2.0,
                  'modelos': {'profundo': {'tramos': [{'inicio': 0.0, 'fin': 2.0, 'idioma': 'es'}],
                                           'cambios': []},
                              'no_supervisado': {'tramos': [], 'cambios': [], 'algoritmo': 'kmeans'}}}
_falla = {'error': None}  # error que lanzará analizar en la próxima petición


@pytest.fixture
def servidor_vivo(monkeypatch):
    def analizar_falso(ruta):
        if _falla['error'] is not None:
            raise _falla['error']
        return RESULTADO_FIJO
    monkeypatch.setattr(inferencia, 'analizar', analizar_falso)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), servidor.Manejador)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()
    srv.server_close()
    _falla['error'] = None


def _abrir(peticion):
    """Estado y cuerpo de la respuesta, también en 4xx/5xx; la conexión queda cerrada."""
    try:
        respuesta = urllib.request.urlopen(peticion, timeout=30)
    except urllib.error.HTTPError as e:
        respuesta = e
    with respuesta:
        cuerpo = respuesta.read()
    return SimpleNamespace(status=respuesta.status, read=lambda: cuerpo)


def post(url, cuerpo, tipo, error=None):
    _falla['error'] = error
    return _abrir(urllib.request.Request(url, data=cuerpo, method='POST',
                                         headers={'Content-Type': tipo}))


def get(url):
    return _abrir(urllib.request.Request(url))


def wav_de(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def test_analizar_audio_invalido(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', b'basura', 'audio/wav', error=ErrorAudio('No se pudo leer el audio: x'))
    assert r.status == 400 and 'No se pudo leer' in json.loads(r.read())['error']


def test_analizar_sin_modelo(servidor_vivo, tono):
    r = post(servidor_vivo + '/api/analizar', wav_de(tono(2.0)), 'audio/wav', error=ErrorModelo('no está entrenado'))
    assert r.status == 409


def test_analizar_demasiado_grande(servidor_vivo):
    r = post(servidor_vivo + '/api/analizar', b'0' * (20 * 1024 * 1024 + 1), 'audio/wav')
    assert r.status == 413


def test_analizar_ok(servidor_vivo, tono):
    r = post(servidor_vivo + '/api/analizar', wav_de(tono(2.0)), 'audio/wav')
    assert r.status == 200 and 'modelos' in json.loads(r.read())


def test_sirve_fuentes(servidor_vivo):
    with urllib.request.urlopen(servidor_vivo + '/fuentes/atkinson-next.woff2', timeout=30) as r:
        assert r.status == 200 and r.headers['Content-Type'] == 'font/woff2'


@pytest.mark.parametrize('ruta', ['/resultados/../src/servidor.py', '/audio/prueba/..%2F..%2Fsrc%2Fconfig.py',
                                  '/fuentes/..%2Fservidor.py'])
def test_sin_salir_de_carpeta(servidor_vivo, ruta):
    assert get(servidor_vivo + ruta).status == 404


@pytest.fixture
def prueba_falsa(tmp_path, monkeypatch, tono):
    metas = [{'id': f'pru_{i:05d}', 'tipo': tipo, 'duracion': 2.0,
              'fragmentos': [{'idioma': 'es', 'hablante': 'a', 'genero': 'F', 'inicio': 0.0, 'fin': 0.9},
                             {'idioma': 'qu', 'hablante': 'b', 'genero': 'M', 'inicio': 1.1, 'fin': 2.0}]}
             for i, tipo in enumerate(['mixta'] * 25 + ['control'])]
    (tmp_path / 'meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in metas), encoding='utf-8')
    sf.write(tmp_path / 'pru_00000.wav', tono(2.0), SR)
    monkeypatch.setattr(servidor, 'PRUEBA', tmp_path)
    return tmp_path


def test_frases_por_tipo(servidor_vivo, prueba_falsa):
    mixtas = json.loads(get(servidor_vivo + '/api/frases?tipo=mixta').read())
    assert len(mixtas) == 20 and all(f['tipo'] == 'mixta' for f in mixtas)
    assert set(mixtas[0]) == {'id', 'tipo', 'duracion'}
    assert get(servidor_vivo + '/api/frases?tipo=otra').status == 400


def test_frase_con_verdad(servidor_vivo, prueba_falsa):
    r = json.loads(get(servidor_vivo + '/api/frase/pru_00000').read())
    assert r['modelos'] == RESULTADO_FIJO['modelos']
    assert r['verdad']['tramos'] == [{'inicio': 0.0, 'fin': 0.9, 'idioma': 'es'},
                                     {'inicio': 1.1, 'fin': 2.0, 'idioma': 'qu'}]
    assert r['verdad']['cambios'] == [pytest.approx(1.0)]
    assert get(servidor_vivo + '/api/frase/no_existe').status == 404
    assert get(servidor_vivo + '/audio/prueba/pru_00000.wav').status == 200


def test_audio_por_rangos(servidor_vivo, prueba_falsa):
    completo = (prueba_falsa / 'pru_00000.wav').read_bytes()
    url = servidor_vivo + '/audio/prueba/pru_00000.wav'
    r = _abrir(urllib.request.Request(url, headers={'Range': 'bytes=100-199'}))
    assert r.status == 206 and r.read() == completo[100:200]
    r = _abrir(urllib.request.Request(url, headers={'Range': f'bytes={len(completo)}-'}))
    assert r.status == 416


def test_pasos_en_orden():
    assert [p['id'] for p in servidor.PASOS] == [
        'preparar', 'sintetizar', 'profundo', 'no_supervisado', 'evaluar', 'informe']
    assert all(p['segundos'] > 0 for p in servidor.PASOS)


@pytest.fixture
def sin_ejecutar(monkeypatch):
    """Corrida que no lanza ningún script: solo queda terminada."""
    monkeypatch.setattr(servidor, 'corrida', None)
    monkeypatch.setattr(servidor.Corrida, 'ejecutar', lambda self: setattr(self, 'terminada', True))


def correr(url, cuerpo, tipo='application/json', origen=None):
    cabeceras = {'Content-Type': tipo, **({'Origin': origen} if origen else {})}
    return _abrir(urllib.request.Request(url + '/api/correr', data=cuerpo, method='POST', headers=cabeceras))


def test_correr_desde_otro_origen(servidor_vivo, sin_ejecutar):
    r = correr(servidor_vivo, b'{"pasos": ["evaluar"]}', 'text/plain', origen='https://evil.example')
    assert r.status == 403 and 'error' in json.loads(r.read())
    assert servidor.corrida is None


def test_correr_sin_json(servidor_vivo, sin_ejecutar):
    r = correr(servidor_vivo, b'{"pasos": ["evaluar"]}', 'text/plain')
    assert r.status == 400 and servidor.corrida is None


def test_correr_paso_desconocido(servidor_vivo, sin_ejecutar):
    r = correr(servidor_vivo, b'{"pasos": ["borrar_todo"]}', origen=servidor_vivo)
    assert r.status == 400 and servidor.corrida is None


def test_correr_mismo_origen(servidor_vivo, sin_ejecutar):
    origen = servidor_vivo.replace('127.0.0.1', 'localhost')
    r = correr(servidor_vivo, b'{"pasos": ["evaluar"]}', 'application/json; charset=utf-8', origen=origen)
    assert r.status == 202 and json.loads(r.read())['pasos'] == ['evaluar']


def test_ruta_con_byte_nulo(servidor_vivo):
    assert get(servidor_vivo + '/resultados/%00').status == 404


@pytest.mark.parametrize('ruta', ['/api/analizar', '/api/correr'])
def test_largo_negativo(servidor_vivo, ruta):
    peticion = urllib.request.Request(servidor_vivo + ruta, data=b'', method='POST',
                                      headers={'Content-Type': 'application/json', 'Content-Length': '-1'})
    assert _abrir(peticion).status == 400


def test_tiempos_corrupto(tmp_path, capsys):
    ruta = tmp_path / 'tiempos.json'
    ruta.write_text('{roto', encoding='utf-8')
    assert servidor.leer_tiempos(ruta) == {}
    assert 'tiempos.json' in capsys.readouterr().err


def test_corrida_guarda_tiempo(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'bien.py').write_text('print("listo")\n', encoding='utf-8')
    (tmp_path / 'src' / 'mal.py').write_text('raise SystemExit(3)\n', encoding='utf-8')
    tiempos = tmp_path / 'tiempos.json'
    tiempos.write_text('{"preparar": 66.0}\n', encoding='utf-8')
    monkeypatch.setattr(servidor, 'RAIZ', tmp_path)
    monkeypatch.setattr(servidor, '_ruta_tiempos', tiempos)
    for nombre in ('bien', 'mal'):
        monkeypatch.setitem(servidor.POR_ID, nombre, {'id': nombre, 'script': f'{nombre}.py', 'segundos': 60.0})
    servidor.Corrida(['bien']).ejecutar()
    servidor.Corrida(['mal']).ejecutar()
    guardado = json.loads(tiempos.read_text(encoding='utf-8'))
    assert set(guardado) == {'preparar', 'bien'} and guardado['preparar'] == 66.0
    assert 0 <= guardado['bien'] < 60 and servidor.POR_ID['bien']['segundos'] == guardado['bien']

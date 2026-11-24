"""
Servidor local de la aplicación: datos, entrenamiento, resultados y prueba de audio en una sola ventana.

Sirve src/app.html (vistas Datos, Entrenar, Resultados y Probar), los archivos de
resultados/ y el audio de las frases de prueba. La vista Entrenar corre los scripts
del proyecto en un hilo aparte y transmite su salida en vivo (Server-Sent Events);
si se recarga la página, la corrida sigue y el registro se vuelve a mostrar. La vista
Probar envía un WAV mono de 16 kHz a /api/analizar y recibe los tramos por modelo.

Solo escucha en 127.0.0.1 y solo ejecuta los scripts de la lista PASOS.

Uso:  npm run dev
      .venv/bin/python src/servidor.py        (lo mismo, sin npm)
      .venv/bin/python src/servidor.py --sin-navegador
"""

import json
import mimetypes
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import inferencia
from audio import ErrorAudio
from config import CLIPS, CRUDO, MODELOS, RAIZ, RESULTADOS, SINTETICO
from evaluar import cambios_de_referencia
from inferencia import ErrorEntrada
from modelo_profundo import ErrorModelo

APP = Path(__file__).resolve().parent / 'app.html'
FUENTES = APP.parent / 'fuentes'  # tipografías servidas en local: la app funciona sin internet
PRUEBA = SINTETICO / 'prueba'
PUERTOS = range(8000, 8011)
MAX_AUDIO = 20 * 1024 * 1024  # bytes del WAV que acepta /api/analizar
TIPOS = ('mixta', 'monolingue', 'control')
MAX_FRASES = 20

# 'segundos' es la duración medida en la corrida de referencia (resultados/tiempos.json),
# solo para que la interfaz muestre una estimación; un paso sin medición usa SEGUNDOS_POR_DEFECTO.
# 'salida': archivo (relativo a RAIZ) cuya fecha indica cuándo corrió el paso por última vez.
_ruta_tiempos = RESULTADOS / 'tiempos.json'


def leer_tiempos(ruta):
    """Duraciones medidas por paso; {} si el archivo falta o está dañado (con aviso en consola)."""
    if not ruta.exists():
        return {}
    try:
        return json.loads(ruta.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        print(f'Aviso: no se pudo leer {ruta.name} ({error}); se usan tiempos por defecto.', file=sys.stderr)
        return {}


_tiempos = leer_tiempos(_ruta_tiempos)
SEGUNDOS_POR_DEFECTO = 60.0
PASOS = [
    {'id': 'preparar', 'nombre': 'Preparar clips de Common Voice',
     'script': 'preparar.py', 'salida': 'resultados/datos.json'},
    {'id': 'sintetizar', 'nombre': 'Sintetizar el corpus de code-switching',
     'script': 'sintetizar.py', 'salida': 'datos/sintetico/prueba/meta.jsonl'},
    {'id': 'profundo', 'nombre': 'Entrenar el modelo profundo (CNN + BiLSTM)',
     'script': 'modelo_profundo.py', 'salida': 'modelos/profundo.pt'},
    {'id': 'no_supervisado', 'nombre': 'Entrenar el modelo no supervisado',
     'script': 'modelo_no_supervisado.py', 'salida': 'modelos/no_supervisado.joblib'},
    {'id': 'evaluar', 'nombre': 'Evaluar y generar gráficos',
     'script': 'evaluar.py', 'salida': 'resultados/metricas.json'},
    {'id': 'informe', 'nombre': 'Generar el informe',
     'script': 'informe.py', 'salida': 'docs/INFORME FINAL - DETECCION DE CODE-SWITCHING.docx'},
]
for _p in PASOS:
    _p['segundos'] = float(_tiempos.get(_p['id'], SEGUNDOS_POR_DEFECTO))
POR_ID = {p['id']: p for p in PASOS}


class Corrida:
    """Una ejecución de uno o varios pasos. Guarda todos sus eventos para poder reenviarlos."""

    def __init__(self, ids):
        self.ids = ids
        self.eventos = []
        self.terminada = False
        self.cancelada = False
        self.proceso = None
        self.cond = threading.Condition()

    def emitir(self, **evento):
        with self.cond:
            self.eventos.append({**evento, 't': round(time.time(), 2)})
            self.cond.notify_all()

    def ejecutar(self):
        ok = True
        self.emitir(tipo='corrida', pasos=self.ids, hora=time.strftime('%H:%M:%S'))
        for paso_id in self.ids:
            if self.cancelada:
                break
            paso = POR_ID[paso_id]
            self.emitir(tipo='inicio', paso=paso_id)
            inicio = time.monotonic()
            try:
                with subprocess.Popen(  # al salir del with se cierra la tubería de salida
                        [sys.executable, '-u', str(RAIZ / 'src' / paso['script'])],
                        cwd=RAIZ, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as self.proceso:
                    for linea in self.proceso.stdout:
                        self.emitir(tipo='linea', paso=paso_id, texto=linea.rstrip('\n'))
                    codigo = self.proceso.wait()
            except OSError as error:
                self.emitir(tipo='linea', paso=paso_id, texto=f'No se pudo ejecutar: {error}')
                codigo = -1
            segundos = round(time.monotonic() - inicio, 1)
            self.emitir(tipo='fin', paso=paso_id, codigo=codigo, segundos=segundos)
            if codigo == 0:
                guardar_tiempo(paso_id, segundos)
            else:
                ok = False
                break
        if {'profundo', 'no_supervisado'} & set(self.ids):
            inferencia._modelos = None  # que el próximo análisis cargue los modelos recién entrenados
        self.emitir(tipo='terminado', ok=ok and not self.cancelada, cancelada=self.cancelada)
        with self.cond:
            self.terminada = True
            self.cond.notify_all()

    def cancelar(self):
        self.cancelada = True
        if self.proceso and self.proceso.poll() is None:
            self.proceso.terminate()


def guardar_tiempo(paso_id, segundos):
    """Anota en resultados/tiempos.json la duración de un paso que terminó bien (conserva los demás)."""
    tiempos = leer_tiempos(_ruta_tiempos)
    tiempos[paso_id] = segundos
    _ruta_tiempos.parent.mkdir(parents=True, exist_ok=True)
    _ruta_tiempos.write_text(json.dumps(tiempos, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    POR_ID[paso_id]['segundos'] = segundos


corrida = None                  # la última corrida, terminada o en curso
cerrojo = threading.Lock()      # impide lanzar dos corridas a la vez


def estado_salidas():
    """Fecha de la última modificación de la salida de cada paso (None si no existe)."""
    estado = {}
    for paso in PASOS:
        ruta = RAIZ / paso['salida']
        estado[paso['id']] = (time.strftime('%Y-%m-%d %H:%M', time.localtime(ruta.stat().st_mtime))
                              if ruta.exists() else None)
    return estado


def leer_metas():
    """Metadata de las frases de prueba, en el orden de meta.jsonl."""
    ruta = PRUEBA / 'meta.jsonl'
    if not ruta.exists():
        raise ErrorModelo('No hay frases de prueba. Corre el paso "sintetizar".')
    return [json.loads(linea) for linea in ruta.read_text(encoding='utf-8').splitlines() if linea.strip()]


def dentro_de(carpeta, relativa):
    """Ruta de un archivo dentro de carpeta, o None si la ruta pedida sale de ella."""
    base = carpeta.resolve()
    try:
        ruta = (base / unquote(relativa)).resolve()
    except ValueError:  # p. ej. un byte nulo (%00) en la ruta
        return None
    return ruta if ruta.is_relative_to(base) else None


class Manejador(BaseHTTPRequestHandler):
    server_version = 'CodeSwitching'

    def log_message(self, formato, *args):
        pass  # la consola queda para la URL y los errores

    def host_valido(self):
        # Rechaza peticiones con otro Host (ataques de DNS rebinding desde páginas externas).
        puerto = self.server.server_address[1]
        return self.headers.get('Host') in (f'127.0.0.1:{puerto}', f'localhost:{puerto}')

    def origen_valido(self):
        # Rechaza POST enviados desde otras páginas (CSRF): el navegador manda Origin en esos casos.
        puerto = self.server.server_address[1]
        origen = self.headers.get('Origin')
        return origen is None or origen in (f'http://127.0.0.1:{puerto}', f'http://localhost:{puerto}')

    def descartar_cuerpo(self, largo):
        # Se lee el cuerpo antes de responder: si se cierra con datos sin leer, el
        # cliente recibe un reinicio de conexión en vez del mensaje.
        restante = largo
        while restante > 0 and (trozo := self.rfile.read(min(restante, 1 << 16))):
            restante -= len(trozo)

    def responder_json(self, datos, estado=HTTPStatus.OK):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode('utf-8')
        self.send_response(estado)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(cuerpo)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(cuerpo)

    def enviar_archivo(self, ruta):
        if ruta is None or not ruta.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, 'No existe; corre el paso que lo genera.')
            return
        cuerpo = ruta.read_bytes()
        tipo = mimetypes.guess_type(ruta.name)[0] or 'application/octet-stream'
        # Rango de bytes (un solo tramo): sin esto el navegador no puede adelantar un <audio>.
        estado, total = HTTPStatus.OK, len(cuerpo)
        rango = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', '').strip())
        if rango and any(rango.groups()):
            a, b = rango.groups()
            inicio, fin = (max(0, total - int(b)), total - 1) if a == '' else (int(a), min(int(b or total - 1), total - 1))
            if inicio > fin:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{total}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            cuerpo, estado = cuerpo[inicio:fin + 1], HTTPStatus.PARTIAL_CONTENT
        self.send_response(estado)
        if estado == HTTPStatus.PARTIAL_CONTENT:
            self.send_header('Content-Range', f'bytes {inicio}-{fin}/{total}')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Type', f'{tipo}; charset=utf-8' if tipo.startswith('text/') else tipo)
        self.send_header('Content-Length', str(len(cuerpo)))
        self.send_header('Cache-Control', 'no-store')  # tras entrenar, las vistas deben ver lo nuevo
        self.end_headers()
        self.wfile.write(cuerpo)

    def responder_analisis(self, ruta, extra=None):
        """Analiza el audio en ruta y responde con el JSON de inferencia o un error legible."""
        try:
            resultado = inferencia.analizar(ruta)
        except (ErrorEntrada, ErrorAudio) as error:
            self.responder_json({'error': str(error)}, HTTPStatus.BAD_REQUEST)
        except ErrorModelo as error:
            self.responder_json({'error': str(error)}, HTTPStatus.CONFLICT)
        except Exception as error:  # noqa: BLE001 — se registra en consola y se informa sin traceback
            traceback.print_exc()
            self.responder_json({'error': f'Error interno al analizar el audio: {error}'},
                                HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self.responder_json({**resultado, **(extra or {})})

    def do_GET(self):
        if not self.host_valido():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        if url.path in ('/', '/index.html'):
            self.enviar_archivo(APP)
        elif url.path.startswith('/resultados/'):
            self.enviar_archivo(dentro_de(RESULTADOS, url.path.removeprefix('/resultados/')))
        elif url.path.startswith('/fuentes/'):
            self.enviar_archivo(dentro_de(FUENTES, url.path.removeprefix('/fuentes/')))
        elif url.path.startswith('/audio/prueba/'):
            self.enviar_archivo(dentro_de(PRUEBA, url.path.removeprefix('/audio/prueba/')))
        elif url.path == '/api/estado':
            self.responder_json({
                'datos_crudos': any(CRUDO.glob('*.tar*')),
                'clips': (CLIPS / 'indice.csv').exists(),
                'corpus': (PRUEBA / 'meta.jsonl').exists(),
                'modelos': {'profundo': (MODELOS / 'profundo.pt').exists(),
                            'no_supervisado': (MODELOS / 'no_supervisado.joblib').exists()},
                'salidas': estado_salidas(),
                'pasos': [{k: p[k] for k in ('id', 'nombre', 'script', 'segundos')} for p in PASOS],
                'corriendo': bool(corrida and not corrida.terminada),
                'hay_corrida': corrida is not None,
            })
        elif url.path == '/api/frases':
            tipo = parse_qs(url.query).get('tipo', [''])[0]
            if tipo not in TIPOS:
                self.responder_json({'error': f'Tipo de frase desconocido: {tipo!r}.'}, HTTPStatus.BAD_REQUEST)
                return
            try:
                metas = leer_metas()
            except ErrorModelo as error:
                self.responder_json({'error': str(error)}, HTTPStatus.CONFLICT)
                return
            self.responder_json([{k: m[k] for k in ('id', 'tipo', 'duracion')}
                                 for m in metas if m['tipo'] == tipo][:MAX_FRASES])
        elif url.path.startswith('/api/frase/'):
            self.frase(unquote(url.path.removeprefix('/api/frase/')))
        elif url.path == '/api/eventos':
            self.transmitir(url)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def frase(self, id_frase):
        """Análisis de una frase de prueba, con la verdad tomada de su metadata."""
        try:
            metas = leer_metas()
        except ErrorModelo as error:
            self.responder_json({'error': str(error)}, HTTPStatus.CONFLICT)
            return
        meta = next((m for m in metas if m['id'] == id_frase), None)
        if meta is None:  # solo ids de meta.jsonl: nunca se arma una ruta con texto del cliente
            self.responder_json({'error': f'No existe la frase {id_frase!r}.'}, HTTPStatus.NOT_FOUND)
            return
        verdad = {'tramos': [{k: f[k] for k in ('inicio', 'fin', 'idioma')} for f in meta['fragmentos']],
                  'cambios': cambios_de_referencia(meta)}
        self.responder_analisis(PRUEBA / f'{meta["id"]}.wav', {'verdad': verdad})

    def transmitir(self, url):
        """Server-Sent Events con los eventos de la corrida, desde el índice pedido."""
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        c = corrida
        if c is None:
            self.wfile.write(b'data: {"tipo": "sin_corrida"}\n\n')
            return
        ultimo = self.headers.get('Last-Event-ID')
        desde = parse_qs(url.query).get('desde', ['0'])[0]
        indice = int(ultimo) + 1 if ultimo and ultimo.isdigit() else int(desde) if desde.isdigit() else 0
        try:
            while True:
                with c.cond:
                    if indice >= len(c.eventos) and not c.terminada:
                        c.cond.wait(timeout=15)
                    nuevos, terminada = c.eventos[indice:], c.terminada
                if not nuevos and not terminada:
                    self.wfile.write(b': sigue\n\n')  # mantiene viva la conexión
                for evento in nuevos:
                    datos = json.dumps(evento, ensure_ascii=False)
                    self.wfile.write(f'id: {indice}\ndata: {datos}\n\n'.encode('utf-8'))
                    indice += 1
                self.wfile.flush()
                if terminada and indice >= len(c.eventos):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return  # la pestaña se cerró; la corrida sigue en su hilo

    def analizar_subido(self):
        """POST /api/analizar: cuerpo audio/wav de hasta MAX_AUDIO bytes."""
        try:
            largo = int(self.headers.get('Content-Length', ''))
        except ValueError:
            self.responder_json({'error': 'Falta el tamaño del archivo (Content-Length).'},
                                HTTPStatus.LENGTH_REQUIRED)
            return
        if largo < 0:
            self.responder_json({'error': 'Tamaño del archivo inválido (Content-Length).'},
                                HTTPStatus.BAD_REQUEST)
            return
        if largo > MAX_AUDIO:
            self.descartar_cuerpo(largo)
            self.responder_json({'error': 'El archivo supera 20 MB.'}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        cuerpo = self.rfile.read(largo)
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'audio.wav'  # nombre fijo: aparece en los mensajes de error de audio
            ruta.write_bytes(cuerpo)
            self.responder_analisis(ruta)

    def do_POST(self):
        global corrida
        if not self.host_valido():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not self.origen_valido():
            valor = self.headers.get('Content-Length', '')
            self.descartar_cuerpo(int(valor) if valor.isdigit() else 0)
            self.responder_json({'error': 'Petición rechazada: viene de otra página.'}, HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        if url.path == '/api/analizar':
            self.analizar_subido()
        elif url.path == '/api/correr':
            try:
                largo = int(self.headers.get('Content-Length', 0))
                if largo < 0:
                    raise ValueError('Content-Length negativo')
                ids = json.loads(self.rfile.read(largo) or b'{}').get('pasos') or []
            except (ValueError, AttributeError):
                self.responder_json({'error': 'Cuerpo inválido.'}, HTTPStatus.BAD_REQUEST)
                return
            if self.headers.get_content_type() != 'application/json':
                self.responder_json({'error': 'El cuerpo debe ser JSON (Content-Type: application/json).'},
                                    HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(ids, list) or not ids or any(i not in POR_ID for i in ids):
                self.responder_json({'error': 'Pasos desconocidos.'}, HTTPStatus.BAD_REQUEST)
                return
            with cerrojo:
                if corrida and not corrida.terminada:
                    self.responder_json({'error': 'Ya hay un entrenamiento en curso.'}, HTTPStatus.CONFLICT)
                    return
                orden = [p['id'] for p in PASOS if p['id'] in ids]  # siempre en el orden del pipeline
                corrida = Corrida(orden)
                threading.Thread(target=corrida.ejecutar, daemon=True).start()
            self.responder_json({'ok': True, 'pasos': orden}, HTTPStatus.ACCEPTED)
        elif url.path == '/api/cancelar':
            if corrida and not corrida.terminada:
                corrida.cancelar()
            self.responder_json({'ok': True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    servidor = None
    for puerto in PUERTOS:
        try:
            servidor = ThreadingHTTPServer(('127.0.0.1', puerto), Manejador)
            break
        except OSError:
            continue
    if servidor is None:
        sys.exit(f'Los puertos {PUERTOS.start}-{PUERTOS.stop - 1} están ocupados.')
    servidor.daemon_threads = True
    url = f'http://127.0.0.1:{servidor.server_address[1]}/'
    print(f'Aplicación en {url}  (Ctrl+C para detener)', flush=True)
    if '--sin-navegador' not in sys.argv:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        if corrida and not corrida.terminada:
            corrida.cancelar()
        print('\nServidor detenido.')


if __name__ == '__main__':
    main()

"""Evaluación de los modelos sobre la partición de prueba y las grabaciones reales.

Escribe resultados/metricas.json y las figuras de resultados/figuras/.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

from audio import ErrorAudio, cargar_audio, mascara_habla, n_tramas  # noqa: E402
from config import (ES, FIGURAS, HOP, IGNORAR, MODELOS, NOMBRES, QU,  # noqa: E402
                    REALES, RESULTADOS, SEMILLA, SINTETICO, SR)
from metricas import contar_cambios, metricas_trama, prf, tasa_falsos_cambios  # noqa: E402
from modelo_no_supervisado import (ALGORITMOS, cargar_no_supervisado,  # noqa: E402
                                   grupos_de_ventanas, predecir_no_supervisado,
                                   ventanas_de_particion)
from modelo_profundo import ErrorModelo, cargar_profundo, predecir_profundo  # noqa: E402
from postproceso import puntos_cambio, suavizar, tramos  # noqa: E402

# Paleta apta para daltonismo; la app web usa los mismos colores.
COLORES = {ES: '#0072B2', QU: '#E69F00', IGNORAR: '#BBBBBB'}
_ETIQUETAS_ES = {'es', 'español', 'espanol'}
_ETIQUETAS_QU = {'qu', 'quechua'}
_CUBOS = ('<1s', '1-2s', '>=2s')


def leer_etiquetas_audacity(ruta: Path) -> list[tuple[float, float, int]]:
    ruta = Path(ruta)
    salida = []
    for n, linea in enumerate(ruta.read_text(encoding='utf-8').splitlines(), 1):
        if not linea.strip():
            continue
        partes = linea.rstrip('\r\n').split('\t')
        if len(partes) != 3:
            raise ValueError(f'Línea {n} de {ruta.name}: se esperaban 3 columnas separadas por tabulador')
        try:
            ini, fin = float(partes[0]), float(partes[1])
        except ValueError:
            raise ValueError(f'Línea {n} de {ruta.name}: inicio o fin no son números') from None
        if fin <= ini:
            raise ValueError(f'Línea {n} de {ruta.name}: el fin no es posterior al inicio')
        etiqueta = partes[2].strip().lower()
        if etiqueta in _ETIQUETAS_ES:
            idioma = ES
        elif etiqueta in _ETIQUETAS_QU:
            idioma = QU
        else:
            raise ValueError(f'Línea {n} de {ruta.name}: etiqueta desconocida "{partes[2].strip()}"')
        salida.append((ini, fin, idioma))
    return salida


def etiquetas_desde_tramos(tramos_ref, n: int, mascara: np.ndarray) -> np.ndarray:
    """Etiquetas por trama; fuera de un tramo o sin habla → IGNORAR."""
    etiquetas = np.full(n, IGNORAR, int)
    for ini, fin, idioma in tramos_ref:
        a, b = int(round(ini * SR / HOP)), int(round(fin * SR / HOP))
        etiquetas[a:min(b, n)] = idioma
    etiquetas[~np.asarray(mascara, bool)[:n]] = IGNORAR
    return etiquetas


def cambios_de_referencia(meta: dict) -> list[float]:
    frags = meta['fragmentos']
    return [(a['fin'] + b['inicio']) / 2 for a, b in zip(frags, frags[1:])
            if a['idioma'] != b['idioma']]


def _cubo(duracion: float) -> str:
    return _CUBOS[0] if duracion < 1 else _CUBOS[1] if duracion < 2 else _CUBOS[2]


class _Acumulador:
    """Suma tramas y puntos de cambio de varias frases."""

    def __init__(self):
        self.ref, self.pred = [], []
        self.vp = self.fp = self.fn = 0

    def agregar(self, ref, pred, cambios_ref, cambios_pred):
        self.ref.append(ref)
        self.pred.append(pred)
        vp, fp, fn = contar_cambios(cambios_ref, cambios_pred)
        self.vp, self.fp, self.fn = self.vp + vp, self.fp + fp, self.fn + fn

    def resultado(self) -> dict:
        ref = np.concatenate(self.ref) if self.ref else np.zeros(0, int)
        pred = np.concatenate(self.pred) if self.pred else np.zeros(0, int)
        return {'trama': metricas_trama(ref, pred), 'cambios': prf(self.vp, self.fp, self.fn)}


def evaluar_particion(dir_particion: Path, predictores: dict[str, Callable]) -> dict:
    dir_particion = Path(dir_particion)
    with open(dir_particion / 'meta.jsonl', encoding='utf-8') as f:
        metas = [json.loads(l) for l in f if l.strip()]
    glob = {n: _Acumulador() for n in predictores}
    tipo = {n: {} for n in predictores}
    sin_cambios = {n: {'monolingue': [], 'control': []} for n in predictores}  # frases sin cambio real
    cubos = {n: {c: [0, 0] for c in _CUBOS} for n in predictores}  # [aciertos, total]
    for meta in metas:
        y = cargar_audio(dir_particion / f'{meta["id"]}.wav')
        ref = np.load(dir_particion / f'{meta["id"]}.npz')['etiquetas']
        cambios_ref = cambios_de_referencia(meta)
        for nombre, predictor in predictores.items():
            pred = np.asarray(predictor(y))
            cambios_pred = puntos_cambio(tramos(pred))
            glob[nombre].agregar(ref, pred, cambios_ref, cambios_pred)
            tipo[nombre].setdefault(meta['tipo'], _Acumulador()).agregar(
                ref, pred, cambios_ref, cambios_pred)
            if meta['tipo'] in sin_cambios[nombre]:
                sin_cambios[nombre][meta['tipo']].append(cambios_pred)
            for fr in meta['fragmentos']:
                a, b = int(round(fr['inicio'] * SR / HOP)), int(round(fr['fin'] * SR / HOP))
                r, p = ref[a:b], pred[a:b]
                v = r != IGNORAR
                c = cubos[nombre][_cubo(fr['fin'] - fr['inicio'])]
                c[0] += int((r[v] == p[v]).sum())
                c[1] += int(v.sum())
    salida = {}
    for nombre in predictores:
        salida[nombre] = {
            **glob[nombre].resultado(),
            'por_tipo': {t: a.resultado() for t, a in tipo[nombre].items()},
            'por_duracion': {c: ac / tot for c, (ac, tot) in cubos[nombre].items() if tot},
            'falsos_cambios_control': tasa_falsos_cambios(sin_cambios[nombre]['control']),
            'falsos_cambios_monolingue': tasa_falsos_cambios(sin_cambios[nombre]['monolingue'])}
    return salida


def evaluar_reales(dir_reales: Path, predictores: dict[str, Callable]) -> dict | None:
    dir_reales = Path(dir_reales)
    if not dir_reales.is_dir():
        return None
    wavs = [w for w in sorted(dir_reales.glob('*.wav')) if w.with_suffix('.txt').exists()]
    if not wavs:
        return None
    acum = {n: _Acumulador() for n in predictores}
    omitidos = []
    for wav in wavs:
        try:
            y = cargar_audio(wav)
            tr = leer_etiquetas_audacity(wav.with_suffix('.txt'))
        except (ErrorAudio, ValueError) as e:
            print(f'Se omite {wav.name}: {e}', file=sys.stderr, flush=True)
            omitidos.append({'archivo': wav.name, 'motivo': str(e)})
            continue
        ref = etiquetas_desde_tramos(tr, n_tramas(len(y)), mascara_habla(y))
        cambios_ref = puntos_cambio(tramos(ref))
        for nombre, predictor in predictores.items():
            pred = np.asarray(predictor(y))
            acum[nombre].agregar(ref, pred, cambios_ref, puntos_cambio(tramos(pred)))
    return {**{n: a.resultado() for n, a in acum.items()}, 'omitidos': omitidos}


def _metricas_ventanas(ns: dict, dir_prueba: Path, silueta_k2: float) -> dict:
    v = ventanas_de_particion(dir_prueba, None, SEMILLA)
    valido = v['idioma'] != IGNORAR
    por_algoritmo = {}
    for a in ALGORITMOS:
        g, e = grupos_de_ventanas(ns, v['X'], a)[valido], v['idioma'][valido]
        pureza = sum(np.bincount(e[g == k]).max() for k in np.unique(g)) / len(e)
        por_algoritmo[a] = {'ari': float(adjusted_rand_score(e, g)),
                            'nmi': float(normalized_mutual_info_score(e, g)),
                            'pureza': float(pureza)}
    return {**por_algoritmo[ns['mejor']], 'algoritmo': ns['mejor'],
            'silueta': silueta_k2, 'por_algoritmo': por_algoritmo, 'n_ventanas': int(valido.sum())}


def _barras_tramos(ax, y0, lista, alto=0.8):
    for ini, fin, idioma in lista:
        ax.broken_barh([(ini, fin - ini)], (y0, alto), facecolors=COLORES[idioma])


def generar_figuras(metricas: dict, entrenamiento: dict, no_supervisado: dict,
                    ejemplos: list[dict], dir_figuras: Path) -> None:
    """Escribe las cinco figuras del informe en dir_figuras."""
    dir_figuras = Path(dir_figuras)
    dir_figuras.mkdir(parents=True, exist_ok=True)
    clases = [NOMBRES[ES], NOMBRES[QU]]

    ep = entrenamiento['epocas']
    x = [e['epoca'] for e in ep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(x, [e['perdida_entrenamiento'] for e in ep], label='Entrenamiento', color=COLORES[ES])
    a1.plot(x, [e['perdida_validacion'] for e in ep], label='Validación', color=COLORES[QU])
    a1.set(title='Pérdida por época', xlabel='Época', ylabel='Pérdida')
    a1.legend()
    a2.plot(x, [e['f1_validacion'] for e in ep], color=COLORES[ES])
    a2.set(title='F1 macro en validación', xlabel='Época', ylabel='F1 macro')
    fig.tight_layout()
    fig.savefig(dir_figuras / 'curvas_entrenamiento.png', dpi=120)
    plt.close(fig)

    for nombre, titulo in (('profundo', 'Modelo profundo'), ('no_supervisado', 'Modelo no supervisado')):
        m = np.array(metricas['prueba'][nombre]['trama']['matriz'])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(m, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha='center', va='center',
                        color='white' if m[i, j] > m.max() / 2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=clases, yticklabels=clases,
               xlabel='Predicho', ylabel='Real', title=f'Matriz de confusión: {titulo}')
        fig.tight_layout()
        fig.savefig(dir_figuras / f'matriz_{nombre}.png', dpi=120)
        plt.close(fig)

    sil = no_supervisado['silueta_por_k']
    ks = sorted(int(k) for k in sil)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([str(k) for k in ks], [sil[str(k)] for k in ks],
           color=COLORES[ES])
    ax.set(title='Silueta según el número de grupos', xlabel='Número de grupos k',
           ylabel='Coeficiente de silueta')
    fig.tight_layout()
    fig.savefig(dir_figuras / 'silueta_k.png', dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(len(ejemplos), 1, figsize=(10, 2.6 * len(ejemplos)), squeeze=False)
    for ax, ej in zip(axs[:, 0], ejemplos):
        for fila, (clave, nombre) in enumerate((('verdad', 'Verdad'), ('profundo', 'Profundo'),
                                                ('no_supervisado', 'No supervisado'))):
            _barras_tramos(ax, 2 - fila, ej[clave])
        ax.set(yticks=[2.4, 1.4, 0.4], yticklabels=['Verdad', 'Profundo', 'No supervisado'],
               xlim=(0, ej['duracion']), xlabel='Tiempo (s)', title=f'Frase {ej["tipo"]}')
    axs[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORES[i]) for i in (ES, QU)],
                     labels=clases, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig(dir_figuras / 'ejemplos_lineas_tiempo.png', dpi=120)
    plt.close(fig)


def _linea_base(dir_entrenamiento: Path) -> Callable:
    cuentas = np.zeros(2, int)
    for npz in Path(dir_entrenamiento).glob('*.npz'):
        e = np.load(npz)['etiquetas']
        cuentas += np.bincount(e[e != IGNORAR], minlength=2)
    clase = int(cuentas.argmax())
    return lambda y: np.where(mascara_habla(y), clase, IGNORAR)


def _ejemplos(dir_prueba: Path, predictores: dict) -> list[dict]:
    with open(dir_prueba / 'meta.jsonl', encoding='utf-8') as f:
        metas = [json.loads(l) for l in f if l.strip()]
    elegidas = {}
    for m in metas:
        elegidas.setdefault(m['tipo'], m)
    salida = []
    for t in ('mixta', 'monolingue', 'control'):
        if t not in elegidas:
            continue
        m = elegidas[t]
        y = cargar_audio(dir_prueba / f'{m["id"]}.wav')
        ref = np.load(dir_prueba / f'{m["id"]}.npz')['etiquetas']
        salida.append({'tipo': t, 'duracion': m['duracion'], 'verdad': tramos(ref),
                       **{n: tramos(predictores[n](y)) for n in ('profundo', 'no_supervisado')}})
    return salida


def _leer_json(ruta: Path, paso: str) -> dict:
    if not ruta.exists():
        raise ErrorModelo(f'Falta {ruta.name}. Corre el paso "{paso}".')
    try:
        return json.loads(ruta.read_text(encoding='utf-8'))
    except ValueError:
        raise ErrorModelo(f'{ruta.name} está dañado. Corre el paso "{paso}".') from None


def _paso(texto: str) -> None:
    print(texto, flush=True)


def main() -> None:
    t0 = time.time()
    prueba = SINTETICO / 'prueba'
    if not prueba.exists():
        print('Primero genera el corpus sintético (src/sintetizar.py).', file=sys.stderr)
        sys.exit(1)
    if not (prueba / 'meta.jsonl').exists():
        print('Primero genera el corpus sintético (src/sintetizar.py).', file=sys.stderr)
        sys.exit(1)
    try:
        profundo = cargar_profundo(MODELOS / 'profundo.pt')
        ns = cargar_no_supervisado(MODELOS / 'no_supervisado.joblib')
        entrenamiento = _leer_json(RESULTADOS / 'entrenamiento_profundo.json',
                                   'Entrenar modelo profundo')
        res_ns = _leer_json(RESULTADOS / 'no_supervisado.json', 'Entrenar modelo no supervisado')
        if '2' not in res_ns.get('silueta_por_k', {}):
            raise ErrorModelo('Los resultados del modelo no supervisado están incompletos. '
                              'Corre el paso "Entrenar modelo no supervisado".')
        silueta_k2 = res_ns['silueta_por_k']['2']
        predictores = {
            'profundo': lambda y: suavizar(predecir_profundo(profundo, y)),
            'no_supervisado': lambda y: suavizar(predecir_no_supervisado(ns, y)),
            'linea_base': _linea_base(SINTETICO / 'entrenamiento')}
    except ErrorModelo as e:
        print(e.args[0], file=sys.stderr)
        sys.exit(1)
    _paso('Evaluando la partición de prueba...')
    metricas = {'prueba': evaluar_particion(prueba, predictores)}
    _paso('Evaluando grabaciones reales...')
    metricas['reales'] = evaluar_reales(REALES, predictores)
    _paso('Calculando métricas por ventana del modelo no supervisado...')
    metricas['no_supervisado_ventanas'] = _metricas_ventanas(
        ns, prueba, silueta_k2)
    metricas['generado'] = datetime.now().isoformat(timespec='seconds')
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS / 'metricas.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    _paso('Generando figuras...')
    generar_figuras(metricas, entrenamiento, res_ns, _ejemplos(prueba, predictores), FIGURAS)
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()

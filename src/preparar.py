"""Preparación de clips de Common Voice: lectura de los tar, división por
hablante, selección, decodificación y escritura de wav 16 kHz."""
import csv
import io
import json
import random
import re
import sys
import tarfile
import tempfile
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

from audio import ErrorAudio, cargar_audio, normalizar_volumen, recortar_silencios
from config import CLIPS, CRUDO, RESULTADOS, SEMILLA, SR

IDIOMA_DE_LOCALE = {'qxp': 'qu', 'es': 'es'}
NOMBRE_IDIOMA = {'qu': 'Quechua', 'es': 'Español'}
PARTICIONES = ('entrenamiento', 'validacion', 'prueba')
TSV_PARTICIONES = ('train.tsv', 'dev.tsv', 'test.tsv')


def _es_clip(nombre: str) -> bool:
    partes = nombre.split('/')
    return 'clips' in partes[:-1]


def _leer_tsvs(ruta_tar: Path) -> list[tuple[str, list[dict]]]:
    """Primera pasada en streaming: lee solo los .tsv y se detiene al llegar a clips/."""
    tsvs = []
    with tarfile.open(ruta_tar, mode='r|*') as tar:
        for m in tar:
            if _es_clip(m.name):
                break
            if m.isfile() and m.name.endswith('.tsv'):
                texto = io.StringIO(tar.extractfile(m).read().decode('utf-8'), newline='')
                # QUOTE_NONE: las frases de Common Voice traen comillas sueltas
                filas = list(csv.DictReader(texto, delimiter='\t', quoting=csv.QUOTE_NONE))
                tsvs.append((m.name, filas))
    return tsvs


def _locale_por_estructura(nombre: str) -> str:
    partes = nombre.split('/')
    for i, p in enumerate(partes[:-1]):
        if p.startswith('cv-corpus-') and i + 1 < len(partes) - 1:
            return partes[i + 1]
    if len(partes) > 1:
        return re.split(r'[-_]', partes[0])[0]
    return ''


def locale_de_tar(ruta_tar: Path) -> str:
    """Locale de la columna `locale` del primer .tsv; si falta, de la estructura de carpetas."""
    tsvs = _leer_tsvs(ruta_tar)
    if not tsvs:
        raise ValueError(f'El archivo {Path(ruta_tar).name} no contiene ningún .tsv.')
    nombre, filas = tsvs[0]
    if filas and (filas[0].get('locale') or '').strip():
        return filas[0]['locale'].strip()
    locale = _locale_por_estructura(nombre)
    if not locale:
        raise ValueError(f'No se pudo determinar el idioma de {Path(ruta_tar).name}.')
    return locale


def _genero(valor: str) -> str:
    valor = valor or ''
    return 'M' if valor.startswith('male') else 'F' if valor.startswith('female') else ''


def leer_validados(ruta_tar: Path) -> list[dict]:
    """Filas de validated.tsv, o la unión sin duplicados de train/dev/test si no existe."""
    por_nombre = {Path(n).name: f for n, f in _leer_tsvs(ruta_tar)}
    if 'validated.tsv' in por_nombre:
        origen = por_nombre['validated.tsv']
    else:
        origen = [f for n in TSV_PARTICIONES for f in por_nombre.get(n, [])]
    vistos, salida = set(), []
    for f in origen:
        if f['path'] in vistos:
            continue
        vistos.add(f['path'])
        salida.append({'hablante': f['client_id'], 'archivo': f['path'],
                       'genero': _genero(f.get('gender'))})
    return salida


def dividir_hablantes(hablantes: list[str], semilla: int = SEMILLA) -> dict[str, str]:
    unicos = sorted(set(hablantes))
    random.Random(semilla).shuffle(unicos)
    n = len(unicos)
    n_extra = max(1, round(0.1 * n)) if n >= 3 else 0
    cortes = [n - 2 * n_extra, n - n_extra]
    return {h: PARTICIONES[(i >= cortes[0]) + (i >= cortes[1])] for i, h in enumerate(unicos)}


def seleccionar(filas: list[dict], max_por_hablante: int, horas_max: float,
                duracion_media_s: float, semilla: int = SEMILLA) -> list[dict]:
    rng = random.Random(semilla)
    por_hablante = {}
    for f in sorted(filas, key=lambda f: (f['hablante'], f['archivo'])):
        por_hablante.setdefault(f['hablante'], []).append(f)
    elegidas = []
    for h in sorted(por_hablante):
        grupo = por_hablante[h]
        elegidas += rng.sample(grupo, min(max_por_hablante, len(grupo)))
    tope = int(horas_max * 3600 / duracion_media_s)
    return rng.sample(elegidas, tope) if len(elegidas) > tope else elegidas


def _procesar(args: tuple) -> float | None:
    """Decodifica, normaliza, recorta y escribe un wav. None si se descarta."""
    origen, destino = args
    try:
        y = recortar_silencios(normalizar_volumen(cargar_audio(origen)))
    except ErrorAudio:
        return None
    if len(y) < SR:
        return None
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, y, SR, subtype='PCM_16')
    return len(y) / SR


def _extraer(ruta_tar: Path, pedidos: dict[str, dict], carpeta: Path) -> list[tuple]:
    """Segunda pasada en streaming: extrae solo los clips pedidos a `carpeta`."""
    hechos = []
    pendientes = set(pedidos)
    with tarfile.open(ruta_tar, mode='r|*') as tar:
        for m in tar:
            if not pendientes:
                break
            if not (m.isfile() and _es_clip(m.name)):
                continue
            archivo = m.name.split('/')[-1]
            if archivo not in pendientes:
                continue
            pendientes.discard(archivo)
            tmp = carpeta / f'{len(hechos)}{Path(archivo).suffix}'
            tmp.write_bytes(tar.extractfile(m).read())
            hechos.append((tmp, pedidos[archivo]))
    return hechos


def preparar(rutas_tar: list[Path], horas_max: float = 12.0,
             max_por_hablante: int = 200) -> list[dict]:
    """Prepara los clips de todos los tar; los de un mismo idioma se tratan como un conjunto."""
    por_idioma = {}
    for ruta in rutas_tar:
        locale = locale_de_tar(ruta)
        if locale not in IDIOMA_DE_LOCALE:
            raise ValueError(f'Idioma no soportado ({locale}) en {Path(ruta).name}.')
        idioma = IDIOMA_DE_LOCALE[locale]
        for f in leer_validados(ruta):
            por_idioma.setdefault(idioma, []).append({**f, 'tar': ruta})

    filas_indice = []
    for idioma, filas in sorted(por_idioma.items()):
        particion = dividir_hablantes([f['hablante'] for f in filas])
        elegidas = seleccionar(filas, max_por_hablante, horas_max, 4.0)
        contador = 0
        for ruta in dict.fromkeys(f['tar'] for f in elegidas):
            pedidos = {}
            for f in (f for f in elegidas if f['tar'] == ruta):
                meta = {**f, 'particion': particion[f['hablante']], 'n': contador}
                pedidos[f['archivo']] = meta
                contador += 1
            with tempfile.TemporaryDirectory() as tmp:
                trabajos = _extraer(ruta, pedidos, Path(tmp))
                args = []
                for origen, meta in trabajos:
                    rel = Path(idioma) / meta['particion'] / f"{meta['hablante'][:8]}_{meta['n']}.wav"
                    args.append((str(origen), str(CLIPS / rel)))
                with Pool() as pool:
                    duraciones = pool.map(_procesar, args, chunksize=16)
            for (_, meta), (_, destino), dur in zip(trabajos, args, duraciones):
                if dur is None:
                    continue
                filas_indice.append({
                    'ruta': str(Path(destino).relative_to(CLIPS)), 'idioma': idioma,
                    'hablante': meta['hablante'], 'genero': meta['genero'],
                    'particion': meta['particion'], 'duracion': round(dur, 3)})
    return filas_indice


def _resumen(filas: list[dict]) -> dict:
    res = {}
    for f in filas:
        r = res.setdefault(f['idioma'], {}).setdefault(
            f['particion'], {'clips': 0, 'horas': 0.0, 'hablantes': set()})
        r['clips'] += 1
        r['horas'] += f['duracion'] / 3600
        r['hablantes'].add(f['hablante'])
    return {i: {p: {'clips': r['clips'], 'horas': round(r['horas'], 4),
                    'hablantes': len(r['hablantes'])} for p, r in ps.items()}
            for i, ps in res.items()}


def main() -> None:
    tars = sorted(list(CRUDO.glob('*.tar.gz')) + list(CRUDO.glob('*.tar')))
    try:
        idiomas = {r: IDIOMA_DE_LOCALE[locale_de_tar(r)] for r in tars}
    except (KeyError, ValueError, tarfile.TarError) as e:
        sys.exit(f'No se pudo leer {CRUDO}: {e!r}')
    for idioma in ('qu', 'es'):
        if idioma not in idiomas.values():
            print(f'Falta el audio de {NOMBRE_IDIOMA[idioma]} en datos/crudo/. '
                  f'Archivos encontrados: {", ".join(t.name for t in tars) or "ninguno"}. '
                  'Ver README, sección Datos.', file=sys.stderr)
            sys.exit(1)
    filas = preparar(tars)
    CLIPS.mkdir(parents=True, exist_ok=True)
    columnas = ['ruta', 'idioma', 'hablante', 'genero', 'particion', 'duracion']
    with open(CLIPS / 'indice.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, columnas)
        w.writeheader()
        w.writerows(filas)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / 'datos.json').write_text(
        json.dumps(_resumen(filas), indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(filas)} clips escritos en {CLIPS}')


if __name__ == '__main__':
    main()

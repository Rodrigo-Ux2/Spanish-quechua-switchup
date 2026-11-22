from string import Template

from config import RAIZ
from informe import generar, valores


def test_valores_cubren_marcadores():
    texto = (RAIZ / 'docs/informe/contenido.md').read_text(encoding='utf-8')
    faltan = {m.group('named') or m.group('braced') for m in Template.pattern.finditer(texto)
              if m.group('named') or m.group('braced')} - set(valores())
    assert not faltan, faltan


def test_generar_docx(tmp_path):
    salida = generar(tmp_path / 'informe.docx')
    assert salida.exists() and salida.stat().st_size > 50_000


def test_valores_derivados_de_metricas():
    v = valores()
    # control 44.4 % y monolingüe 22.2 % en metricas.json; la matriz tiene más QU→ES que ES→QU
    assert v['falsos_mono_profundo'] == '22.2 %'
    assert v['efecto_hablante'] == '22.2 puntos'
    assert v['error_dominante'] == 'quechua→español'

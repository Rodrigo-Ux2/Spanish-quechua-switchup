from string import Template

from config import RAIZ
from informe import generar, valores


def test_valores_cubren_marcadores():
    texto = (RAIZ / 'docs/informe/contenido.md').read_text()
    faltan = {m.group('named') or m.group('braced') for m in Template.pattern.finditer(texto)
              if m.group('named') or m.group('braced')} - set(valores())
    assert not faltan, faltan


def test_generar_docx(tmp_path):
    salida = generar(tmp_path / 'informe.docx')
    assert salida.exists() and salida.stat().st_size > 50_000

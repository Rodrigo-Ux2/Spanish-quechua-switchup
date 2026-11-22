import numpy as np

from audio import mascara_habla, n_tramas
from caracteristicas import logmel, mfcc_ventanas


def test_logmel_forma(tono):
    y = tono(2.0)
    x = logmel(y)
    assert x.shape == (n_tramas(len(y)), 64) and x.dtype == np.float32


def test_mfcc_ventanas(tono, silencio):
    y = np.concatenate([tono(2.0), silencio(1.0)])
    X, c = mfcc_ventanas(y, mascara_habla(y))
    assert X.shape[1] == 80 and len(X) == len(c) and c.max() < 200 + 25
    X0, c0 = mfcc_ventanas(silencio(1.0), mascara_habla(silencio(1.0)))
    assert X0.shape == (0, 80)

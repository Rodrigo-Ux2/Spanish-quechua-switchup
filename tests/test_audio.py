import numpy as np, soundfile as sf, pytest
from audio import n_tramas, cargar_audio, normalizar_volumen, mascara_habla, pausas, ErrorAudio
from config import SR

def test_n_tramas():
    assert n_tramas(16000) == 101
    assert n_tramas(0) == 1

def test_cargar_audio_estereo_44k(tmp_path):
    t = np.arange(44100) / 44100  # 1 s a 44,1 kHz (la fixture tono es de 16 kHz)
    mono = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = np.stack([mono, mono], axis=1)
    sf.write(tmp_path / 'a.wav', y, 44100)
    x = cargar_audio(tmp_path / 'a.wav')
    assert x.ndim == 1 and x.dtype == np.float32
    assert abs(len(x) - SR) < 200  # ~1 s remuestreado a 16 kHz

def test_cargar_audio_invalido(tmp_path):
    (tmp_path / 'malo.wav').write_bytes(b'no es audio')
    with pytest.raises(ErrorAudio):
        cargar_audio(tmp_path / 'malo.wav')

def test_normalizar_volumen(tono):
    y = normalizar_volumen(0.01 * tono(1.0))
    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)))
    assert abs(rms_db - (-20.0)) < 0.5
    assert np.all(normalizar_volumen(np.zeros(1600, np.float32)) == 0)

def test_mascara_y_pausas(tono, silencio):
    y = np.concatenate([tono(1.0), silencio(0.5), tono(1.0)])
    m = mascara_habla(y)
    assert len(m) == n_tramas(len(y))
    assert m[50] and not m[125] and m[200]
    p = pausas(y)
    assert len(p) == 1 and abs(p[0] - int(1.25 * SR)) < 0.05 * SR
    assert not mascara_habla(np.zeros(SR, np.float32)).any()

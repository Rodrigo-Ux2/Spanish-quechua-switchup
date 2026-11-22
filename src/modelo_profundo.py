"""Modelo profundo CNN + BiLSTM: identificación de idioma por trama (ES / QU)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio import mascara_habla, n_tramas
from caracteristicas import logmel
from config import IGNORAR, MODELOS, N_MELS, RESULTADOS, SEMILLA, SINTETICO
from metricas import metricas_trama

REDUCCION = 4  # reducción temporal total de la CNN


class ErrorModelo(FileNotFoundError):
    """Modelo sin entrenar o checkpoint ausente."""


class CRNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, canales: tuple = (32, 64, 64), oculto: int = 128):
        super().__init__()
        entradas = (1, *canales[:-1])
        # los dos primeros bloques reducen tiempo y frecuencia; el tercero solo frecuencia
        pools = ((2, 2), (2, 2), (1, 2))
        self.cnn = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                          nn.MaxPool2d(p))
            for i, o, p in zip(entradas, canales, pools)])
        self.lstm = nn.LSTM(canales[-1] * (n_mels // 8), oculto, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.salida = nn.Linear(2 * oculto, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.cnn(x.unsqueeze(1))                      # (B, C, T/4, F/8)
        h = h.permute(0, 2, 1, 3).flatten(2)              # (B, T/4, C*F/8)
        h, _ = self.lstm(h)
        h = self.salida(h).repeat_interleave(REDUCCION, dim=1)
        if h.shape[1] >= t:
            return h[:, :t]
        relleno = h[:, -1:].expand(-1, t - h.shape[1], -1)
        return torch.cat([h, relleno], dim=1)


class FrasesDataset(Dataset):
    def __init__(self, dir_sintetico: Path, particion: str, aumentar: bool = False,
                 semilla: int = SEMILLA):
        self.rutas = sorted((Path(dir_sintetico) / particion).glob('*.npz'))
        if not self.rutas:
            raise ErrorModelo(f'No hay frases en {Path(dir_sintetico) / particion}. '
                              'Primero genera el corpus sintético (src/sintetizar.py).')
        self.aumentar = aumentar
        self.rng = np.random.default_rng(semilla)

    def __len__(self) -> int:
        return len(self.rutas)

    def _specaugment(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for _ in range(2):
            ancho = int(self.rng.integers(0, 21))
            ini = int(self.rng.integers(0, max(1, len(x) - ancho + 1)))
            x[ini:ini + ancho] = 0.0
        for _ in range(2):
            ancho = int(self.rng.integers(0, 9))
            ini = int(self.rng.integers(0, x.shape[1] - ancho + 1))
            x[:, ini:ini + ancho] = 0.0
        return x

    def __getitem__(self, i: int):
        with np.load(self.rutas[i]) as d:
            x, et = d['logmel'], d['etiquetas']
        if self.aumentar:
            x = self._specaugment(x)
        return torch.from_numpy(x), torch.from_numpy(et.astype(np.int64))


def rellenar(lote):
    """Collate: ceros en el logmel e IGNORAR en las etiquetas."""
    x = nn.utils.rnn.pad_sequence([a for a, _ in lote], batch_first=True)
    y = nn.utils.rnn.pad_sequence([b for _, b in lote], batch_first=True,
                                  padding_value=IGNORAR)
    return x, y


def _evaluar(modelo: CRNN, cargador: DataLoader, criterio: nn.Module) -> tuple[float, float]:
    """(pérdida media, F1 macro por trama) en validación."""
    modelo.eval()
    perdida, n, refs, preds = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in cargador:
            s = modelo(x)
            validas = int((y != IGNORAR).sum())
            if validas:
                perdida += criterio(s.reshape(-1, 2), y.reshape(-1)).item() * validas
                n += validas
            refs.append(y.reshape(-1).numpy())
            preds.append(s.argmax(-1).reshape(-1).numpy())
    f1 = metricas_trama(np.concatenate(refs), np.concatenate(preds))['f1_macro']
    return perdida / max(n, 1), f1


def entrenar(dir_sintetico: Path = SINTETICO, epocas: int = 30, paciencia: int = 5,
             lote: int = 32, lr: float = 1e-3, semilla: int = SEMILLA) -> dict:
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    entrenamiento = DataLoader(FrasesDataset(dir_sintetico, 'entrenamiento', True, semilla),
                               batch_size=lote, shuffle=True, collate_fn=rellenar,
                               generator=gen)
    validacion = DataLoader(FrasesDataset(dir_sintetico, 'validacion'), batch_size=lote,
                            collate_fn=rellenar)
    config = {'n_mels': N_MELS, 'canales': [32, 64, 64], 'oculto': 128}
    modelo = CRNN(config['n_mels'], tuple(config['canales']), config['oculto'])
    parametros = sum(p.numel() for p in modelo.parameters())
    criterio = nn.CrossEntropyLoss(ignore_index=IGNORAR)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    historial, mejor_f1, mejor_epoca, mejor_estado = [], -1.0, 0, None
    for epoca in range(1, epocas + 1):
        t0 = time.time()
        modelo.train()
        total, n = 0.0, 0
        for x, y in entrenamiento:
            validas = int((y != IGNORAR).sum())
            if validas == 0:
                continue
            opt.zero_grad()
            perdida = criterio(modelo(x).reshape(-1, 2), y.reshape(-1))
            perdida.backward()
            opt.step()
            total += perdida.item() * validas
            n += validas
        p_val, f1 = _evaluar(modelo, validacion, criterio)
        fila = {'epoca': epoca, 'perdida_entrenamiento': total / max(n, 1),
                'perdida_validacion': p_val, 'f1_validacion': f1,
                'segundos': time.time() - t0}
        historial.append(fila)
        print(f"época {epoca}: pérdida {fila['perdida_entrenamiento']:.4f}, "
              f"pérdida val. {p_val:.4f}, F1 val. {f1:.4f}, {fila['segundos']:.1f} s",
              flush=True)
        if f1 > mejor_f1:
            mejor_f1, mejor_epoca = f1, epoca
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        elif epoca - mejor_epoca >= paciencia:
            print(f'Parada temprana: sin mejora desde la época {mejor_epoca}.', flush=True)
            break
    MODELOS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    torch.save({'estado': mejor_estado, 'config': config}, MODELOS / 'profundo.pt')
    resultado = {'epocas': historial, 'mejor_epoca': mejor_epoca, 'parametros': parametros}
    with open(RESULTADOS / 'entrenamiento_profundo.json', 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    return resultado


def cargar_profundo(ruta: Path = MODELOS / 'profundo.pt') -> CRNN:
    if not Path(ruta).exists():
        raise ErrorModelo('El modelo profundo no está entrenado. '
                          'Corre el paso "Entrenar modelo profundo".')
    ck = torch.load(ruta, map_location='cpu', weights_only=True)
    c = ck['config']
    modelo = CRNN(c['n_mels'], tuple(c['canales']), c['oculto'])
    modelo.load_state_dict(ck['estado'])
    return modelo.eval()


def predecir_profundo(modelo: CRNN, y: np.ndarray) -> np.ndarray:
    """Idioma por trama (argmax), IGNORAR donde no hay habla; sin suavizar."""
    modelo.eval()
    with torch.no_grad():
        x = torch.from_numpy(logmel(y)).unsqueeze(0)
        pred = modelo(x)[0].argmax(-1).numpy().astype(int)
    pred[~mascara_habla(y)] = IGNORAR
    assert len(pred) == n_tramas(len(y))
    return pred


def main() -> None:
    p = argparse.ArgumentParser(description='Entrena el modelo profundo CNN + BiLSTM.')
    p.add_argument('--epocas', type=int, default=30)
    a = p.parse_args()
    if not (SINTETICO / 'entrenamiento').exists():
        print('Primero genera el corpus sintético (src/sintetizar.py).', file=sys.stderr)
        sys.exit(1)
    t0 = time.time()
    entrenar(epocas=a.epocas)
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()

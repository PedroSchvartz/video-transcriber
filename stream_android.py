"""Adaptador AO VIVO: tela do Android (via ponte_android) -> OnlineSelector.

Camada FINA (ver planning/modo_online.md): captura a tela do celular com
timestamp-PC e alimenta a engine incremental do stream_selector; salva os frames
que "assentaram" (mudanca deduplicada) numa pasta. E o produtor de (t, frame) que
faltava para o modo online receber uma tela real — o motor e a config ja existem.

Uso:
    python stream_android.py [--serial SER] [--seconds N] [--fps F]
                             [--out DIR] [--drive]

--drive: dirige o proprio aparelho com uma sequencia de teclas (home/recent) em
intervalos, para gerar mudancas de tela garantidas sem precisar tocar no celular.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import stream_selector as ss

# ponte_android e tool irma no hub; entra por path (nao e pacote instalado).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "ponte_android"))
from ponte import Ponte  # noqa: E402


def _acoes_drive(seconds: float) -> list[tuple[float, str]]:
    """Agenda de teclas (elapsed_s, tecla) p/ forcar mudancas de tela no teste."""
    seq = ["home", "recent", "home", "recent", "back", "home"]
    passo = max(2.0, seconds / (len(seq) + 1))
    return [((i + 1) * passo, k) for i, k in enumerate(seq)]


def capturar(ponte: Ponte, seconds: float, fps: float, drive: bool):
    """Gera (t_pc, frame_ndarray) por `seconds`, opcionalmente dirigindo o aparelho."""
    period = 1.0 / fps if fps > 0 else 0.0
    agenda = _acoes_drive(seconds) if drive else []
    t0 = time.time()
    t_end = t0 + seconds
    proximo = t0
    while time.time() < t_end:
        agora = time.time()
        if agora < proximo:
            time.sleep(proximo - agora)
        # dispara acoes agendadas ja vencidas (mesma conexao u2, sequencial)
        el = time.time() - t0
        while agenda and el >= agenda[0][0]:
            _, tecla = agenda.pop(0)
            try:
                ponte.key(tecla)
            except Exception as e:  # nao derruba a captura por uma acao
                print(f"  (drive: falha em '{tecla}': {e})")
        img = ponte.grab()          # PIL Image
        t = time.time()
        yield t, np.asarray(img)
        proximo = t + period


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="serial do aparelho (default: unico conectado)")
    ap.add_argument("--seconds", type=float, default=25.0, help="duracao da captura")
    ap.add_argument("--fps", type=float, default=4.0, help="taxa alvo de captura")
    ap.add_argument("--out", default=None, help="pasta de saida dos frames emitidos")
    ap.add_argument("--drive", action="store_true", help="dirige o aparelho (home/recent) p/ gerar mudancas")
    args = ap.parse_args()

    out = Path(args.out) if args.out else _HERE / "_cache" / f"stream_android_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)

    ponte = Ponte(args.serial)
    print(f"conectado: {ponte.serial}")
    print(f"capturando {args.seconds:.0f}s @ ~{args.fps} fps"
          f"{' (dirigindo o aparelho)' if args.drive else ''} -> {out}")

    sel = ss.OnlineSelector()
    n_frames = n_emit = 0
    t0 = time.time()

    def _salvar(ev, sufixo=""):
        nonlocal n_emit
        n_emit += 1
        nome = f"{n_emit:03d}_{ev.t - t0:06.2f}s_dirty{ev.dirty:.2f}{sufixo}.jpg"
        Image.fromarray(ss._as_rgb_uint8(ev.frame)).save(out / nome, quality=85)
        print(f"  [emit {n_emit}] t=+{ev.t - t0:5.2f}s dirty={ev.dirty:.2f} -> {nome}")

    for t, frame in capturar(ponte, args.seconds, args.fps, args.drive):
        n_frames += 1
        for ev in sel.feed(t, frame):
            _salvar(ev)
    for ev in sel.flush():
        _salvar(ev, sufixo="_flush")

    dt = time.time() - t0
    fps_real = n_frames / dt if dt else 0.0
    print(f"\n{n_frames} frames em {dt:.1f}s (~{fps_real:.1f} fps efetivo); "
          f"{n_emit} telas emitidas")
    print(f"saida: {out}")


if __name__ == "__main__":
    main()

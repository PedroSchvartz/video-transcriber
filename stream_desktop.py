"""Assiste a TELA DO PC ao vivo (visual + narracao) e monta uma timeline crescente.

Irmao do `stream_android.py`, mas a fonte e a tela do proprio computador. Junta
tres subsistemas rodando ao vivo, cada um numa thread:

  - VISUAL: captura a tela (PIL.ImageGrab) e alimenta o `OnlineSelector` do
    `stream_selector` — que ja e AUTO-CALIBRANTE para tela (perfil TELA do config).
    Emite so os screenshots que "assentaram" (mudanca de UI deduplicada).
  - AUDIO: grava mic e/ou audio do sistema (loopback WASAPI via `soundcard`),
    escolhido em runtime (--audio), em pedacos de `--chunk` segundos.
  - NARRACAO: cada pedaco de audio e transcrito ao vivo com o Whisper que ja existe
    (`processor.transcribe`, Groq -> faster-whisper) e os trechos entram na timeline.

Tudo compartilha o RELOGIO-PC (time.time(), t0 no inicio): prints e fala sao
intercalados por tempo. A timeline e reescrita a cada novidade em `timeline.md`.

Entrega em estagios verificaveis via flags:
    --audio none                  -> so visual (estagio 1)
    --audio both --no-transcribe  -> captura audio sem transcrever (estagio 2)
    --audio both                  -> completo (estagio 3)

Uso:
    python stream_desktop.py [--audio none|mic|system|both] [--seconds N]
                             [--fps F] [--chunk S] [--out DIR] [--no-transcribe]
Sem --seconds roda ate Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
import warnings
import wave
from pathlib import Path

# Loopback WASAPI avisa 'data discontinuity' a cada bloco de SILENCIO (sistema sem
# som): e ruido cosmetico, nao fala perdida. Silencia so essa mensagem.
warnings.filterwarnings("ignore", message="data discontinuity in recording")

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageGrab

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")   # GROQ_API_KEY para a transcricao (senao cai no local)

import processor            # noqa: E402  (transcribe reusa o motor Whisper existente)
import stream_selector as ss  # noqa: E402

SAMPLERATE = 16000          # taxa amiga do Whisper (mono)


# ===========================================================================
# Audio
# ===========================================================================
def _open_sources(mode: str):
    """Devolve [(rotulo, microphone)] conforme --audio. Loopback = audio do sistema."""
    import soundcard as sc
    srcs = []
    if mode in ("mic", "both"):
        srcs.append(("mic", sc.default_microphone()))
    if mode in ("system", "both"):
        srcs.append(("sistema", sc.get_microphone(
            sc.default_speaker().name, include_loopback=True)))
    return srcs


class _Recorder(threading.Thread):
    """Grava UMA fonte em pedacos de `chunk_s`; enfileira (t_inicio_pc, samples).

    Le em blocos PEQUENOS e CONTINUOS acumulando ate fechar um pedaco: manter o
    recorder sempre drenado evita o 'data discontinuity' do soundcard (que seria
    audio perdido nas bordas — trechos comidos na transcricao).
    """
    BLOCK_S = 0.25   # bloco de leitura; pequeno o bastante p/ nao estourar o buffer

    def __init__(self, label, mic, chunk_s, out_q, stop):
        super().__init__(daemon=True, name=f"rec-{label}")
        self.mic, self.out_q, self.stop = mic, out_q, stop
        self.block_n = max(1, int(SAMPLERATE * self.BLOCK_S))
        self.chunk_n = int(SAMPLERATE * chunk_s)

    def run(self):
        try:
            with self.mic.recorder(samplerate=SAMPLERATE, channels=1) as rec:
                buf, filled, t_start = [], 0, None
                while not self.stop.is_set():
                    data = rec.record(self.block_n)[:, 0].astype(np.float32)
                    if t_start is None:   # inicio do pedaco: t do 1o bloco (menos sua duracao)
                        t_start = time.time() - len(data) / SAMPLERATE
                    buf.append(data)
                    filled += len(data)
                    if filled >= self.chunk_n:
                        self.out_q.put((t_start, np.concatenate(buf)))
                        buf, filled, t_start = [], 0, None
                if buf:                   # pedaco final parcial ao encerrar
                    self.out_q.put((t_start, np.concatenate(buf)))
        except Exception as exc:   # nao derruba a sessao inteira por uma fonte
            print(f"  (audio '{self.name}' parou: {exc})", file=sys.stderr)


def _check_audio(mode: str, seconds: float = 3.0) -> int:
    """Diagnostico: grava `seconds` de cada fonte e reporta RMS/peak (PASS/FAIL).

    Isola a camada de maior risco — o Windows entregar audio ao processo. Peak ~0
    = sinal nao chega (mic mudo, privacidade de microfone, ou processo sem sessao
    de audio). Rode no SEU terminal, com som tocando e falando algo.
    """
    srcs = _open_sources(mode)
    if not srcs:
        print("--audio none: nada a checar.")
        return 0
    block_n = max(1, int(SAMPLERATE * 0.25))
    ok = True
    for label, mic in srcs:
        buf = []
        with mic.recorder(samplerate=SAMPLERATE, channels=1) as rec:
            t_end = time.time() + seconds
            while time.time() < t_end:
                buf.append(rec.record(block_n)[:, 0].astype(np.float32))
        a = np.concatenate(buf) if buf else np.zeros(1, np.float32)
        rms, peak = float(np.sqrt((a * a).mean())), float(np.abs(a).max())
        passed = peak > 1e-4
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FALHA'}  {label:8s} rms={rms:.5f} peak={peak:.4f}"
              f"{'' if passed else '  <- sinal ~zero (mic mudo/privacidade/sem sessao de audio)'}")
    print("resultado:", "OK — audio chega ao processo" if ok
          else "FALHA — ver privacidade de microfone do Windows / device ativo")
    return 0 if ok else 1


def _mix(chunks: list[np.ndarray]) -> np.ndarray:
    """Mistura N fontes: trunca ao menor comprimento e faz a MEDIA (evita clip)."""
    n = min(len(c) for c in chunks)
    return np.mean([c[:n] for c in chunks], axis=0).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLERATE)
        w.writeframes(pcm.tobytes())


class _Chunker(threading.Thread):
    """Junta o pedaco-k de cada fonte, mistura, grava o WAV e enfileira p/ transcrever."""

    def __init__(self, source_qs, work: Path, chunk_out, stop, transcribe: bool):
        super().__init__(daemon=True, name="chunker")
        self.source_qs, self.work = source_qs, work
        self.chunk_out, self.stop, self.transcribe = chunk_out, stop, transcribe
        self.k = 0

    def run(self):
        while not (self.stop.is_set() and all(q.empty() for q in self.source_qs)):
            try:
                items = [q.get(timeout=0.5) for q in self.source_qs]
            except queue.Empty:
                continue
            t_start = min(t for t, _ in items)
            mixed = _mix([s for _, s in items]) if len(items) > 1 else items[0][1]
            self.k += 1
            wav = self.work / f"chunk_{self.k:04d}.wav"
            _write_wav(wav, mixed)
            self.chunk_out.put((t_start, wav))


class _Transcriber(threading.Thread):
    """Transcreve cada WAV com o motor existente e devolve trechos com t absoluto."""

    def __init__(self, chunk_in, work: Path, t0, timeline, stop):
        super().__init__(daemon=True, name="transcriber")
        self.chunk_in, self.work, self.t0 = chunk_in, work, t0
        self.timeline, self.stop = timeline, stop

    def run(self):
        while not (self.stop.is_set() and self.chunk_in.empty()):
            try:
                t_start, wav = self.chunk_in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                tr = processor.transcribe(str(wav), str(self.work))
            except Exception as exc:
                print(f"  (transcricao falhou: {exc})", file=sys.stderr)
                continue
            off = t_start - self.t0
            for seg in tr.get("segments", []):
                txt = (seg.get("text") or "").strip()
                if txt:
                    self.timeline.add("fala", off + float(seg.get("start", 0)), text=txt)


# ===========================================================================
# Timeline compartilhada (fala + prints intercalados por tempo)
# ===========================================================================
class Timeline:
    def __init__(self, path: Path):
        self.path, self._items, self._lock = path, [], threading.Lock()

    def add(self, kind: str, t: float, **payload) -> None:
        with self._lock:
            self._items.append({"kind": kind, "t": max(0.0, t), **payload})
            self._render()

    def _render(self) -> None:
        lines = ["# Timeline ao vivo da tela\n"]
        for it in sorted(self._items, key=lambda x: x["t"]):
            ts = processor.fmt_time(it["t"])
            if it["kind"] == "fala":
                lines.append(f"[{ts}] {it['text']}")
            else:
                lines.append(f"[{ts}] PRINT {it['file']} (dirty={it['dirty']:.2f})")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# Loop principal (visual na thread principal; audio/transcricao em background)
# ===========================================================================
def _capture_screen(sel, timeline, out: Path, t0, seconds, fps, stop):
    period = 1.0 / fps if fps > 0 else 0.0
    n_emit = 0

    def _save(ev):
        nonlocal n_emit
        n_emit += 1
        name = f"{n_emit:03d}_{ev.t - t0:07.2f}s.jpg"
        Image.fromarray(ss._as_rgb_uint8(ev.frame)).save(out / name, quality=85)
        timeline.add("print", ev.t - t0, file=name, dirty=ev.dirty)
        print(f"  [print {n_emit}] t=+{ev.t - t0:6.2f}s dirty={ev.dirty:.2f} -> {name}")

    t_end = (t0 + seconds) if seconds else None
    nxt = t0
    try:
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            now = time.time()
            if now < nxt:
                time.sleep(nxt - now)
            frame = np.asarray(ImageGrab.grab())     # RGB HxWx3
            t = time.time()
            for ev in sel.feed(t, frame):
                _save(ev)
            nxt = t + period
    except KeyboardInterrupt:
        print("\n  (Ctrl+C — encerrando)")
    finally:
        stop.set()
        for ev in sel.flush():
            _save(ev)
    return n_emit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", choices=["none", "mic", "system", "both"], default="both",
                    help="fonte de audio a transcrever (default: both)")
    ap.add_argument("--seconds", type=float, default=0.0, help="duracao; 0 = ate Ctrl+C")
    ap.add_argument("--fps", type=float, default=2.0, help="taxa de captura da tela")
    ap.add_argument("--chunk", type=float, default=12.0, help="tamanho (s) do pedaco de audio")
    ap.add_argument("--out", default=None, help="pasta de saida (frames + timeline.md)")
    ap.add_argument("--no-transcribe", action="store_true",
                    help="grava o audio mas nao transcreve (estagio 2)")
    ap.add_argument("--check-audio", action="store_true",
                    help="so diagnostica se o audio (--audio) chega ao processo, e sai")
    args = ap.parse_args()

    if args.check_audio:
        print(f"checando audio '{args.audio}' (fale algo / deixe um video tocando)...")
        sys.exit(_check_audio(args.audio if args.audio != "none" else "both"))

    out = Path(args.out) if args.out else _HERE / "_cache" / f"stream_desktop_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="deskaud_"))
    t0 = time.time()
    stop = threading.Event()
    timeline = Timeline(out / "timeline.md")

    print(f"assistindo a tela @ ~{args.fps} fps"
          f"{f' por {args.seconds:.0f}s' if args.seconds else ' (Ctrl+C p/ parar)'}; "
          f"audio={args.audio}"
          f"{'' if (args.audio != 'none' and not args.no_transcribe) else ' (sem transcricao)'}"
          f" -> {out}")

    threads: list[threading.Thread] = []
    if args.audio != "none":
        sources = _open_sources(args.audio)
        source_qs = [queue.Queue() for _ in sources]
        for (label, mic), q in zip(sources, source_qs):
            threads.append(_Recorder(label, mic, args.chunk, q, stop))
        chunk_q: queue.Queue = queue.Queue()
        threads.append(_Chunker(source_qs, work, chunk_q, stop, not args.no_transcribe))
        if not args.no_transcribe:
            threads.append(_Transcriber(chunk_q, work, t0, timeline, stop))
        else:
            # drena os WAVs num sub-thread p/ nao encher a fila (estagio 2)
            def _drain():
                while not (stop.is_set() and chunk_q.empty()):
                    try:
                        chunk_q.get(timeout=0.5)
                    except queue.Empty:
                        pass
            threads.append(threading.Thread(target=_drain, daemon=True, name="drain"))
        for th in threads:
            th.start()

    sel = ss.OnlineSelector()
    n_emit = _capture_screen(sel, timeline, out, t0, args.seconds, args.fps, stop)

    # Encerramento: sinaliza e da um tempo p/ o audio/transcricao drenarem a fila.
    stop.set()
    for th in threads:
        th.join(timeout=args.chunk + 30)

    dt = time.time() - t0
    print(f"\nsessao de {dt:.1f}s; {n_emit} telas emitidas")
    print(f"timeline: {out / 'timeline.md'}")


if __name__ == "__main__":
    main()

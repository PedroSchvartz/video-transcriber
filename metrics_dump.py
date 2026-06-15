"""
metrics_dump.py - batch de metricas sobre uma colecao de videos.

Para cada video extrai uma linha de metricas e grava tudo num CSV (1 linha por
video), para tirar conclusoes baseadas em dados sobre uma colecao inteira.

Uso:
    python metrics_dump.py <pasta|lista.txt|video...> [-o metrics.csv]

  <pasta>      -> varre videos (mp4/mkv/webm/mov/avi/m4v) recursivamente
  <lista.txt>  -> um caminho/URL por linha (# = comentario)
  <video...>   -> um ou mais caminhos locais ou URLs (YouTube/Instagram via yt-dlp)

Fase A (sem chave/modelo): container (ffprobe) + audio (volume/silencio) +
assinatura de mudanca por blocos + brilho + proxy de densidade de texto (bordas)
+ proxy de movimento global (correlacao de fase). Transcricao (Groq Whisper) e
visao (1 chamada/video) entram como Fase B. Grava o CSV de forma incremental.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

import ffmpeg_tools

# --- parametros do scan (iguais ao _prescan, ja validados) ------------------
FPS = 2                            # amostras por segundo
W, H = 192, 108                    # resolucao do scan (16:9)
BS = 12                            # lado do bloco -> grade 16x9 = 144 blocos
GX, GY = W // BS, H // BS
T_COR = 40                         # limiar de distancia de cor por pixel (0-255)
BLOCK_HOT = 0.15                   # bloco "mudou" se >15% dos seus pixels mudaram
LOW_BAND = (0.005, 0.05)           # faixa baixa-mas-nao-zero (assinatura)
REST_MAX = 0.002                   # frame "em repouso" se muda < 0.2% (discriminador)
STATIC_MAX = 0.02                  # bloco "estatico" se ativo em < 2% dos frames
MASS_KEEP = 0.80                   # retangulo de foco = blocos com 80% da atividade
HIST_EDGES = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.01]
HIST_COLS = ["hist_lt02", "hist_02_05", "hist_05_1", "hist_1_2",
             "hist_2_5", "hist_5_10", "hist_10_20", "hist_gt20"]

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "www."))


def probe_container(path: str) -> dict:
    """Metricas de container via ffprobe (duracao, resolucao, fps, codec, bitrate)."""
    ffmpeg_tools.ensure_on_path()
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=120).stdout
    try:
        info = json.loads(out)
    except Exception:
        return {}
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def _fps(rate: str) -> float:
        try:
            num, den = rate.split("/")
            return float(num) / float(den) if float(den) else 0.0
        except Exception:
            return 0.0

    dur = float(fmt.get("duration", 0) or v.get("duration", 0) or 0)
    return {
        "duration_s": round(dur, 2),
        "width": int(v.get("width", 0) or 0),
        "height": int(v.get("height", 0) or 0),
        "fps": round(_fps(v.get("r_frame_rate", "0/1")), 3),
        "vcodec": v.get("codec_name", ""),
        "bitrate_kbps": round(float(fmt.get("bit_rate", 0) or 0) / 1000, 1),
        "has_audio": a is not None,
        "container": fmt.get("format_name", ""),
    }


def _grep_float(text: str, pattern: str):
    m = re.search(pattern, text)
    return round(float(m.group(1)), 1) if m else None


def audio_metrics(path: str, duration: float) -> dict:
    """Volume medio/maximo (volumedetect) + fracao de silencio (silencedetect),
    numa unica passada de ffmpeg. So roda se o video tiver trilha de audio."""
    ffmpeg_tools.ensure_on_path()
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
           "-af", "volumedetect,silencedetect=noise=-35dB:d=0.4",
           "-f", "null", "-"]
    err = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=300).stderr.decode("utf-8", "ignore")
    silence = sum(float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err))
    return {
        "mean_volume_db": _grep_float(err, r"mean_volume:\s*(-?[\d.]+) dB"),
        "max_volume_db": _grep_float(err, r"max_volume:\s*(-?[\d.]+) dB"),
        "silence_frac": round(silence / duration, 3) if duration else 0.0,
    }


def read_frames(path: str):
    """Gera frames RGB (H,W,3) int16 amostrados a FPS via ffmpeg pipe."""
    ffmpeg_tools.ensure_on_path()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsize = W * H * 3
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).astype(np.int16)
    proc.stdout.close()
    proc.wait()


def block_view(changed: np.ndarray) -> np.ndarray:
    """Mapa booleano (H,W) -> fracao de pixels mudados por bloco (GY,GX)."""
    g = changed[: GY * BS, : GX * BS].reshape(GY, BS, GX, BS)
    return g.mean(axis=(1, 3))


def _phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Deslocamento global (px) entre dois frames via correlacao de fase.
    Proxy de movimento de camera: ~0 em tela digital estatica, >0 em pan/camera na mao."""
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-8
    corr = np.fft.irfft2(r, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return float((dx * dx + dy * dy) ** 0.5)


def change_signature(path: str) -> dict:
    """Passada unica: assinatura de mudanca por blocos + brilho de carona."""
    prev = None
    prev_luma = None
    d_series: list[float] = []        # d(t): fracao global de pixels mudados
    nblocks_series: list[int] = []    # quantos blocos mudaram em cada t
    bright_series: list[float] = []   # luma media por frame (de carona)
    edge_series: list[float] = []     # energia de bordas (proxy de densidade de texto)
    motion_series: list[float] = []   # deslocamento global (proxy de pan/camera)
    block_hits = np.zeros((GY, GX), dtype=np.int64)
    n = 0
    for f in read_frames(path):
        luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        bright_series.append(float(luma.mean()))
        edge_series.append(float(np.abs(np.diff(luma, axis=1)).mean()
                                 + np.abs(np.diff(luma, axis=0)).mean()))
        if prev is not None:
            diff2 = ((f - prev) ** 2).sum(axis=2)
            changed = diff2 > (T_COR * T_COR)
            d_series.append(float(changed.mean()))
            hot = block_view(changed) >= BLOCK_HOT
            nblocks_series.append(int(hot.sum()))
            block_hits += hot.astype(np.int64)
            motion_series.append(_phase_shift(prev_luma, luma))
            n += 1
        prev = f
        prev_luma = luma

    total_blocks = GX * GY
    if n == 0:
        return {"frames_sampled": len(bright_series), "error_scan": "poucos frames"}

    d = np.array(d_series)
    nb = np.array(nblocks_series)
    bright = np.array(bright_series)
    edge = np.array(edge_series)
    motion = np.array(motion_series)

    # 1. tela estatica: blocos que quase nunca mudam (canvas congelado)
    static_blocks = int((block_hits < STATIC_MAX * n).sum())

    # 2. localizacao: menor conjunto de blocos com MASS_KEEP da atividade
    flat = block_hits.flatten().astype(float)
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    total_mass = csum[-1] if csum[-1] > 0 else 1.0
    k = int(np.searchsorted(csum, MASS_KEEP * total_mass)) + 1
    ys, xs = np.unravel_index(order[:k], (GY, GX))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    bbox_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / total_blocks

    # 3. recorrencia na faixa baixa (transicoes fora->dentro = eventos distintos)
    in_band = (d >= LOW_BAND[0]) & (d < LOW_BAND[1])
    events = int((in_band[1:] & ~in_band[:-1]).sum()) + int(in_band[0])

    hist, _ = np.histogram(d, bins=HIST_EDGES)
    hpct = (hist / max(len(d), 1) * 100).round(1)

    row = {
        "frames_sampled": n + 1,
        "d_mean": round(float(d.mean()), 5),
        "d_std": round(float(d.std()), 5),
        "d_p50": round(float(np.percentile(d, 50)), 5),
        "d_p90": round(float(np.percentile(d, 90)), 5),
        "d_p99": round(float(np.percentile(d, 99)), 5),
        "rest_frac": round(float((d < REST_MAX).mean()), 4),
        "lowband_frac": round(float(in_band.mean()), 4),
        "lowband_events": events,
        "nblocks_mean": round(float(nb.mean()), 2),
        "nblocks_p90": round(float(np.percentile(nb, 90)), 1),
        "static_frac": round(static_blocks / total_blocks, 3),
        "active_frac": round(k / total_blocks, 3),
        "bbox_area_frac": round(bbox_area, 3),
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bright_mean": round(float(bright.mean()), 1),
        "bright_std": round(float(bright.std()), 1),
        "edge_mean": round(float(edge.mean()), 2),
        "motion_mean": round(float(motion.mean()), 2),
        "motion_p90": round(float(np.percentile(motion, 90)), 2),
    }
    for col, p in zip(HIST_COLS, hpct):
        row[col] = float(p)
    return row


# ===========================================================================
# Fase B - transcricao (Groq Whisper) + visao (1 chamada/video). Precisam de
# GROQ_API_KEY e ENVIAM dados ao Groq. Opt-in (--transcript/--vision/--all).
# ===========================================================================
def transcript_metrics(source: str, duration: float) -> dict:
    """Metricas de fala via pipeline existente (Groq Whisper, fallback local)."""
    try:
        import processor
    except Exception as e:
        return {"transcript_error": f"import: {type(e).__name__}"}
    wd = tempfile.mkdtemp(prefix="vtm_tr_")
    try:
        audio = processor.extract_audio(source, wd)
        tr = processor.transcribe(audio, wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    segs = tr.get("segments") or []
    # Conta palavras por espaco (consistente): a lista 'words' do Whisper traz
    # word-tokens (sub-palavras) cuja contagem varia entre chamadas do Groq.
    nwords = len((tr.get("text") or "").split())
    speech = sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in segs)
    return {
        "transcript_words": nwords,
        "words_per_min": round(nwords / (duration / 60), 1) if duration else 0.0,
        "speech_coverage": round(min(speech / duration, 1.0), 3) if duration else 0.0,
        "transcript_engine": tr.get("engine") or "",
    }


VISION_CATS = ["tabuleiro", "slides", "talking_head", "screencast", "dashboard",
               "gameplay", "esporte", "vlog", "outro"]
VISION_PROMPT = (
    "Esta imagem e uma COLAGEM de quadros amostrados de um video. Classifique o TIPO "
    "visual predominante e responda SOMENTE com JSON valido, sem texto extra:\n"
    '{"categoria": "<um de: ' + "|".join(VISION_CATS) + '>", '
    '"rosto": true|false, "texto": true|false}\n'
    "rosto = ha rosto humano visivel; texto = ha texto substancial na tela."
)


def _montage_bytes(source: str, duration: float, n: int = 6, cell_w: int = 360):
    """Extrai n quadros espacados e monta uma colagem 3xN -> bytes JPEG (ou None)."""
    from PIL import Image
    ffmpeg_tools.ensure_on_path()
    wd = tempfile.mkdtemp(prefix="vtm_mon_")
    thumbs = []
    try:
        for i in range(n):
            t = duration * (i + 1) / (n + 1) if duration else 0.0
            out = os.path.join(wd, f"f{i}.jpg")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", f"{t:.2f}", "-i", source, "-frames:v", "1",
                            "-vf", f"scale={cell_w}:-1", out],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            if os.path.isfile(out):
                thumbs.append(Image.open(out).convert("RGB"))
        if not thumbs:
            return None
        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        cw = max(t.width for t in thumbs)
        ch = max(t.height for t in thumbs)
        grid = Image.new("RGB", (cw * cols, ch * rows), (0, 0, 0))
        for idx, th in enumerate(thumbs):
            grid.paste(th, ((idx % cols) * cw, (idx // cols) * ch))
        mpath = os.path.join(wd, "montage.jpg")
        grid.save(mpath, "JPEG", quality=80)
        with open(mpath, "rb") as fh:
            return fh.read()
    finally:
        for th in thumbs:
            try:
                th.close()
            except Exception:
                pass
        shutil.rmtree(wd, ignore_errors=True)


def vision_metrics(source: str, duration: float) -> dict:
    """Categoria semantica + flags rosto/texto via 1 chamada Groq vision sobre colagem."""
    import frame_annotators as fa
    if not fa.groq_vision_available():
        return {"vision_error": "sem groq/GROQ_API_KEY"}
    try:
        data = _montage_bytes(source, duration)
    except Exception as e:
        return {"vision_error": f"colagem: {type(e).__name__}"}
    if not data:
        return {"vision_error": "sem frames"}
    import base64
    from groq import Groq
    b64 = base64.b64encode(data).decode()
    resp = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
        model=fa.GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=120, temperature=0)
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    cat = str(obj.get("categoria", "")).strip().lower()
    return {
        "vision_category": cat if cat in VISION_CATS else (cat or "?"),
        "vision_face": bool(obj["rosto"]) if "rosto" in obj else "",
        "vision_text": bool(obj["texto"]) if "texto" in obj else "",
        "vision_raw": "" if cat in VISION_CATS else raw[:200],
    }


def compute_metrics(source: str, do_transcript: bool = False,
                    do_vision: bool = False) -> dict:
    """Linha de metricas de UM video (caminho local ou URL). Erros por fase viram
    coluna '*_error'. URLs sao baixadas via yt-dlp para um temp e medidas localmente."""
    row = {"source": source, "name": os.path.basename(source.rstrip("/")) or source}
    dl_dir = None
    dur = 0.0
    try:
        local = source
        if _is_url(source):  # baixa para um temp (limpo no finally)
            import processor
            dl_dir = tempfile.mkdtemp(prefix="vtm_dl_")
            local, info = processor.download_or_locate(source, dl_dir)
            if isinstance(info, dict) and info.get("title"):
                row["name"] = info["title"]
        try:  # Fase A (sem chave): container + audio + assinatura + proxies
            cont = probe_container(local)
            row.update(cont)
            dur = cont.get("duration_s") or 0.0
            if cont.get("has_audio"):
                row.update(audio_metrics(local, dur))
            row.update(change_signature(local))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        if do_transcript and row.get("has_audio"):
            try:
                row.update(transcript_metrics(local, dur))
            except Exception as e:
                row["transcript_error"] = f"{type(e).__name__}: {e}"
        if do_vision:
            try:
                row.update(vision_metrics(local, dur))
            except Exception as e:
                row["vision_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # falha no download/resolucao da fonte
        row["error"] = f"download: {type(e).__name__}: {e}"
    finally:
        if dl_dir:
            shutil.rmtree(dl_dir, ignore_errors=True)
    return row


def gather_sources(inputs: list[str]) -> list[str]:
    """Expande pasta(s) / lista(.txt) / paths em uma lista de fontes."""
    srcs: list[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            for dp, _, files in os.walk(inp):
                for fn in sorted(files):
                    if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                        srcs.append(os.path.join(dp, fn))
        elif inp.lower().endswith(".txt") and os.path.isfile(inp):
            with open(inp, encoding="utf-8") as fh:
                srcs += [ln.strip() for ln in fh
                         if ln.strip() and not ln.strip().startswith("#")]
        else:
            srcs.append(inp)
    return srcs


def write_csv(rows: list[dict], out_path: str) -> None:
    """Grava o CSV com a uniao ordenada das chaves (preserva ordem de aparicao)."""
    cols: list[str] = []
    for r in rows:
        for kcol in r:
            if kcol not in cols:
                cols.append(kcol)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch de metricas de video -> CSV")
    ap.add_argument("inputs", nargs="+", help="pasta, lista .txt, ou video(s)")
    ap.add_argument("-o", "--out", default="metrics.csv", help="CSV de saida")
    ap.add_argument("--transcript", action="store_true",
                    help="Fase B: transcreve (Groq Whisper) -> words/min, cobertura de fala")
    ap.add_argument("--vision", action="store_true",
                    help="Fase B: 1 chamada de visao/video -> categoria, rosto, texto")
    ap.add_argument("--all", action="store_true", help="liga --transcript e --vision")
    args = ap.parse_args()

    do_transcript = args.transcript or args.all
    do_vision = args.vision or args.all
    if do_transcript or do_vision:  # Fase B precisa da GROQ_API_KEY do .env
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except Exception:
            pass

    sources = gather_sources(args.inputs)
    if not sources:
        print("Nenhuma fonte encontrada.")
        return
    fase = "A+B" if (do_transcript or do_vision) else "A"
    print(f"{len(sources)} video(s) | fase {fase} -> {args.out}\n")

    rows: list[dict] = []
    for i, src in enumerate(sources, 1):
        t0 = time.time()
        print(f"[{i}/{len(sources)}] {os.path.basename(src)} ...", end="", flush=True)
        row = compute_metrics(src, do_transcript=do_transcript, do_vision=do_vision)
        rows.append(row)
        dt = time.time() - t0
        err = row.get("error") or row.get("error_scan")
        if err:
            print(f" ERRO ({dt:.1f}s): {err}")
        else:
            extra = (f" cat={row.get('vision_category')}" if do_vision else "")
            extra += (f" wpm={row.get('words_per_min')}" if do_transcript else "")
            print(f" ok ({dt:.1f}s)  rest={row.get('rest_frac')} "
                  f"d_mean={row.get('d_mean')} static={row.get('static_frac')}{extra}")
        write_csv(rows, args.out)  # incremental: sobrevive a interrupcao

    print(f"\nPronto: {len(rows)} linha(s) em {args.out}")


if __name__ == "__main__":
    main()

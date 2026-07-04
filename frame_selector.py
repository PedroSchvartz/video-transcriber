"""
Selecao de timestamps relevantes para captura de frames + deduplicacao visual.

Fontes de timestamps combinadas:
  1. Inicio de fala/frase   -> derivado dos word-timestamps do Whisper (Groq)
  2. Mudanca de cena visual -> detectada pelo ffmpeg (filtro scene)
  3. Edicao (pixels sujos)  -> fracao de pixels que mudam de cor entre quadros
  4. Onset de audio         -> fim de pausas relevantes (silencedetect)
  5. Capitulos              -> marcadores curados (yt-dlp, quando existirem)

Os timestamps sao mesclados e deduplicados por uma janela temporal minima.
Depois da extracao, os frames passam por dedup visual (perceptual hash + pixels sujos).
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass

import config

log = logging.getLogger("video_transcriber.frames")

# Windows: suprime a janela de console que cada ffmpeg piscaria na tela quando o
# worker roda destacado (sem console proprio). Em POSIX e 0 (sem efeito).
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# --- Parametros ajustaveis -------------------------------------------------
SPEECH_GAP_THRESHOLD = 0.7   # seg de silencio entre palavras p/ marcar nova fala
# Cena: limiar AUTO-CALIBRADO a distribuicao de scores do proprio video, em vez
# de um corte fixo (ver detect_scenes). media + K*desvio, preso entre piso e teto.
SCENE_ADAPT_K = 2.0          # quantos desvios acima da media conta como mudanca
SCENE_SCORE_FLOOR = 0.06     # piso: abaixo disso e micro-variacao/ruido (video parado)
SCENE_SCORE_CAP = 0.40       # teto: em video muito agitado nao exige mais que isso
# Capturar EXATAMENTE no instante da mudanca pega a tela em transicao (borrada /
# so meio composta). Capturamos um tico ANTES = ultimo frame estavel do que estava
# na tela (ja-composto, nitido) — ver detect_scenes.
SCENE_LEAD = 0.15            # seg antes da mudanca a capturar (frame ja assentado)
SCENE_SEARCH = 0.5           # janela (s) no fim do shot p/ escolher o frame mais nitido
AUDIO_DROP_DB = 14           # silencio = este tanto de dB abaixo da media (adaptativo)
AUDIO_MIN_SILENCE = 0.35     # seg de silencio antes de um onset contar como evento
MIN_GAP_SECONDS = 1.5        # janela minima entre dois frames selecionados
PHASH_DISTANCE = 6           # limiar de estrutura (pHash) p/ considerar 2 frames iguais
COLOR_DISTANCE = 5           # limiar de cor (colorhash); pHash ignora cor, este cobre

# Trigger "edicao": conta a fracao de pixels cuja DISTANCIA de cor ao quadro
# anterior passa de T_cor — ambos os limiares AUTO-calibrados (ver detect_edits).
EDIT_FPS = 6                 # quadros/s amostrados para medir mudanca
EDIT_SIZE = 128              # downscale (quadrado) — fracao suja e robusta a baixa res
EDIT_NBINS, EDIT_BINW = 64, 4   # histograma da distancia de cor por pixel (0..256)
EDIT_NOISE_BIN = 4           # bin ~distancia 16: acima disso conta como "mudou" (atividade)
EDIT_NOISE_FACTOR = 1.8      # T_cor = ruido(p99 dos quadros quietos) * fator
T_COR_MIN, T_COR_MAX = 40.0, 150.0   # piso/teto da resistencia a cor por pixel
EDIT_AREA_K = 2.0            # limiar de area = media + K*desvio da fracao suja
EDIT_AREA_FLOOR = 0.02       # area minima (2% da tela) p/ contar como edicao

# Dedup por pixels sujos (alem do pHash): compara dois prints pela fracao de
# pixels que diferem de cor — distingue telas de texto que o pHash funde por engano.
DEDUP_T_COR = 50.0           # distancia de cor p/ contar um pixel como "diferente"
DIRTY_DUP_LOW = 0.02         # < 2% diferente -> quase identicos -> duplicata
DIRTY_DUP_VETO = 0.10        # pHash so funde se < 10% diferente (senao e outro conteudo)

# Budget de prints proporcional a duracao (ver screenshot_budget).
# Defaults: MIN=1, MAX=12. Configuravel via VT_MIN/MAX_FRAMES_PER_MINUTE ou config.json.
# MAX=9999 desativa o teto (sem limite de prints por minuto).

# Custo/seguranca em videos longos:
SCENE_SCALE_WIDTH = 192      # downscale antes do filtro de cena (calculo mais barato)
FFMPEG_TIMEOUT = 900         # teto (s) p/ um passe de ffmpeg sobre a midia inteira


@dataclass
class FrameMark:
    """Um momento candidato a virar print, com a razao que o gerou."""
    time: float
    reason: str  # "fala" | "cena" | "edicao" | "audio" | "capitulo" | "intervalo"
    # Para "cena"/"edicao": janela (lo, hi) do shot onde a extracao busca o frame
    # mais nitido (evita meio-fade/borrao). None nas demais -> captura no `time`.
    window: tuple[float, float] | None = None


def from_words(words: list[dict], gap_threshold: float = SPEECH_GAP_THRESHOLD) -> list[FrameMark]:
    """Marca o inicio de novas falas a partir dos word-timestamps do Whisper.

    Uma nova fala comeca quando ha um silencio relevante entre palavras,
    ou quando a palavra anterior termina uma frase (. ! ?).
    """
    marks: list[FrameMark] = []
    if not words:
        return marks

    # Primeira palavra: sempre marca o inicio do conteudo falado.
    marks.append(FrameMark(time=float(words[0]["start"]), reason="fala"))

    for prev, cur in zip(words, words[1:]):
        gap = float(cur["start"]) - float(prev["end"])
        prev_word = str(prev.get("word", "")).strip()
        ends_sentence = prev_word.endswith((".", "!", "?", "…"))
        if gap > gap_threshold or (ends_sentence and gap > 0.25):
            marks.append(FrameMark(time=float(cur["start"]), reason="fala"))
    return marks


def detect_scenes(video_path: str, timeout: float = FFMPEG_TIMEOUT) -> list[FrameMark]:
    """Detecta mudancas de cena com limiar AUTO-CALIBRADO ao proprio video.

    Um corte fixo erra para os dois lados: alto demais perde overlays/b-roll/texto
    na tela (mudancas parciais, score ~0.1-0.3), baixo demais vira ruido. Em vez
    disso, faz uma passada coletando o score de cena de CADA frame e deriva o
    limiar da distribuicao do video — `media + K*desvio`, preso entre um piso
    (ignora micro-variacao de video quase parado) e um teto (nao exige demais em
    video agitado). Mesma filosofia adaptativa do detector de audio.

    `-skip_frame nokey` + downscale mantem a passada barata em videos longos. Em
    timeout, segue sem marcas de cena — as outras fontes garantem cobertura.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-skip_frame", "nokey", "-i", video_path,
        "-filter:v",
        f"scale={SCENE_SCALE_WIDTH}:-2,select='gt(scene,0)',metadata=print:file=-",
        "-an", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout,
                              creationflags=CREATE_NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg nao encontrado no PATH. Instale com: winget install Gyan.FFmpeg"
        )
    except subprocess.TimeoutExpired:
        log.warning("detect_scenes excedeu %ss; seguindo sem marcas de cena", timeout)
        return []

    # metadata=print emite, por frame, uma linha com 'pts_time:T' seguida de
    # 'lavfi.scene_score=S'. Pareia as duas para ter (tempo, score) de cada frame.
    pairs: list[tuple[float, float]] = []
    cur_t: float | None = None
    for line in proc.stdout.splitlines():
        mt = re.search(r"pts_time:(\d+\.?\d*)", line)
        if mt:
            cur_t = float(mt.group(1))
            continue
        ms = re.search(r"lavfi\.scene_score=([\d.]+)", line)
        if ms and cur_t is not None:
            pairs.append((cur_t, float(ms.group(1))))
            cur_t = None

    scores = [s for _, s in pairs if s > 0]
    if not scores:
        log.info("Cena: sem variacao (video estatico?); 0 marcas")
        return []

    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    threshold = min(max(mean + SCENE_ADAPT_K * std, SCENE_SCORE_FLOOR), SCENE_SCORE_CAP)

    # Os frames acima do limiar sao os instantes de MUDANCA.
    change_times = [t for t, s in pairs if s >= threshold]
    marks = _marks_from_changes(change_times, "cena")
    log.info("Cena: media=%.4f std=%.4f -> limiar auto=%.4f -> %d mudancas, %d marcas",
             mean, std, threshold, len(change_times), len(marks))
    return marks


def _marks_from_changes(change_times: list[float], reason: str) -> list[FrameMark]:
    """Transforma instantes de MUDANCA em marcas prontas para captura.

    Capturar no instante exato da mudanca pega a tela em transicao (borrada / so
    meio composta). Em vez disso: entre mudancas a menos de MIN_GAP fica com a MAIS
    TARDIA (numa tela que se compoe palavra por palavra, o frame logo antes de ser
    trocada e o mais completo) e da a cada marca uma janela [lo, hi] no fim do shot
    — a extracao escolhe ali o frame mais nitido, sem vazar para a proxima tela.
    """
    if not change_times:
        return []
    kept: list[float] = []
    for t in sorted(change_times):
        if kept and t - kept[-1] <= MIN_GAP_SECONDS:
            kept[-1] = t            # prefere a mudanca mais tardia do grupo
        else:
            kept.append(t)

    marks: list[FrameMark] = []
    for i, c in enumerate(kept):
        prev = kept[i - 1] if i > 0 else 0.0
        hi = c - 0.05
        lo = max(prev + 0.05, c - SCENE_SEARCH)
        if hi <= lo:                # shot curtissimo: cai p/ um instante so
            lo = hi = max(0.0, c - SCENE_LEAD)
        marks.append(FrameMark(time=max(0.0, c - SCENE_LEAD), reason=reason,
                               window=(lo, hi)))
    return marks


def detect_edits(video_path: str, timeout: float = FFMPEG_TIMEOUT,
                 progress=None) -> list[FrameMark]:
    """Detecta EDICOES pela fracao de pixels 'sujos' entre quadros consecutivos.

    `progress(segundos_varridos)` (opcional): chamado durante a varredura com a
    posicao (em segundos de video) ja processada — e o sinal honesto de quanto
    falta para a deteccao terminar (esta passada e a mais lenta das tres).

    Complementa o detector de cena com DOIS limiares auto-calibrados ao video:
      - T_cor (resistencia a cor): um pixel so conta como mudado se a DISTANCIA de
        cor ao quadro anterior passa de T_cor. Calibrado pelo ruido do proprio
        video (p99 da distancia nos quadros quietos) — grao/compressao/luz leve nao
        conta, mas preto->branco / verde->amarelo conta.
      - Limiar de AREA: a fracao da tela que precisa mudar p/ contar como edicao,
        calibrada pela distribuicao (media + K*desvio) — adapta a velocidade de
        edicao do video.

    Uma passada de ffmpeg em streaming (downscale + ~6fps) alimenta um histograma
    por quadro; o numpy calibra e marca os picos. Em timeout, degrada para [].
    """
    import numpy as np

    w = h = EDIT_SIZE
    frame_bytes = w * h * 3
    total_px = w * h
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", video_path,
        "-vf", f"fps={EDIT_FPS},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=CREATE_NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg nao encontrado no PATH (winget install Gyan.FFmpeg)")

    def _read_exact(stream, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    hists: list = []
    times: list[float] = []
    prev = None
    idx = 0
    start = time.time()
    last_report = 0.0
    while True:
        if time.time() - start > timeout:
            log.warning("detect_edits excedeu %ss; seguindo com o que coletou", timeout)
            break
        buf = _read_exact(proc.stdout, frame_bytes)
        if buf is None:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.int32)
        if prev is not None:
            dist = np.sqrt(((cur - prev) ** 2).sum(axis=2))   # distancia de cor por pixel
            db = np.clip((dist // EDIT_BINW).astype(np.int32), 0, EDIT_NBINS - 1)
            hists.append(np.bincount(db.ravel(), minlength=EDIT_NBINS))
            times.append(idx / EDIT_FPS)
        prev = cur
        idx += 1
        if progress is not None and time.time() - last_report >= 1.0:
            last_report = time.time()
            try:
                progress(idx / EDIT_FPS)   # segundos de video ja varridos
            except Exception:
                pass
    try:
        proc.terminate()
    except Exception:
        pass

    if len(hists) < 3:
        log.info("Edicao: poucos quadros (%d); 0 marcas", len(hists))
        return []

    hist = np.array(hists)
    bin_centers = (np.arange(EDIT_NBINS) + 0.5) * EDIT_BINW

    # T_cor pelo RUIDO: nos quadros mais quietos (metade menos ativa), 99% das
    # distancias por pixel ficam abaixo deste valor — esse e o piso de ruido.
    activity = hist[:, EDIT_NOISE_BIN:].sum(axis=1)
    quiet = hist[activity <= np.percentile(activity, 50)].sum(axis=0)
    cum = np.cumsum(quiet)
    noise = float(bin_centers[np.searchsorted(cum, cum[-1] * 0.99)]) if cum[-1] > 0 else 0.0
    t_cor = min(max(noise * EDIT_NOISE_FACTOR, T_COR_MIN), T_COR_MAX)

    # Fracao suja por quadro (pixels com distancia > t_cor) e limiar de area auto.
    bin_cut = int(t_cor // EDIT_BINW)
    dirty = hist[:, bin_cut + 1:].sum(axis=1) / total_px
    area_thr = max(float(dirty.mean() + EDIT_AREA_K * dirty.std()), EDIT_AREA_FLOOR)

    change_times = [times[i] for i in range(len(dirty)) if dirty[i] >= area_thr]
    marks = _marks_from_changes(change_times, "edicao")
    log.info("Edicao: T_cor=%.0f (ruido %.0f) limiar_area=%.4f -> %d quadros, %d marcas",
             t_cor, noise, area_thr, len(change_times), len(marks))
    return marks


def _mean_volume_db(audio_path: str) -> float | None:
    """Volume medio do audio em dB (via volumedetect), p/ limiar adaptativo."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=FFMPEG_TIMEOUT, creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        log.warning("volumedetect excedeu o tempo; usando limiar padrao")
        return None
    m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", proc.stderr)
    return float(m.group(1)) if m else None


def detect_audio_events(audio_path: str | None,
                        drop_db: float = AUDIO_DROP_DB,
                        min_silence: float = AUDIO_MIN_SILENCE) -> list[FrameMark]:
    """Detecta onsets sonoros: o fim de cada pausa relevante.

    Marca o instante em que o som retorna depois de uma pausa — captura efeitos
    sonoros, entradas de musica, batidas e falas apos silencio. Complementa a
    deteccao por fala (so voz) e a de cena (so visual).

    O limiar de silencio e ADAPTATIVO: `drop_db` abaixo do volume medio do
    proprio video. Assim funciona igual em audios gravados alto ou baixo, em vez
    de depender de um valor absoluto que varia com o piso de ruido de cada video.
    """
    if not audio_path:
        return []
    try:
        mean = _mean_volume_db(audio_path)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg nao encontrado no PATH (winget install Gyan.FFmpeg)")
    noise = round(mean - drop_db) if mean is not None else -35

    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
             "-af", f"silencedetect=noise={noise}dB:d={min_silence}", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=FFMPEG_TIMEOUT, creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        log.warning("silencedetect excedeu o tempo; seguindo sem onsets de audio")
        return []
    marks = [FrameMark(time=float(m.group(1)), reason="audio")
             for m in re.finditer(r"silence_end:\s*([\d.]+)", proc.stderr)]
    log.info("Audio: media=%.1fdB, limiar=%ddB -> %d onsets",
             mean if mean is not None else 0.0, noise, len(marks))
    return marks


def from_chapters(chapters: list[dict] | None) -> list[FrameMark]:
    """Um frame no inicio de cada capitulo (quando o video tiver capitulos)."""
    marks: list[FrameMark] = []
    for ch in chapters or []:
        start = ch.get("start_time")
        if start is not None:
            marks.append(FrameMark(time=float(start), reason="capitulo"))
    return marks


def uniform(duration: float, spacing: float) -> list[FrameMark]:
    """Rede de seguranca: marcas em intervalos regulares ao longo do video.

    Garante cobertura visual mesmo em videos sem fala e com poucas mudancas de
    cena detectadas (ex.: reels "mudos" com texto). Tem a menor prioridade —
    no merge, qualquer marca de fala/cena/capitulo proxima prevalece.
    """
    marks: list[FrameMark] = []
    if duration <= 0 or spacing <= 0:
        return marks
    t = spacing / 2.0  # comeca no meio do primeiro intervalo
    while t < duration:
        marks.append(FrameMark(time=t, reason="intervalo"))
        t += spacing
    return marks


def merge(*mark_lists: list[FrameMark], min_gap: float = MIN_GAP_SECONDS) -> list[FrameMark]:
    """Mescla varias fontes, ordena por tempo e remove marcas muito proximas.

    Em caso de empate temporal, capitulo > cena > edicao > fala (mais significativo vence).
    """
    priority = {"capitulo": 0, "cena": 1, "edicao": 2, "fala": 3, "audio": 4, "intervalo": 5}
    all_marks = [m for lst in mark_lists for m in lst]
    all_marks.sort(key=lambda m: (m.time, priority.get(m.reason, 9)))

    kept: list[FrameMark] = []
    for mark in all_marks:
        if not kept or (mark.time - kept[-1].time) >= min_gap:
            kept.append(mark)
    return kept


def cap_uniform(marks: list[FrameMark], max_count: int) -> list[FrameMark]:
    """Se houver marcas demais, faz amostragem uniforme no tempo.

    Garante cobertura do video inteiro em vez de concentrar prints no inicio.
    """
    if len(marks) <= max_count:
        return marks
    step = len(marks) / max_count
    return [marks[int(i * step)] for i in range(max_count)]


def screenshot_budget(requested: int, duration: float) -> int:
    """Ajusta o numero de prints a duracao do video.

    Piso de min_fpm por minuto (no minimo 1, mesmo em videos < 1 min) e teto de
    max_fpm por minuto. O valor pedido e encaixado nessa faixa. max_fpm >= 9999
    desativa o teto — o pipeline usa o numero pedido sem cap por minuto.
    Ex. padrao: video de 1 min -> [1, 12]; de 5 min -> [5, 60].
    """
    cfg = config.get()
    min_fpm = cfg["min_frames_per_minute"]
    max_fpm = cfg["max_frames_per_minute"]
    minutes = max(1, math.ceil((duration or 0) / 60))
    floor = min_fpm * minutes
    if max_fpm >= 9999:
        return max(floor, int(requested))
    return max(floor, min(int(requested), max_fpm * minutes))


def fill_to_budget(smart: list[FrameMark], duration: float, max_count: int) -> list[FrameMark]:
    """Compoe ate max_count frames priorizando marcas "inteligentes".

    Mantem as marcas de fala/cena/capitulo (amostrando se passarem do teto) e
    so preenche o que faltar com marcas uniformes nos maiores buracos temporais.
    Assim a rede de seguranca nunca rouba o lugar de um momento relevante.
    """
    smart = sorted(smart, key=lambda m: m.time)
    if len(smart) >= max_count:
        return cap_uniform(smart, max_count)
    if duration <= 0:
        return smart

    deficit = max_count - len(smart)
    spacing = max(MIN_GAP_SECONDS, duration / (deficit + 1))
    merged = merge(smart, uniform(duration, spacing))  # smart vence empates
    if len(merged) > max_count:
        merged = cap_uniform(merged, max_count)
    return merged


def drop_blank(frames: list[tuple[FrameMark, str]],
               min_mean: float = 12.0, min_std: float = 3.0):
    """Remove frames sem informacao visual: pretos, fades e telas uniformes.

    Cobre o caso comum de capturar em t=0 num video com fade-in (frame preto)
    e tambem fades no meio. Avalia luminancia media e desvio padrao em escala
    de cinza — um frame "vazio" tem media baixa e/ou variacao quase nula.
    """
    from PIL import Image, ImageStat

    kept: list[tuple[FrameMark, str]] = []
    for mark, path in frames:
        try:
            stat = ImageStat.Stat(Image.open(path).convert("L"))
            mean, std = stat.mean[0], stat.stddev[0]
        except Exception as exc:
            log.warning("Falha ao avaliar %s: %s", path, exc)
            kept.append((mark, path))  # na duvida, mantem
            continue
        if mean >= min_mean and std >= min_std:
            kept.append((mark, path))
        else:
            log.info("Frame nao-informativo descartado: t=%.1fs (mean=%.1f std=%.1f)",
                     mark.time, mean, std)
    return kept


def _small_rgb(path: str, size: int = 64):
    """Versao pequena RGB do frame (numpy int32) p/ comparar pixels sujos."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img, dtype=np.int32)


def _dirty_fraction(a, b, t_cor: float = DEDUP_T_COR) -> float:
    """Fracao de pixels cuja distancia de cor entre `a` e `b` passa de t_cor."""
    return float((((a - b) ** 2).sum(axis=2) > t_cor * t_cor).mean())


def _is_dup_of(ph, ch, small, sigs,
               distance: int = PHASH_DISTANCE,
               color_distance: int = COLOR_DISTANCE,
               t_cor: float = DEDUP_T_COR) -> bool:
    """True se (ph, ch, small) e duplicata de alguma assinatura em `sigs`.

    Mesma regra do dedup_visual, isolada para reuso pelo seletor online (janela
    de dedup limitada). `sigs` e uma sequencia de (phash, colorhash, small_rgb).
    Duplicata quando: a fracao de pixels diferentes e minima (< DIRTY_DUP_LOW), ou
    pHash E colorhash concordam E os pixels nao diferem muito (< DIRTY_DUP_VETO).
    """
    for p, c, arr in sigs:
        df = _dirty_fraction(small, arr, t_cor)
        if df < DIRTY_DUP_LOW:
            return True
        if (ph - p) <= distance and (ch - c) <= color_distance and df < DIRTY_DUP_VETO:
            return True
    return False


def dedup_visual(frames: list[tuple[FrameMark, str]],
                 distance: int = PHASH_DISTANCE,
                 color_distance: int = COLOR_DISTANCE,
                 t_cor: float = DEDUP_T_COR):
    """Remove frames quase identicos combinando pHash/colorhash e pixels sujos.

    `frames` e uma lista de (FrameMark, caminho_do_jpg), em ordem temporal. Um
    frame e descartado como duplicata de algum ja aceito quando:
      - a fracao de pixels DIFERENTES entre eles e minima (< DIRTY_DUP_LOW), ou
      - pHash E colorhash concordam que sao parecidos E os pixels nao diferem
        muito (< DIRTY_DUP_VETO).
    A fracao de pixels sujos veta o segundo caso: duas telas de texto preto com
    palavras DIFERENTES (que o pHash funde por terem a mesma composicao) tem
    fracao alta -> nao sao consideradas duplicatas.
    """
    import imagehash
    from PIL import Image

    kept: list[tuple[FrameMark, str]] = []
    sigs: list[tuple] = []  # (phash, colorhash, small_rgb)
    for mark, path in frames:
        try:
            img = Image.open(path)
            ph = imagehash.phash(img)
            ch = imagehash.colorhash(img)
            small = _small_rgb(path)
        except Exception as exc:  # frame corrompido -> ignora silenciosamente
            log.warning("Falha ao hashear %s: %s", path, exc)
            continue
        if _is_dup_of(ph, ch, small, sigs, distance, color_distance, t_cor):
            continue
        sigs.append((ph, ch, small))
        kept.append((mark, path))
    log.info("Dedup visual: %d -> %d frames", len(frames), len(kept))
    return kept

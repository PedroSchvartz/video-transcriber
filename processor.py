"""
Pipeline de processamento de video:

    fonte (URL ou .mp4) -> download/metadados -> extrai audio -> transcreve
                        -> seleciona timestamps -> extrai frames -> dedup visual

A transcricao usa Groq (Whisper Large v3 Turbo) quando ha GROQ_API_KEY;
caso contrario cai para faster-whisper local. Retorna um dicionario pronto
para o server MCP montar a resposta (texto + imagens).
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import ffmpeg_tools
import config
import frame_annotators as fa
import frame_selector as fs

log = logging.getLogger("video_transcriber.processor")

# Resolve ffmpeg/ffprobe no PATH assim que o modulo carrega (antes de qualquer chamada).
ffmpeg_tools.ensure_on_path()

# --- Parametros ------------------------------------------------------------
__version__ = "1.10.0"               # versao do pipeline; carimbada no meta.json
GROQ_MODEL = "whisper-large-v3-turbo"
LOCAL_MODEL = "base"                 # modelo faster-whisper de fallback
GROQ_SIZE_LIMIT = 24 * 1024 * 1024   # 24 MB — abaixo do limite free de 25 MB
CHUNK_SECONDS = 600                  # 10 min por pedaco quando o audio e longo
FRAME_MAX_WIDTH = 1080               # largura maxima do print (reduz base64)

# Guard rails (evitam travar a maquina em videos longos):
FFMPEG_TIMEOUT = 1800                # teto (s) p/ um passe de ffmpeg (download/extracao)
AUDIO_BITRATE = "24k"                # Opus p/ Whisper: ~16MB em 90min, cabe num request
# Teto de download pela MENOR dimensao (short side), via format_sort res:N. Cobre
# vertical (reels) e landscape. ATENCAO: nao usar filtro height<=N — em video
# VERTICAL o height e a dimensao ALTA (1080x1920 tem height=1920), entao height<=720
# capava reels a 360p. res:N usa min(w,h), entao 1080 = "1080p" nas duas orientacoes.
DOWNLOAD_RES = 1080
_cfg = config.load()  # lido apos load_dotenv(); reflete .env + config.json
MAX_DURATION_HARD = _cfg["max_duration_hours"] * 3600
SILENCE_THRESHOLD_DBFS = _cfg["silence_threshold_dbfs"]

# Cache persistente: cada video processado vira uma pasta com video + transcript
# + frames + meta.json. Permite reler o artefato e capturar frames sob demanda
# (get_frame) em conversas posteriores, sem reprocessar.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
MAX_CACHE_ENTRIES = _cfg["max_cache_entries"]

# Windows: o worker roda destacado (sem console proprio), entao cada ffmpeg/ffprobe
# filho ganharia uma janela de console nova que pisca na tela. CREATE_NO_WINDOW
# suprime isso. Em POSIX e 0 (sem efeito).
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# ETA por etapa: tempo tipico em segundos por segundo-de-video. Recalibrado a cada
# run no _timings.json; estes valores sao so o chute inicial (sem historico ainda).
# 'frames' foi dividido em 'deteccao' (cena/edicao/audio — a parte pesada e
# variavel, agora com sinal de progresso real) e 'extracao' (1 frame por marca,
# fracao honesta i/n). Separar as duas e o que permite um ETA que nao trava.
ETA_STAGES = ("download", "audio", "transcribe", "deteccao", "extracao")
ETA_DEFAULT_RATES = {"download": 0.05, "audio": 0.015, "transcribe": 0.01,
                     "deteccao": 0.09, "extracao": 0.02}
TIMINGS_PATH = os.path.join(CACHE_DIR, "_timings.json")


def _load_timings() -> dict:
    try:
        with open(TIMINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _record_timing(stage: str, secs: float, video_secs: float) -> None:
    """Media movel da taxa (s por s-de-video) de cada etapa — calibra a proxima run."""
    if video_secs <= 0 or secs <= 0:
        return
    rate = secs / video_secs
    t = _load_timings()
    cur = t.get(stage) or {}
    n = int(cur.get("n", 0))
    avg = float(cur.get("rate", rate))
    t[stage] = {"rate": (avg * n + rate) / (n + 1), "n": n + 1}
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(TIMINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(t, f)
    except Exception:
        pass


# ===========================================================================
# Download / localizacao da fonte
# ===========================================================================
def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "www."))


# === Erros de servicos NAO-locais (rede/nuvem) =============================
# Servicos locais (ffmpeg, faster-whisper ja baixado) nao entram aqui. Os de
# rede podem falhar por: CONEXAO (sem internet/DNS/timeout — sem status HTTP),
# LIMITE (cota do free tier estourada — NAO e erro real), COBRANCA (servico pago
# sem credito) ou CONTEUDO/OUTRO. A conexao e checada por ausencia de status, sem
# conflitar com os demais (que tem status HTTP).
_CONN_HINTS = (
    "connection", "timed out", "timeout", "getaddrinfo", "failed to resolve",
    "name resolution", "network is unreachable", "temporary failure", "unreachable",
    "max retries", "no address associated", "urlopen error", "ssl",
    "remote end closed", "[errno 11001]", "[errno -2]", "winerror 10054",
    "winerror 10060", "winerror 10051", "connection reset", "connection refused",
)


def _looks_like_connection_error(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _CONN_HINTS)


class _QuietYDLLogger:
    """Silencia a saida do yt-dlp. Os erros viram excecao (que classificamos); sem
    isso, linhas cruas 'ERROR: ...' vazam pro terminal e assustam o usuario."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


_QUIET_YDL = _QuietYDLLogger()


def _describe_download_error(exc: Exception) -> tuple[str, str]:
    """(categoria, mensagem) para falha do yt-dlp ao baixar/sondar (servico de rede)."""
    text = str(getattr(exc, "msg", "") or exc).strip()
    text = re.sub(r"^(ERROR:\s*)+", "", text).strip()   # tira o(s) prefixo(s) cru(s) do yt-dlp
    low = text.lower()
    # Falha ao LER os cookies do Chrome (DPAPI/decrypt) — distinta de "exige login".
    if any(k in low for k in ("dpapi", "decrypt", "could not copy", "unable to read cookie")):
        return "cookies", text
    if _looks_like_connection_error(low):
        return "conexao", text
    if "429" in low or "too many requests" in low or ("rate" in low and "limit" in low):
        return "limite", text
    if any(k in low for k in ("private", "unavailable", "removed", "geo", "sign in",
                              "login", "age", "not available", "members-only", "drm")):
        return "conteudo", text
    return "outro", text


def _download_error_text(cat: str, msg: str, source: str) -> str:
    if cat == "conexao":
        return (f"Sem conexao com a internet/plataforma ao baixar ({source}). "
                f"Verifique sua rede e tente de novo. Detalhe: {msg}")
    if cat == "limite":
        return (f"A plataforma limitou as requisicoes (HTTP 429) ao baixar ({source}). "
                f"Aguarde alguns minutos e tente de novo. Detalhe: {msg}")
    if cat == "conteudo":
        return (f"Video indisponivel (privado/removido/restrito por regiao ou login): "
                f"{source}. Se for privado, faca login no site pelo Chrome. Detalhe: {msg}")
    if cat == "cookies":
        return (f"Nao consegui ler os cookies do Chrome para acessar conteudo logado "
                f"({source}). Feche o Chrome e tente de novo, e confirme que voce esta "
                f"logado no site no Chrome. Detalhe: {msg}")
    return (f"Nao foi possivel baixar ({source}) — verifique se o link e mesmo de um "
            f"video (YouTube/Instagram/etc). Detalhe: {msg}")


def download_or_locate(source: str, workdir: str, progress=None) -> tuple[str, dict]:
    """Baixa o video (URL) ou usa o arquivo local; retorna (caminho, metadados).

    `progress(frac, eta)` (opcional): chamado durante o download via hook do yt-dlp
    com a fracao baixada (0-1) e o ETA em segundos que o proprio yt-dlp calcula.
    """
    if not _is_url(source):
        path = os.path.expanduser(source.strip('"').strip("'"))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
        return path, {"title": os.path.basename(path), "source": path}

    import yt_dlp

    def _hook(d):
        if progress and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            dl = d.get("downloaded_bytes") or 0
            try:
                progress((dl / tot) if tot else None, d.get("eta"))
            except Exception:
                pass

    opts = {
        "progress_hooks": [_hook],
        # Melhor video + melhor audio, limitado a DOWNLOAD_RES pela MENOR dimensao
        # (format_sort res:N usa min(largura,altura)). Vale p/ vertical e landscape
        # — o antigo `height<=720` baixava reels (verticais) a 360p.
        "format": "bv*+ba/b",
        "format_sort": [f"res:{DOWNLOAD_RES}"],
        "outtmpl": os.path.join(workdir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "logger": _QUIET_YDL,   # nao deixa o yt-dlp cuspir 'ERROR: ...' cru no terminal
        "noplaylist": True,
        "restrictfilenames": True,
        "socket_timeout": 60,
    }

    def _run(extra: dict | None = None) -> dict:
        o = {**opts, **(extra or {})}
        with yt_dlp.YoutubeDL(o) as ydl:
            return ydl.extract_info(source, download=True)

    try:
        info = _run()
    except Exception as exc:
        cat, msg = _describe_download_error(exc)
        # Cookies do Chrome SO ajudam conteudo logado (privado/login). Em conexao,
        # limite, URL nao-suportada etc., reintentar e inutil e ainda surfaria o erro
        # de DPAPI dos cookies: entao falha logo com a mensagem clara do 1o erro.
        if cat != "conteudo":
            raise RuntimeError(_download_error_text(cat, msg, source)) from exc
        log.warning("Conteudo exige login (%s); tentando cookies do Chrome", msg)
        try:
            info = _run({"cookiesfrombrowser": ("chrome",)})
        except Exception as exc2:
            cat2, msg2 = _describe_download_error(exc2)
            raise RuntimeError(_download_error_text(cat2, msg2, source)) from exc2

    files = sorted(glob.glob(os.path.join(workdir, "source.*")))
    if not files:
        raise RuntimeError("Download concluiu mas nenhum arquivo foi encontrado")
    video_path = files[0]

    metadata = {
        "title": info.get("title") or os.path.basename(video_path),
        "description": info.get("description"),
        "chapters": info.get("chapters"),
        "duration": info.get("duration"),
        "source": info.get("webpage_url") or source,
    }
    return video_path, metadata


def probe_source(source: str) -> dict:
    """Metadados rapidos (titulo, duracao, capitulos) SEM baixar o video.

    Usado pelo launcher para validar a duracao e estimar o trabalho antes de
    comprometer download + processamento (que rodam em background).
    """
    if not _is_url(source):
        path = os.path.expanduser(source.strip('"').strip("'"))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
        return {"title": os.path.basename(path), "duration": get_duration(path),
                "source": path, "chapters": None}

    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "logger": _QUIET_YDL,
                           "noplaylist": True, "skip_download": True,
                           "socket_timeout": 30}) as ydl:
        info = ydl.extract_info(source, download=False)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "source": info.get("webpage_url") or source,
        "chapters": info.get("chapters"),
    }


# ===========================================================================
# Audio
# ===========================================================================
def _run_ffmpeg(cmd: list[str], timeout: float = FFMPEG_TIMEOUT) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              creationflags=CREATE_NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg nao encontrado no PATH. Instale com: winget install Gyan.FFmpeg "
            "e reabra o terminal."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg excedeu {timeout}s e foi abortado ({cmd[0]} {cmd[-1]})."
        )


def get_duration(path: str) -> float:
    """Duracao em segundos via ffprobe (0.0 se nao conseguir descobrir)."""
    proc = _run_ffmpeg([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def extract_audio(video_path: str, workdir: str) -> str | None:
    """Extrai audio mono 16kHz em Opus (~24kbps). None se nao houver audio.

    Opus comprimido em vez de FLAC sem perda: 90min cabem em ~16MB, abaixo do
    limite do Groq — a transcricao vai num unico request, sem fatiamento. O
    Whisper nao perde qualidade com voz nessa taxa.
    """
    audio_path = os.path.join(workdir, "audio.ogg")
    proc = _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", AUDIO_BITRATE, audio_path,
    ])
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        log.warning("Sem faixa de audio extraivel: %s", proc.stderr[-300:])
        return None
    return audio_path


def _slice_audio(audio_path: str, start: float, dur: float, workdir: str, idx: int) -> str:
    out = os.path.join(workdir, f"chunk_{idx:03d}.ogg")
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-y", "-ss", str(start), "-t", str(dur),
        "-i", audio_path, "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", AUDIO_BITRATE, out,
    ])
    return out


# ===========================================================================
# Transcricao
# ===========================================================================
def _audio_mean_volume(audio_path: str) -> "float | None":
    """Mede o volume medio do audio (dBFS) via ffmpeg volumedetect.

    Retorna None em qualquer falha — fail-safe: em caso de duvida, transcreve.
    """
    try:
        proc = _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-i", audio_path,
            "-af", "volumedetect", "-f", "null", "-",
        ], timeout=60)
        for line in proc.stderr.splitlines():
            if "mean_volume" in line:
                # "[Parsed_volumedetect_0 @ ...] mean_volume: -56.3 dBFS"
                return float(line.split("mean_volume:")[1].strip().split()[0])
    except Exception:
        pass
    return None


def _empty_transcript() -> dict:
    return {"text": "", "segments": [], "words": [], "engine": None}


def _norm(obj, key, default=None):
    """Le `key` de um item que pode ser dict ou objeto pydantic."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _transcribe_groq_file(client, path: str, offset: float = 0.0) -> dict:
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model=GROQ_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )
    segments = [
        {"start": float(_norm(s, "start", 0)) + offset,
         "end": float(_norm(s, "end", 0)) + offset,
         "text": (_norm(s, "text", "") or "").strip()}
        for s in (_norm(resp, "segments") or [])
    ]
    words = [
        {"start": float(_norm(w, "start", 0)) + offset,
         "end": float(_norm(w, "end", 0)) + offset,
         "word": _norm(w, "word", "")}
        for w in (_norm(resp, "words") or [])
    ]
    return {"text": (_norm(resp, "text", "") or "").strip(),
            "segments": segments, "words": words}


def _describe_groq_error(exc: Exception) -> tuple[str, str]:
    """(categoria, mensagem) para um erro da API Groq (nuvem, free tier limitado).

    categoria ∈ {conexao, limite, cobranca, auth, outro}. Em 'limite' (HTTP 429 do
    free tier) extrai o teto e o retry-after dos headers/corpo — para deixar claro
    que foi so estouro de cota, NAO falha. Conexao (sem status HTTP) e checada de
    forma a nao conflitar com limite/cobranca (que tem status).
    """
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    err = body.get("error", {}) if isinstance(body, dict) else {}
    code = str(err.get("code") or "").lower()
    emsg = (err.get("message") or getattr(exc, "message", "") or str(exc)).strip()
    headers = {}
    try:
        headers = {k.lower(): v for k, v in dict(getattr(exc, "response").headers).items()}
    except Exception:
        pass

    # CONEXAO: sem status HTTP => nao chegou no servidor (rede/DNS/timeout).
    if status is None and ("connection" in name or "timeout" in name
                           or _looks_like_connection_error(str(exc))):
        return "conexao", "sem conexao com a Groq (rede indisponivel ou timeout)"
    # COBRANCA: cota paga/credito esgotado — distinto do rate-limit do free tier.
    if status == 402 or "insufficient_quota" in code or "billing" in (code + emsg).lower():
        return "cobranca", f"cota/credito da Groq esgotado ({emsg})"
    # LIMITE: 429 do free tier — informa o teto, o restante e quando libera.
    if status == 429 or "ratelimit" in name or "rate_limit" in code:
        lim = (headers.get("x-ratelimit-limit-audio-seconds")
               or headers.get("x-ratelimit-limit-requests")
               or headers.get("x-ratelimit-limit-tokens"))
        rem = (headers.get("x-ratelimit-remaining-audio-seconds")
               or headers.get("x-ratelimit-remaining-requests")
               or headers.get("x-ratelimit-remaining-tokens"))
        retry = headers.get("retry-after")
        det = [p for p in (f"teto={lim}" if lim is not None else "",
                           f"restante={rem}" if rem is not None else "",
                           f"libera em ~{retry}s" if retry else "") if p]
        return "limite", "free tier da Groq atingido (" + ("; ".join(det) or emsg) + ")"
    # AUTH: chave invalida/sem permissao.
    if status in (401, 403) or "authentication" in name or "permission" in name:
        return "auth", f"GROQ_API_KEY invalida ou sem permissao ({emsg})"
    return "outro", emsg or f"falha na Groq ({type(exc).__name__})"


def transcribe_groq(audio_path: str, workdir: str) -> dict:
    """Transcreve via Groq, fatiando o audio se exceder o limite de tamanho."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    result = {"text": "", "segments": [], "words": [], "engine": "groq:" + GROQ_MODEL}

    if os.path.getsize(audio_path) <= GROQ_SIZE_LIMIT:
        part = _transcribe_groq_file(client, audio_path)
        result.update(text=part["text"], segments=part["segments"], words=part["words"])
        return result

    duration = get_duration(audio_path)
    log.info("Audio grande (%.1f min): fatiando em pedacos de %ds",
             duration / 60, CHUNK_SECONDS)
    texts: list[str] = []
    idx = 0
    start = 0.0
    while start < duration:
        chunk = _slice_audio(audio_path, start, CHUNK_SECONDS, workdir, idx)
        part = _transcribe_groq_file(client, chunk, offset=start)
        texts.append(part["text"])
        result["segments"].extend(part["segments"])
        result["words"].extend(part["words"])
        os.remove(chunk)
        start += CHUNK_SECONDS
        idx += 1
    result["text"] = " ".join(t for t in texts if t)
    return result


def transcribe_local(audio_path: str) -> dict:
    """Fallback local com faster-whisper (sem limite de tamanho, roda em CPU)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        # Caso do usuario sem chave Groq E sem o fallback local: em vez de um
        # "No module named 'faster_whisper'" assustador, diz exatamente o que fazer.
        raise RuntimeError(
            "Sem motor de transcricao disponivel. Faca UMA destas:\n"
            "  - Groq (recomendado, rapido e gratis): crie uma chave em "
            "https://console.groq.com/keys e coloque em GROQ_API_KEY (no arquivo .env); ou\n"
            "  - Local (offline): instale o faster-whisper com  "
            "pip install faster-whisper"
        ) from exc

    log.info("Transcrevendo localmente com faster-whisper '%s'", LOCAL_MODEL)
    try:
        model = WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")
    except Exception as exc:
        # O modelo e local apos a 1a vez, mas a 1a carga BAIXA os pesos (~140MB) do
        # Hugging Face. Sem conexao e sem cache, esse e o unico ponto que precisa de
        # rede no modo local — e nao ha outro fallback. Mensagem clara em vez de stack.
        if _looks_like_connection_error(str(exc)):
            raise RuntimeError(
                f"Sem conexao para baixar o modelo local '{LOCAL_MODEL}' (~140MB, "
                "Hugging Face) na primeira execucao. Conecte a internet uma vez para "
                "baixar o modelo, ou configure GROQ_API_KEY. Detalhe: " + str(exc)
            ) from exc
        raise
    seg_iter, _info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)

    segments, words, texts = [], [], []
    for seg in seg_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        texts.append(seg.text.strip())
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end, "word": w.word})
    return {"text": " ".join(texts), "segments": segments, "words": words,
            "engine": "faster-whisper:" + LOCAL_MODEL}


def transcribe(audio_path: str | None, workdir: str) -> dict:
    """Escolhe o motor de transcricao disponivel e normaliza a saida."""
    if audio_path is None:
        return _empty_transcript()
    vol = _audio_mean_volume(audio_path)
    if vol is not None and vol < SILENCE_THRESHOLD_DBFS:
        log.info(
            "Audio praticamente silencioso (mean_volume=%.1f dBFS < %.0f); "
            "pulando transcricao para evitar alucinacoes do Whisper.",
            vol, SILENCE_THRESHOLD_DBFS,
        )
        return _empty_transcript()
    if os.environ.get("GROQ_API_KEY"):
        try:
            return transcribe_groq(audio_path, workdir)
        except Exception as exc:
            cat, msg = _describe_groq_error(exc)
            if cat == "limite":
                # NAO e falha: so estourou o free tier. Informa o teto e degrada.
                log.warning("Groq: %s. Nao e erro — usando faster-whisper local.", msg)
            elif cat == "conexao":
                log.warning("Groq: %s. Transcrevendo offline (faster-whisper).", msg)
            elif cat == "cobranca":
                log.warning("Groq: %s. Usando faster-whisper local.", msg)
            elif cat == "auth":
                log.warning("Groq: %s. Usando faster-whisper local.", msg)
            else:
                log.warning("Groq falhou: %s. Usando faster-whisper local.", msg)
    return transcribe_local(audio_path)


# ===========================================================================
# Frames
# ===========================================================================
def extract_frame(video_path: str, time: float, out_path: str) -> bool:
    """Captura um unico frame em `time`, redimensionado para no maximo 1024px."""
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-y", "-ss", f"{time:.3f}", "-i", video_path,
        "-frames:v", "1", "-vf", f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
        "-q:v", "3", out_path,
    ])
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _sharpness(path: str) -> float:
    """Medida de foco (variancia do Laplaciano): alta = nitido, baixa = borrado/fade."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("L"), dtype="float32")
    if a.size < 9:
        return 0.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def pick_sharpest_time(video_path: str, lo: float, hi: float, step: float = 0.12) -> float:
    """Devolve o instante do frame mais NITIDO em [lo, hi] (so para marcas de cena).

    Dentro de um mesmo shot, escolhe o frame mais bem composto (texto solido / sem
    borrao) em vez do instante exato da mudanca — resolve overlays que entram em
    fade. Como os candidatos sao do MESMO conteudo, comparar nitidez e justo.
    """
    if hi <= lo:
        return max(0.0, lo)
    times: list[float] = []
    t = lo
    while t <= hi + 1e-6:
        times.append(round(t, 3))
        t += step
    best_t, best_s = times[len(times) // 2], -1.0
    tmp = os.path.join(tempfile.gettempdir(), f"vidtx_cand_{os.getpid()}.jpg")
    for t in times:
        if extract_frame(video_path, t, tmp) and (s := _sharpness(tmp)) > best_s:
            best_s, best_t = s, t
    try:
        os.remove(tmp)
    except OSError:
        pass
    return best_t


# ===========================================================================
# Formatacao da transcricao
# ===========================================================================
def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(transcript: dict, metadata: dict) -> str:
    lines: list[str] = []
    title = metadata.get("title") or "Video"
    lines.append(f"# {title}\n")

    if metadata.get("source"):
        lines.append(f"**Fonte:** {metadata['source']}")
    if metadata.get("duration"):
        lines.append(f"**Duracao:** {fmt_time(metadata['duration'])}")
    if transcript.get("engine"):
        lines.append(f"**Transcrito por:** {transcript['engine']}")
    lines.append("")

    chapters = metadata.get("chapters")
    if chapters:
        lines.append("## Capitulos")
        for ch in chapters:
            ts = fmt_time(ch.get("start_time", 0))
            lines.append(f"- [{ts}] {ch.get('title', '')}")
        lines.append("")

    lines.append("## Transcricao")
    if transcript.get("segments"):
        for seg in transcript["segments"]:
            lines.append(f"[{fmt_time(seg['start'])}] {seg['text']}")
    elif transcript.get("text"):
        lines.append(transcript["text"])
    else:
        lines.append("_(Sem audio falado — veja os prints abaixo.)_")
    return "\n".join(lines)


# ===========================================================================
# Cache persistente (artefato duravel + base para get_frame)
# ===========================================================================
def make_cache_id(title: str, source: str) -> str:
    """Id estavel e legivel: slug do titulo + hash curto da fonte.

    Publico: o launcher (server) cria o status com este id antes de o worker
    comecar, e o worker reaproveita o MESMO id (process(cache_id=...)) para que
    status, meta e result nunca fiquem sob ids divergentes.
    """
    slug = re.sub(r"[^\w\-]+", "_", (title or "video")).strip("_")[:40] or "video"
    h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{h}"


# --- Status do processamento em background ---------------------------------
# Fica num arquivo IRMAO da pasta do cache (que e recriada durante o processo,
# o que apagaria um status guardado dentro dela).
def _status_path(cache_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_id}.status.json")


def _log_path(cache_id: str) -> str:
    """Log por job (o worker grava aqui; vive fora da pasta, como o status)."""
    return os.path.join(CACHE_DIR, f"{cache_id}.log")


def write_status(cache_id: str, data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {**data, "cache_id": cache_id, "updated_at": time.time()}
    with open(_status_path(cache_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def read_status(cache_id: str) -> dict:
    path = _status_path(cache_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Sem status para '{cache_id}'. O processamento pode nao ter iniciado."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Resultado persistido (segments + frames) p/ montar a timeline depois ---
def _result_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "result.json")


def _save_result(cache_dir: str, segments: list, frames: list) -> None:
    with open(_result_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "frames": frames}, f, ensure_ascii=False)


def load_result(cache_id: str) -> dict:
    """Le segments + frames persistidos (para montar a timeline pos-background)."""
    path = _result_path(os.path.join(CACHE_DIR, cache_id))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resultado de '{cache_id}' ainda nao disponivel.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Job em subprocesso matavel (cancelamento real) -------------------------
def _merge_status(cache_id: str, **changes) -> None:
    """Atualiza so alguns campos do status, preservando o resto (ex.: pid/title)."""
    try:
        base = read_status(cache_id)
    except Exception:
        base = {}
    base.update(changes)
    write_status(cache_id, base)


def set_pid(cache_id: str, pid: int) -> None:
    """Registra o pid do processo do job (para o cancel saber quem matar)."""
    _merge_status(cache_id, pid=pid)


def mark_cancelled(cache_id: str) -> None:
    """Marca o job como cancelado (nao e erro)."""
    _merge_status(cache_id, stage="cancelled", detail="", done=True, error=None, pid=None)


def run_job(cache_id: str, source: str, max_screenshots: int = 20,
            do_ocr: bool = False, caption_backend: str | None = None) -> None:
    """Roda o pipeline atualizando o status. Entrypoint do worker em subprocesso.

    Atualiza o status em arquivo a cada etapa (o server le isso em get_status) e,
    ao final, marca done/error. Roda num processo proprio para poder ser morto
    pelo cancel sem afetar o server MCP.
    """
    def report(stage, detail="", frac=None, eta_stage=None, eta=None):
        _merge_status(cache_id, stage=stage, detail=detail, frac=frac,
                      eta_stage=eta_stage, eta=eta, eta_at=time.time(),
                      done=False, error=None)

    try:
        process(source, max_screenshots, cache_id=cache_id, progress=report,
                do_ocr=do_ocr, caption_backend=caption_backend)
        _merge_status(cache_id, stage="done", detail="", done=True, error=None, pid=None)
    except Exception as exc:
        log.exception("Job %s falhou", cache_id)
        _merge_status(cache_id, stage="error", detail="", done=True, error=str(exc), pid=None)


def kill_process_tree(pid: int) -> bool:
    """Mata o processo do job e seus filhos (ffmpeg/yt-dlp). True se enviou o kill.

    No Windows usa `taskkill /T` (mata a arvore inteira, incluindo os ffmpeg que o
    worker disparou). Em POSIX, mata o grupo de processos.
    """
    try:
        if os.name == "nt":
            r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, text=True,
                               creationflags=CREATE_NO_WINDOW)
            return r.returncode == 0
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            os.kill(pid, signal.SIGKILL)
        return True
    except Exception as exc:
        log.warning("Falha ao matar pid %s: %s", pid, exc)
        return False


def is_running(cache_id: str) -> bool:
    """True se ha um job ativo (status existe e nao terminou)."""
    try:
        st = read_status(cache_id)
    except Exception:
        return False
    return not st.get("done", True)


def pid_alive(pid) -> bool:
    """True se o processo `pid` ainda existe."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=CREATE_NO_WINDOW).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def reconcile_status(cache_id: str) -> dict | None:
    """Le o status e o conserta se o worker morreu sem finalizar.

    Sem isto, um worker que morre no meio (ex.: o server foi reciclado e levou o
    processo junto) deixaria o status preso "em andamento" para sempre. Aqui, se o
    job nao terminou mas o pid nao existe mais, marcamos como erro recuperavel.
    """
    try:
        st = read_status(cache_id)
    except Exception:
        return None
    if (not st.get("done")) and st.get("pid") and not pid_alive(st["pid"]):
        _merge_status(cache_id, stage="error", done=True, pid=None,
                      error="o processo de transcricao terminou inesperadamente; reprocesse")
        try:
            return read_status(cache_id)
        except Exception:
            return st
    return st


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.json")


def _save_meta(cache_dir: str, meta: dict) -> None:
    with open(_meta_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(cache_id: str) -> dict:
    """Le o meta.json de um video no cache (levanta se nao existir)."""
    path = _meta_path(os.path.join(CACHE_DIR, cache_id))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Video '{cache_id}' nao esta no cache. Reprocesse com process_video."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_artifact(cache_dir: str, transcript_text: str, frames: list[dict]) -> None:
    """Grava transcript.md com a transcricao e a lista de prints (caminhos relativos)."""
    lines = [transcript_text, "", "## Prints"]
    for fr in frames:
        rel = os.path.relpath(fr["path"], cache_dir).replace("\\", "/")
        lines.append(f"- [{fmt_time(fr['time'])}] ({fr['reason']}) {rel}")
        if fr.get("caption"):
            lines.append(f"    - descricao: {fr['caption']}")
        if fr.get("ocr"):
            lines.append(f"    - texto na tela: {fr['ocr']}")
    with open(os.path.join(cache_dir, "transcript.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _remove_job_sidecars(cache_id: str) -> None:
    """Remove os arquivos irmaos do cache (status/log) que vivem fora da pasta."""
    for path in (_status_path(cache_id), _log_path(cache_id)):
        try:
            os.remove(path)
        except OSError:
            pass


def _sweep_orphan_sidecars() -> None:
    """Apaga status/log de jobs FINALIZADOS cuja pasta de cache nao existe mais.

    Sem isto, podar uma pasta antiga (ou move-la a mao) deixava o
    `{id}.status.json`/`.log` orfao acumulando no diretorio _cache. So remove
    sidecar de job concluido: um job recem-enfileirado ainda nao criou a pasta.
    """
    if not os.path.isdir(CACHE_DIR):
        return
    for name in os.listdir(CACHE_DIR):
        if name.endswith(".status.json"):
            cid = name[: -len(".status.json")]
        elif name.endswith(".log"):
            cid = name[: -len(".log")]
        else:
            continue
        if os.path.isdir(os.path.join(CACHE_DIR, cid)):
            continue  # pasta existe -> sidecar valido
        try:
            st = read_status(cid)
        except Exception:
            st = None
        if st is None or st.get("done"):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
            except OSError:
                pass


def _prune_cache(max_entries: int = MAX_CACHE_ENTRIES) -> None:
    """Mantem apenas os `max_entries` videos mais recentes (por mtime)."""
    if not os.path.isdir(CACHE_DIR):
        return
    dirs = [os.path.join(CACHE_DIR, d) for d in os.listdir(CACHE_DIR)]
    dirs = [d for d in dirs if os.path.isdir(d)]
    dirs.sort(key=os.path.getmtime, reverse=True)
    for old in dirs[max_entries:]:
        shutil.rmtree(old, ignore_errors=True)
        _remove_job_sidecars(os.path.basename(old))
        log.info("Cache antigo removido: %s", os.path.basename(old))
    _sweep_orphan_sidecars()


def get_frame_from_cache(cache_id: str, timestamp: float) -> str:
    """Extrai um frame arbitrario do video cacheado. Retorna o caminho do JPG.

    Permite ao Claude "rebobinar" o video durante a conversa e olhar qualquer
    instante, mesmo que nao tenha virado print no processamento inicial.
    """
    meta = load_meta(cache_id)
    video_path = meta.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"O video de '{cache_id}' nao esta mais disponivel. Reprocesse com process_video."
        )
    out = os.path.join(tempfile.gettempdir(), f"vidtx_frame_{cache_id}_{int(timestamp * 1000)}.jpg")
    if not extract_frame(video_path, max(timestamp, 0.0), out):
        raise RuntimeError(f"Falha ao extrair frame em {timestamp:.1f}s")
    return out


# ===========================================================================
# Orquestracao
# ===========================================================================
def process(source: str, max_screenshots: int = 20, cache_id: str | None = None,
            progress=None, do_ocr: bool = False, caption_backend: str | None = None) -> dict:
    """Executa o pipeline completo e guarda o resultado num cache persistente.

    Retorna dados estruturados (metadados, segmentos da transcricao, prints e
    cache_id) para o server montar a timeline. O video e os prints ficam no
    cache para uso posterior por get_frame — so a area temporaria e limpa.

    `cache_id` (opcional): id ja emitido pelo server no fluxo background — quando
    presente e reusado para nao divergir do status; senao e calculado pelo titulo.
    `progress(stage, detail)` (opcional) e chamado a cada etapa para alimentar o
    acompanhamento em background (get_status).
    """
    # ETA: a duracao do video (escrita pelo server no status) permite estimar o
    # tempo de cada etapa pelas taxas (historico/_default). `rates` = s por s-de-video.
    video_secs = 0.0
    if cache_id:
        try:
            video_secs = float((read_status(cache_id) or {}).get("duration") or 0)
        except Exception:
            video_secs = 0.0
    rates = {**ETA_DEFAULT_RATES,
             **{k: v["rate"] for k, v in _load_timings().items()
                if isinstance(v, dict) and "rate" in v}}
    stage_start: dict[str, float] = {}

    def _eta(stage, frac, eta_cur):
        """(resto da etapa atual, resto total) em segundos — None,None sem duracao.

        Resto da etapa: ETA proprio (yt-dlp) > extrapolacao da fracao medida >
        taxa estimada. No estouro SEM sinal de progresso, nao crava 0 (nao mente
        "quase pronto"): assume um resto proporcional ao ja decorrido. O total
        soma o resto da etapa atual com a estimativa das etapas seguintes.
        """
        if not video_secs:
            return None, None
        elapsed = time.time() - stage_start.get(stage, time.time())
        if eta_cur is not None:           # etapa deu o proprio ETA (ex.: yt-dlp)
            stage_rem = max(0.0, float(eta_cur))
        elif frac and frac > 0:           # fracao medida -> extrapola pelo decorrido
            stage_rem = max(0.0, elapsed * (1.0 - frac) / frac)
        else:                             # so a taxa estimada
            est = rates.get(stage, 0.0) * video_secs
            stage_rem = est - elapsed if elapsed < est else max(3.0, 0.25 * elapsed)
        idx = ETA_STAGES.index(stage) if stage in ETA_STAGES else len(ETA_STAGES)
        future = sum(rates.get(s, 0.0) * video_secs for s in ETA_STAGES[idx + 1:])
        return stage_rem, stage_rem + future

    def _p(stage: str, detail: str = "", frac=None, eta_cur=None) -> None:
        now = time.time()
        if stage not in stage_start:
            seen = [s for s in ETA_STAGES if s in stage_start]
            if seen and video_secs:       # registra no historico a etapa que acabou
                _record_timing(seen[-1], now - stage_start[seen[-1]], video_secs)
            stage_start[stage] = now
        if progress:
            try:
                stage_rem, total_rem = _eta(stage, frac, eta_cur)
                progress(stage, detail, frac, stage_rem, total_rem)
            except Exception:  # progresso nunca pode derrubar o pipeline
                log.warning("Falha ao reportar progresso (%s)", stage)

    work = tempfile.mkdtemp(prefix="vidtx_")  # area temporaria (audio/chunks)
    try:
        _p("download", "baixando video e metadados")
        video_path, metadata = download_or_locate(
            source, work,
            progress=lambda fr, eta: _p("download", "baixando video", frac=fr, eta_cur=eta))
        if not video_secs:
            video_secs = float(metadata.get("duration") or 0)

        # Pasta de cache legivel; reprocessar a mesma fonte sobrescreve. Reusa o
        # id emitido pelo server (fluxo background) para casar com o status; so
        # calcula pelo titulo quando chamado standalone (cache_id ausente).
        if not cache_id:
            cache_id = make_cache_id(metadata.get("title", "video"), source)
        cache_dir = os.path.join(CACHE_DIR, cache_id)
        frames_dir = os.path.join(cache_dir, "frames")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)
        if _is_url(source):
            cached = os.path.join(cache_dir, "video" + os.path.splitext(video_path)[1])
            shutil.move(video_path, cached)
            video_path = cached  # arquivos locais permanecem no lugar original

        # Validacao precoce: o arquivo abre como video? (ffprobe da a duracao). Se
        # nao — corrompido, truncado ou nao e video — falha JA com mensagem clara,
        # antes de rodar audio/deteccao no lixo (o que ainda jorraria erros do ffmpeg).
        duration = float(metadata.get("duration") or 0) or get_duration(video_path)
        if duration <= 0:
            raise RuntimeError(
                "O arquivo nao parece ser um video valido (corrompido, incompleto ou "
                f"formato nao suportado): {source}. Tente outro arquivo ou baixe de novo."
            )

        _p("audio", "extraindo audio (opus)")
        audio_path = extract_audio(video_path, work)
        _p("transcribe", "transcrevendo a fala")
        transcript = transcribe(audio_path, work)
        if not video_secs:
            video_secs = float(duration or 0)

        # Cinco fontes de marcas; merge por prioridade capitulo>cena>edicao>fala>audio.
        # As tres passadas de ffmpeg (cena e edicao no video, onsets no audio) sao
        # independentes e subprocess-bound: rodam em paralelo (as threads liberam
        # o GIL enquanto o ffmpeg trabalha), cortando o tempo desta etapa.
        _p("deteccao", "detectando cenas, edicao e audio")
        # Sinal de progresso honesto da deteccao: o detector de edicao (a passada
        # mais lenta das tres) reporta quantos segundos do video ja varreu. Como
        # roda em thread enquanto o principal espera em .result(), so essa thread
        # escreve o status aqui — sem corrida.
        det_total = float(duration or video_secs or 0)

        def _det_progress(done_secs):
            if det_total:
                _p("deteccao", "detectando cenas, edicao e audio",
                   frac=max(0.0, min(1.0, done_secs / det_total)))

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_scenes = ex.submit(fs.detect_scenes, video_path)
            f_edits = ex.submit(fs.detect_edits, video_path, progress=_det_progress)
            f_audio = ex.submit(fs.detect_audio_events, audio_path)
            scenes, edits, audio_events = f_scenes.result(), f_edits.result(), f_audio.result()
        smart = fs.merge(
            fs.from_words(transcript["words"]),
            scenes,
            edits,
            audio_events,
            fs.from_chapters(metadata.get("chapters")),
        )
        # Budget de prints proporcional a duracao: 1 a 12 por minuto de video.
        budget = fs.screenshot_budget(max_screenshots, duration)
        marks = fs.fill_to_budget(smart, duration, budget)

        extracted: list[tuple[fs.FrameMark, str]] = []
        n_marks = len(marks)
        step = max(1, n_marks // 40)  # nao escreve status a cada frame em videos longos
        # Etapa propria: aqui a fracao i/n e um sinal honesto (1 frame por marca),
        # ao contrario da deteccao. O ETA da extracao extrapola desse progresso real.
        _p("extracao", f"capturando prints (0/{n_marks})", frac=0.0)
        for i, mark in enumerate(marks):
            # Marcas de cena: escolhe o frame mais nitido na janela do shot (evita
            # meio-fade/borrao); as demais sao capturadas no proprio instante.
            if mark.window:
                mark.time = pick_sharpest_time(video_path, mark.window[0], mark.window[1])
            name = f"{i:03d}_{fmt_time(mark.time).replace(':', '-')}_{mark.reason}.jpg"
            out = os.path.join(frames_dir, name)
            if extract_frame(video_path, mark.time, out):
                extracted.append((mark, out))
            if i % step == 0 or i == n_marks - 1:
                _p("extracao", f"capturando prints ({i + 1}/{n_marks})",
                   frac=(i + 1) / max(1, n_marks))
        extracted = fs.drop_blank(extracted)
        frames = fs.dedup_visual(extracted)
        if video_secs and "extracao" in stage_start:
            _record_timing("extracao", time.time() - stage_start["extracao"], video_secs)

        frame_list = [{"time": m.time, "reason": m.reason, "path": p} for m, p in frames]

        # Anotacao OPCIONAL dos prints (OCR + descricao), so nos frames que sobraram.
        # Etapa fora do modelo de ETA (e condicional): reporta andamento direto, com
        # ETA simples por fracao i/n. Dependencia ausente -> pulada com aviso (fa).
        # `caption_backend`: None | 'florence' (local) | 'groq' (nuvem) — a escolha
        # ja vem resolvida do chamador (CLI mede a RAM/pede permissao; server resolve).
        if (do_ocr or caption_backend) and frame_list:
            an_start = time.time()
            n_fr = len(frame_list)
            cap_lbl = {"florence": "descricao(local)", "groq": "descricao(nuvem)"}.get(caption_backend)
            label = " + ".join(p for p in ("OCR" if do_ocr else "", cap_lbl or "") if p)

            def _an_prog(i, n):
                if not progress:
                    return
                el = time.time() - an_start
                frac = i / max(1, n)
                rem = el * (1 - frac) / frac if frac > 0 else None
                try:
                    progress("anotacao", f"{label} dos prints ({i}/{n})", frac, rem, rem)
                except Exception:
                    pass

            fa.annotate_frames(frame_list, do_ocr=do_ocr, caption_backend=caption_backend,
                               progress=_an_prog)

        transcript_text = format_transcript(transcript, metadata)

        # Persiste artefato + metadados (B) e aplica limpeza LRU do cache.
        _save_artifact(cache_dir, transcript_text, frame_list)
        _save_result(cache_dir, transcript["segments"], frame_list)
        _save_meta(cache_dir, {
            "cache_id": cache_id,
            "version": __version__,
            "title": metadata.get("title"),
            "source": metadata.get("source"),
            "duration": duration,
            "engine": transcript.get("engine"),
            "video_path": os.path.abspath(video_path),
            "chapters": metadata.get("chapters"),
        })
        _prune_cache()
        _p("done", "")

        return {
            "cache_id": cache_id,
            "cache_dir": cache_dir,
            "metadata": {**metadata, "duration": duration, "engine": transcript.get("engine")},
            "segments": transcript["segments"],
            "transcript_text": transcript_text,
            "frames": frame_list,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)

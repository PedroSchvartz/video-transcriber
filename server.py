"""
MCP server: video-transcriber.

Ferramentas:
  - process_video(source, max_screenshots): transcreve um video (YouTube,
    Instagram ou .mp4 local) e devolve uma TIMELINE unificada — a fala e os
    prints intercalados em ordem cronologica. O video fica em cache.
  - get_frame(cache_id, timestamp): captura qualquer instante de um video ja
    processado, para "rebobinar" e olhar um momento especifico na conversa.

Rodar:  python tools/video_transcriber/server.py
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import time

# Logging vai para stderr — stdout e o canal do protocolo MCP (stdio).
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("video_transcriber.server")

# Garante que os modulos irmaos sejam importaveis ao rodar como script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import processor  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ImageContent, TextContent  # noqa: E402

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

mcp = FastMCP(
    "video-transcriber",
    instructions=(
        "Fluxo: process_video(source) INICIA a transcricao em background e retorna na "
        "hora um cache_id (nao trava em videos longos). Acompanhe com "
        "get_status(cache_id) e, quando estiver pronto, pegue a timeline (fala + prints "
        "intercalados) com get_result(cache_id) — analise os prints visualmente ao "
        "responder. cancel(cache_id) interrompe um processamento em andamento. Use "
        "get_frame(cache_id, timestamp) para olhar qualquer outro instante do video "
        "depois (ex.: responder sobre um momento sem print)."
    ),
)


def _img_block(path: str) -> ImageContent:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return ImageContent(type="image", data=data, mimeType="image/jpeg")


# Teto de saida de uma ferramenta MCP fica em ~25k tokens; reservamos a maior parte
# para imagens e deixamos folga para o texto da transcricao.
PAGE_IMAGE_BUDGET = 18000


def _frame_path(cache_dir: str, fr: dict) -> str:
    """Reconstroi o caminho do print a partir do cache_dir — ignora paths absolutos
    antigos que possam ter ficado no result.json (cache movido/renomeado)."""
    return os.path.join(cache_dir, "frames", os.path.basename(fr["path"]))


def _img_tokens(path: str) -> int:
    """Estimativa de quantos tokens a imagem custa no contexto (~w*h/750, teto ~4784)."""
    try:
        from PIL import Image
        w, h = Image.open(path).size
        return min(4784, max(1, w * h // 750))
    except Exception:
        return 2000


def _fmt_eta(secs: float) -> str:
    """Formata segundos como ~45s / ~3min / ~1h02min."""
    secs = max(0, int(secs))
    if secs < 60:
        return f"~{secs}s"
    m = secs // 60
    if m < 60:
        return f"~{m}min"
    return f"~{m // 60}h{m % 60:02d}min"


def _parse_ts(ts: str) -> float:
    """Aceita 'mm:ss', 'hh:mm:ss' ou segundos ('83', '83.5')."""
    ts = str(ts).strip()
    if ":" in ts:
        secs = 0.0
        for part in ts.split(":"):
            secs = secs * 60 + float(part or 0)
        return secs
    return float(ts)


def _header(meta: dict, cache_id: str, n_frames: int) -> str:
    lines = [f"# {meta.get('title') or 'Video'}", ""]
    if meta.get("source"):
        lines.append(f"**Fonte:** {meta['source']}")
    if meta.get("duration"):
        lines.append(f"**Duracao:** {processor.fmt_time(meta['duration'])}")
    if meta.get("engine"):
        lines.append(f"**Transcrito por:** {meta['engine']}")
    if meta.get("version"):
        lines.append(f"**Transcriber:** v{meta['version']}")
    lines.append(f"**cache_id:** `{cache_id}` — {n_frames} print(s)")
    chapters = meta.get("chapters")
    if chapters:
        lines.append("\n## Capitulos")
        for c in chapters:
            lines.append(f"- [{processor.fmt_time(c.get('start_time', 0))}] {c.get('title', '')}")
    lines.append("\n## Timeline")
    return "\n".join(lines)


def _timeline_blocks(meta: dict, segments: list, frames: list,
                     cache_id: str, cache_dir: str,
                     start: int = 0, count: int = 0) -> list:
    """Monta a timeline (fala + prints) PAGINADA por prints.

    A timeline inteira pode passar do teto de saida da ferramenta MCP (~25k tokens)
    em videos com muitos prints. Entao limitamos os prints por chamada:
      - `count <= 0`: automatico — o maximo de prints que cabe no orcamento.
      - `count > 0`: exatamente esse numero, a partir de `start`.
    A transcricao e janelada por tempo (so os segmentos do trecho desta pagina), de
    modo que nada se repete entre paginas nem estoura o limite.
    """
    n = len(frames)
    start = max(0, start)
    if n and start >= n:
        return [TextContent(type="text",
                            text=f"start={start} fora do alcance (sao {n} print(s)).")]

    # Auto-dimensiona a pagina pelo custo estimado de cada print (todos mesma resolucao).
    if count <= 0:
        per = _img_tokens(_frame_path(cache_dir, frames[start])) if frames else 1
        count = max(1, PAGE_IMAGE_BUDGET // max(per, 1))
    page = frames[start:start + count]
    end = start + len(page)

    # Janela de tempo desta pagina: do 1o print (ou 0 no inicio) ate o 1o print da
    # proxima — particiona transcricao e prints sem sobreposicao nem buraco.
    win_lo = 0.0 if start == 0 else float(frames[start]["time"])
    win_hi = float(frames[end]["time"]) if end < n else float("inf")

    events: list[tuple[float, str, object]] = []
    for seg in segments:
        s = float(seg["start"])
        if win_lo <= s < win_hi:
            events.append((s, "text", f"[{processor.fmt_time(s)}] {seg['text']}"))
    for fr in page:
        events.append((float(fr["time"]), "image", fr))
    events.sort(key=lambda e: e[0])

    blocks: list = []
    if start == 0:
        blocks.append(TextContent(type="text", text=_header(meta, cache_id, n)))
    buf: list[str] = []

    def flush():
        if buf:
            blocks.append(TextContent(type="text", text="\n".join(buf)))
            buf.clear()

    for t, kind, payload in events:
        if kind == "text":
            buf.append(payload)
        else:
            buf.append(f"[{processor.fmt_time(t)}] --- print ({payload['reason']}) ---")
            if payload.get("caption"):
                buf.append(f"    descricao: {payload['caption']}")
            if payload.get("ocr"):
                buf.append(f"    texto na tela: {payload['ocr']}")
            flush()
            blocks.append(_img_block(_frame_path(cache_dir, payload)))
    flush()

    if n == 0:
        blocks.append(TextContent(type="text", text="_(Sem prints extraidos.)_"))
        return blocks

    if end < n:
        footer = (f'\n_Prints {start + 1}–{end} de {n}. Para os proximos: '
                  f'get_result(cache_id="{cache_id}", start={end})._')
    else:
        footer = (f'\n_Prints {start + 1}–{end} de {n} (fim). Para outro instante: '
                  f'get_frame(cache_id="{cache_id}", timestamp="mm:ss")._')
    blocks.append(TextContent(type="text", text=footer))
    return blocks


def _spawn_worker(cmd: list) -> "subprocess.Popen":
    """Lanca o worker DESTACADO do processo do server.

    Critico no Windows: o Claude Code roda o server MCP e o recicla entre
    interacoes; um filho comum morre junto (foi o bug observado — o worker morria
    calado no meio do job). DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP desacopla
    o worker, e CREATE_BREAKAWAY_FROM_JOB o tira do Job Object do server. Se o job
    nao permitir breakaway, cai no modo destacado simples. O cancel continua
    funcionando (mata por pid, via taskkill /T).
    """
    kw = dict(
        cwd=os.path.dirname(WORKER_SCRIPT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    if os.name != "nt":
        return subprocess.Popen(cmd, start_new_session=True, **kw)

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        return subprocess.Popen(cmd, creationflags=base | CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError:
        log.warning("Breakaway do Job Object negado; usando processo destacado simples")
        return subprocess.Popen(cmd, creationflags=base, **kw)


def _resolve_describe_backend(describe: bool, prefer: str) -> str | None:
    """Resolve o backend da descricao SEM interacao (o server nao pode perguntar).

    'auto' aqui PREFERE a nuvem (Groq) — o padrao e nao rodar o modelo local pesado
    na maquina do usuario sem permissao, e no MCP nao da pra pedi-la. Local so quando
    pedido explicitamente E com RAM folgada; senao degrada para a nuvem (se houver
    chave) ou pula. Retorna 'florence' | 'groq' | None.
    """
    if not describe:
        return None
    import frame_annotators as fa
    has_groq = fa.groq_vision_available()
    if prefer == "groq":
        return "groq" if has_groq else None
    if prefer == "local":
        tier, _, _ = fa.caption_local_readiness()
        return "florence" if tier == "ok" else ("groq" if has_groq else None)
    # auto: nuvem primeiro; senao local so se folgado
    if has_groq:
        return "groq"
    tier, _, _ = fa.caption_local_readiness()
    return "florence" if tier == "ok" else None


@mcp.tool()
def process_video(source: str, max_screenshots: int = 20, ocr: bool = False,
                  describe: bool = False, describe_backend: str = "auto") -> list:
    """Inicia a transcricao de um video em BACKGROUND e retorna na hora.

    O processamento pesado (download, transcricao, prints) roda num processo
    destacado e reporta progresso — isso evita travar a sessao em videos longos
    (uma chamada sincrona estouraria o timeout do MCP num video de 1h+) e o
    processo sobrevive ao reciclo do server. Acompanhe com get_status(cache_id)
    e, ao concluir, pegue a timeline com get_result(cache_id).

    Args:
        source: URL do YouTube/Instagram ou caminho local para um arquivo .mp4.
        max_screenshots: numero maximo de prints a capturar (padrao 20).
        ocr: se True, extrai o texto na tela de cada print (RapidOCR, local). Em
            geral desnecessario quando o Claude ja le as imagens; util p/ tornar a
            timeline pesquisavel.
        describe: se True, gera uma descricao de cada print. Idem: o Claude ja
            enxerga os prints — util sobretudo p/ deixar a timeline pesquisavel.
        describe_backend: 'auto' (padrao — prefere a NUVEM Groq; nao roda o modelo
            local pesado sem permissao), 'groq' (nuvem) ou 'local' (Florence-2; so
            se a RAM estiver folgada, senao cai p/ a nuvem).
    """
    caption_backend = _resolve_describe_backend(describe, describe_backend)
    log.info("process_video: source=%s max_screenshots=%s ocr=%s describe=%s backend=%s",
             source, max_screenshots, ocr, describe, caption_backend)
    try:
        meta = processor.probe_source(source)
    except Exception as exc:
        log.exception("Falha ao ler metadados do video")
        return [TextContent(type="text", text=f"Erro ao ler o video: {exc}")]

    duration = meta.get("duration") or 0
    if duration and duration > processor.MAX_DURATION_HARD:
        return [TextContent(type="text", text=(
            f"Video muito longo ({processor.fmt_time(duration)}). Limite de seguranca: "
            f"{processor.fmt_time(processor.MAX_DURATION_HARD)}. Corte um trecho antes."
        ))]

    title = meta.get("title") or "video"
    cache_id = processor.make_cache_id(title, source)

    # Concorrencia: se ja ha um job ATIVO para este mesmo video, nao lanca outro
    # worker (sobrescreveria o cache em andamento). reconcile_status conserta um
    # status preso de um worker ja morto, entao isto so barra jobs realmente vivos.
    existing = processor.reconcile_status(cache_id)
    if existing and not existing.get("done"):
        return [TextContent(type="text", text=(
            f"Esse video ja esta sendo processado (cache_id `{cache_id}`).\n"
            f'Acompanhe com get_status(cache_id="{cache_id}").'
        ))]

    processor.write_status(cache_id, {
        "stage": "queued", "detail": "iniciando", "title": title,
        "duration": duration, "done": False, "error": None, "pid": None,
    })

    # Processo SEPARADO e DESTACADO (nao thread, nao filho comum): sobrevive ao
    # reciclo do server pelo Claude Code e pode ser morto pelo cancel junto dos
    # ffmpeg/yt-dlp filhos. O status em arquivo serve de IPC: o worker escreve, o
    # server le em get_status/get_result.
    try:
        proc = _spawn_worker([sys.executable, WORKER_SCRIPT, cache_id, source,
                              str(max_screenshots), "1" if ocr else "0",
                              caption_backend or ""])
    except Exception as exc:
        log.exception("Falha ao lancar o worker")
        processor._merge_status(cache_id, stage="error", done=True, error=str(exc), pid=None)
        return [TextContent(type="text", text=f"Erro ao iniciar o processamento: {exc}")]

    processor.set_pid(cache_id, proc.pid)

    dur_txt = processor.fmt_time(duration) if duration else "desconhecida"
    return [TextContent(type="text", text=(
        f"Processando **{title}** (duracao {dur_txt}) em background.\n"
        f"**cache_id:** `{cache_id}`\n\n"
        f'Acompanhe com get_status(cache_id="{cache_id}"); ao concluir, pegue a '
        f'timeline com get_result(cache_id="{cache_id}"). Para interromper, '
        f'cancel(cache_id="{cache_id}").'
    ))]


@mcp.tool()
def cancel(cache_id: str) -> list:
    """Interrompe um processamento em andamento (mata o processo e os ffmpeg filhos).

    Args:
        cache_id: identificador retornado por process_video.
    """
    log.info("cancel: cache_id=%s", cache_id)
    try:
        st = processor.read_status(cache_id)
    except Exception as exc:
        return [TextContent(type="text", text=f"Erro: {exc}")]

    if st.get("done"):
        return [TextContent(type="text",
                            text=f"Nada a cancelar — o job ja finalizou ({st.get('stage')}).")]
    pid = st.get("pid")
    if not pid:
        return [TextContent(type="text",
                            text="O job ainda esta inicializando; tente de novo em alguns instantes.")]

    ok = processor.kill_process_tree(int(pid))
    processor.mark_cancelled(cache_id)
    msg = ("Processamento cancelado." if ok else
           "Pedi o cancelamento, mas o kill nao confirmou (talvez ja tivesse terminado).")
    return [TextContent(type="text", text=msg)]


@mcp.tool()
def get_status(cache_id: str) -> list:
    """Estado atual do processamento iniciado por process_video.

    Args:
        cache_id: identificador retornado por process_video.
    """
    try:
        st = processor.reconcile_status(cache_id)
        if st is None:
            raise FileNotFoundError(f"Sem status para '{cache_id}'. O job pode nao ter iniciado.")
    except Exception as exc:
        return [TextContent(type="text", text=f"Erro: {exc}")]

    stage = st.get("stage")
    if stage == "cancelled":
        msg = "Cancelado pelo usuario."
    elif st.get("error"):
        msg = f"Falhou em '{stage}': {st['error']}"
    elif st.get("done"):
        msg = f'Pronto. Use get_result(cache_id="{cache_id}") para a timeline.'
    else:
        labels = {
            "queued": "Na fila", "download": "Baixando video",
            "audio": "Extraindo audio", "transcribe": "Transcrevendo a fala",
            "deteccao": "Detectando cenas e edicoes", "extracao": "Capturando prints",
            "anotacao": "Anotando prints (OCR/descricao)",
            "frames": "Capturando prints",  # compat com jobs/caches antigos
        }
        label = labels.get(stage, stage)
        detail = st.get("detail")
        msg = f"{label}" + (f" — {detail}" if detail else "") + "..."
        # Progresso: fracao da etapa (quando medida) + ETA da etapa atual e total,
        # ambos descontando o tempo decorrido desde a ultima medicao (eta_at).
        frac = st.get("frac")
        if isinstance(frac, (int, float)):
            msg += f" {int(max(0, min(1, frac)) * 100)}%"
        eta_at = st.get("eta_at")
        if isinstance(eta_at, (int, float)):
            drift = time.time() - eta_at
            eta_stage, eta = st.get("eta_stage"), st.get("eta")
            parts = []
            if isinstance(eta_stage, (int, float)):
                parts.append(f"etapa {_fmt_eta(eta_stage - drift)}")
            if isinstance(eta, (int, float)):
                parts.append(f"total {_fmt_eta(eta - drift)}")
            if parts:
                msg += " · " + " · ".join(parts) + " restantes"
    return [TextContent(type="text", text=f"[{st.get('title', 'video')}] {msg}")]


@mcp.tool()
def get_result(cache_id: str, start: int = 0, count: int = 0) -> list:
    """Monta a timeline (fala + prints) PAGINADA de um video ja processado.

    Em videos com muitos prints a timeline inteira passa do limite de saida da
    ferramenta (~25k tokens), entao ela vem em paginas de prints. O rodape de cada
    pagina informa "prints X–Y de N" e o `start` da proxima.

    Args:
        cache_id: identificador retornado por process_video.
        start: indice do primeiro print desta pagina (0 = comeco).
        count: quantos prints nesta chamada. 0 (padrao) = automatico (o maximo que
            cabe no limite). Informe um numero para controlar manualmente.
    """
    st = processor.reconcile_status(cache_id)
    if st and st.get("stage") == "cancelled":
        return [TextContent(type="text", text=(
            "Esse processamento foi cancelado. Rode process_video de novo para refazer."
        ))]
    if st and st.get("error"):
        return [TextContent(type="text", text=f"O processamento falhou: {st['error']}")]
    if st and not st.get("done"):
        return [TextContent(type="text", text=(
            f"Ainda processando ({st.get('stage')}). Use get_status para acompanhar."
        ))]

    try:
        meta = processor.load_meta(cache_id)
        res = processor.load_result(cache_id)
    except Exception as exc:
        return [TextContent(type="text", text=f"Erro: {exc}")]

    cache_dir = os.path.join(processor.CACHE_DIR, cache_id)
    return _timeline_blocks(meta, res["segments"], res["frames"], cache_id, cache_dir,
                            start=start, count=count)


@mcp.tool()
def get_frame(cache_id: str, timestamp: str) -> list:
    """Captura um instante especifico de um video ja processado.

    Use para "rebobinar" um video e olhar um momento que nao virou print —
    por exemplo, ao responder uma pergunta sobre algo que acontece num segundo
    especifico. O cache_id vem da resposta de process_video.

    Args:
        cache_id: identificador retornado por process_video.
        timestamp: instante desejado em 'mm:ss', 'hh:mm:ss' ou segundos.
    """
    log.info("get_frame: cache_id=%s timestamp=%s", cache_id, timestamp)
    try:
        secs = _parse_ts(timestamp)
        path = processor.get_frame_from_cache(cache_id, secs)
    except Exception as exc:
        return [TextContent(type="text", text=f"Erro: {exc}")]

    block = _img_block(path)
    try:
        os.remove(path)
    except OSError:
        pass
    return [
        TextContent(type="text", text=f"Frame de `{cache_id}` em [{processor.fmt_time(secs)}]:"),
        block,
    ]


if __name__ == "__main__":
    mcp.run()

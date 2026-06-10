"""
CLI para usar o Video Transcriber SEM IA (sem MCP / sem Claude).

Roda o pipeline inteiro de forma sincrona no terminal e mostra o progresso. Ao
final, a transcricao (`transcript.md`) e os prints ficam no cache do video. Veja
USO_SEM_IA.md para o passo a passo completo.

Uso:
    python transcribe.py "<URL do YouTube/Instagram ou caminho .mp4>"
    python transcribe.py "video.mp4" --screenshots 30
    python transcribe.py                 # sem argumento: pergunta a fonte no terminal
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Garante que os modulos irmaos sejam importaveis ao rodar como script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import processor  # noqa: E402

# Logs do pipeline (engine de transcricao, limiar de cena auto-calibrado, dedup)
# direto no terminal - util para acompanhar o que esta acontecendo. Vao para o
# stdout (mesma corrente dos prints de progresso) p/ sair em ordem cronologica.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silencia "HTTP Request: POST..."

_LABELS = {
    "download": "Baixando video",
    "audio": "Extraindo audio",
    "transcribe": "Transcrevendo a fala",
    "deteccao": "Detectando cenas e edicoes",
    "extracao": "Capturando prints",
    "anotacao": "Anotando prints (OCR/descricao)",
    "frames": "Detectando cenas e capturando prints",  # compat
    "done": "Finalizando",
}

_last_label = None


def _fmt(secs) -> str:
    if not isinstance(secs, (int, float)) or secs < 0:
        return ""
    secs = int(secs)
    if secs < 60:
        return f"~{secs}s"
    m = secs // 60
    return f"~{m}min" if m < 60 else f"~{m // 60}h{m % 60:02d}min"


def _progress(stage: str, detail: str = "", frac=None,
              eta_stage=None, eta=None) -> None:
    """Recebe (etapa, detalhe, fracao, resto-da-etapa, resto-total) do pipeline.

    Imprime o cabecalho da etapa so quando ela muda (evita spam) e, nas chamadas
    seguintes da mesma etapa, uma linha curta com fracao + ETA da etapa e total.
    """
    global _last_label
    label = _LABELS.get(stage, stage)
    if label != _last_label:
        _last_label = label
        print(f"  -> {label}" + (f" ({detail})" if detail else "") + "...")
    bits = []
    if isinstance(frac, (int, float)):
        bits.append(f"{int(max(0, min(1, frac)) * 100)}%")
    if _fmt(eta_stage):
        bits.append(f"etapa {_fmt(eta_stage)}")
    if _fmt(eta):
        bits.append(f"total {_fmt(eta)}")
    if bits:
        print("     " + " | ".join(bits) + " restantes")


def force_input(prompt: str) -> str:
    """Pede uma entrada ao usuario e repete ate vir algo nao-vazio.

    Remove espacos e aspas em volta (paths colados do Explorer vem entre aspas).
    Em EOF/Ctrl-C (pipe fechado, sem terminal) sai limpo em vez de loopar.
    """
    while True:
        try:
            ans = input(prompt).strip().strip('"').strip("'").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # quebra a linha pendente do prompt / do ^C
            sys.exit(1)
        if ans:
            return ans
        print("  (entrada vazia - cole a URL ou o caminho do arquivo)")


def _ask_screenshots(default: int) -> int:
    """Pergunta quantos prints (Enter = padrao). Valor invalido cai no padrao."""
    try:
        raw = input(f"Quantos prints? [Enter = {default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  valor invalido; usando {default}")
        return default


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Pergunta sim/nao. Enter = padrao. EOF/Ctrl-C = padrao."""
    suffix = " [s/N]: " if not default else " [S/n]: "
    try:
        raw = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    return raw[0] in ("s", "y")


def _print_services(source: str, do_ocr: bool = False, caption_backend=None) -> None:
    """Lista os servicos que ESTA execucao vai tocar e o custo de cada um.

    E sensivel ao contexto: download so aparece para URL (arquivo local nao baixa
    nada), e a transcricao mostra Groq (nuvem) ou faster-whisper (local) conforme
    haja GROQ_API_KEY. Assim o usuario sabe, antes de comecar, o que sai da maquina.
    """
    is_url = processor._is_url(source)
    is_ig = "instagram.com" in source.lower()
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    print("Servicos que esta execucao acessa:")
    if is_url:
        print("  - yt-dlp -> YouTube/Instagram/etc .......... GRATUITO (local, mas BAIXA "
              "da internet; sujeito a bloqueio/limite da plataforma)")
        # Conteudo logado (tipicamente Instagram privado/reel) cai no fallback que
        # le os cookies do Chrome - acesso local ao navegador, download autenticado.
        print("       \\ se exigir login" + (" (provavel: e Instagram)" if is_ig else
              " (ex.: Instagram privado)") + ", le os COOKIES do seu Chrome logado")
        print("         (acesso LOCAL ao navegador) e baixa AUTENTICADO COMO VOCE - so "
              "com Chrome + sessao ativa no site")
    else:
        print("  - download .................................. NAO USADO (fonte e "
              "arquivo local; nada sai/entra pela rede aqui)")
    print("  - ffmpeg / ffprobe (audio, frames, deteccao) LOCAL (offline, gratuito)")
    if has_groq:
        print("  - Groq Whisper API (api.groq.com) .......... GRATUITO porem LIMITADO "
              "(free tier: envia o audio para a nuvem; ha limite de uso por minuto/dia)")
        print("       \\ em erro/limite, cai automaticamente para o faster-whisper local")
    else:
        print("  - faster-whisper, modelo 'base' ............ LOCAL (offline, gratuito; "
              "sem GROQ_API_KEY). Na 1a vez baixa ~140MB do modelo (huggingface.co)")
    if do_ocr:
        print("  - RapidOCR (texto na tela dos prints) ...... LOCAL (offline, gratuito; ONNX)")
    if caption_backend == "florence":
        print("  - Florence-2 (descricao dos prints) ........ LOCAL (offline, gratuito; MIT). "
              "Na 1a vez baixa ~0.5GB do modelo (huggingface.co)")
    elif caption_backend == "groq":
        print("  - Groq vision / Llama-4 (descricao) ........ GRATUITO porem LIMITADO "
              "(free tier: envia cada print para a nuvem; 1 requisicao por print)")
    print("  (nenhum servico pago: o unico de nuvem e o Groq, que tem free tier)\n")


def _resolve_caption_backend(want: bool, prefer: str) -> "str | None":
    """Decide o backend da descricao e imprime o motivo. Retorna 'florence'|'groq'|None.

    Regra (CLI, interativa): mede a RAM. >= OK roda LOCAL; entre MIN e OK pede
    PERMISSAO (cabe apertado); < MIN (ou torch quebrado) nao roda local e vai pra
    NUVEM (Groq). 'prefer' force: 'local' nunca cai pra nuvem sozinho; 'online' vai
    direto pro Groq.
    """
    if not want:
        return None
    import frame_annotators as fa
    has_groq = fa.groq_vision_available()

    def _online(reason: str):
        if has_groq:
            print(f"  - Descricao: ONLINE (Groq vision) - {reason}.")
            return "groq"
        print(f"  - Descricao: SERA PULADA - {reason}, e sem GROQ_API_KEY p/ a nuvem.")
        return None

    if prefer == "online":
        return _online("backend online escolhido")

    tier, ram, why = fa.caption_local_readiness()
    if tier == "ok":
        print(f"  - Descricao: LOCAL (Florence-2) - {why}.")
        return "florence"
    if tier == "tight":
        if _ask_yes_no(f"  RAM apertada ({why}). Rodar a descricao LOCAL mesmo assim "
                       "(pode deixar a maquina lenta)?", default=False):
            print("  - Descricao: LOCAL (Florence-2) - autorizado com RAM apertada.")
            return "florence"
        if prefer == "local":
            print("  - Descricao: SERA PULADA (local recusado; backend 'local' nao usa nuvem).")
            return None
        return _online("RAM apertada, direcionado para a nuvem")
    # tier == 'low' ou 'no_torch'
    if prefer == "local":
        print(f"  - Descricao: SERA PULADA - local indisponivel ({why}).")
        return None
    return _online(f"nao da pra rodar local ({why})")


def _setup_annotation(do_ocr: bool, want_describe: bool, backend_pref: str) -> tuple:
    """Confere deps + decide os anotadores ANTES do trabalho pesado.

    Retorna (do_ocr_efetivo, caption_backend). Pode perguntar (RAM apertada). Assim
    o usuario sabe/decide na hora, em vez de esperar download+transcricao p/ so entao
    ver o passo ser pulado.
    """
    if not (do_ocr or want_describe):
        return do_ocr, None
    import frame_annotators as fa
    print("Anotacao dos prints:")
    if do_ocr:
        if fa.ocr_available():
            print("  - OCR (RapidOCR) ........... PRONTO")
        else:
            print("  - OCR (RapidOCR) ........... SERA PULADO (nao instalado). "
                  "Instale: pip install rapidocr-onnxruntime")
            do_ocr = False
    caption_backend = _resolve_caption_backend(want_describe, backend_pref)
    print()
    return do_ocr, caption_backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcreve um video (URL ou .mp4) e captura prints relevantes, sem IA.",
    )
    parser.add_argument("source", nargs="?", default=None,
                        help="URL do YouTube/Instagram ou caminho .mp4 "
                             "(se omitido, e pedido interativamente no terminal)")
    parser.add_argument("-s", "--screenshots", type=int, default=20,
                        help="prints desejados (encaixado em 1-12 por minuto de video; padrao 20)")
    parser.add_argument("--ocr", action="store_true",
                        help="extrai o TEXTO na tela dos prints (RapidOCR, local/gratuito)")
    parser.add_argument("--describe", action="store_true",
                        help="gera DESCRICAO do conteudo dos prints (mede a RAM e escolhe o backend)")
    parser.add_argument("--describe-backend", choices=["auto", "local", "online"], default="auto",
                        help="backend da descricao: auto (mede a RAM; local se folgada, senao Groq), "
                             "local (Florence-2; pede permissao se apertado) ou online (Groq vision)")
    args = parser.parse_args()

    # Dois modos: fonte ja no comando, ou modo interativo (sem argumento) - pergunta
    # a URL/caminho ao usuario, insistindo ate vir algo, o numero de prints e as
    # anotacoes opcionais (OCR/descricao).
    source = args.source.strip().strip('"').strip("'").strip() if args.source else ""
    screenshots = args.screenshots
    do_ocr, want_describe, backend_pref = args.ocr, args.describe, args.describe_backend
    if not source:
        print("Video Transcriber (sem IA)\n")
        source = force_input("Cole a URL (YouTube/Instagram) ou o caminho do .mp4: ")
        screenshots = _ask_screenshots(screenshots)
        do_ocr = _ask_yes_no("Extrair o texto na tela dos prints (OCR)?", do_ocr)
        want_describe = _ask_yes_no("Gerar descricao do conteudo dos prints?", want_describe)
        print()

    # Validacao precoce de arquivo local: erro claro e IMEDIATO se o caminho nao
    # existe (erro mais comum de quem nao tem instrucao), em vez de so descobrir
    # depois do aviso de servicos e do "Processando...".
    if source and not processor._is_url(source) and not os.path.isfile(os.path.expanduser(source)):
        print(f"Arquivo nao encontrado: {source}\n"
              "Confira o caminho do arquivo, ou cole uma URL do YouTube/Instagram.",
              file=sys.stderr)
        sys.exit(1)

    # Decide os anotadores ANTES do trabalho pesado: confere deps, mede a RAM e (se
    # apertado) pede permissao p/ rodar a descricao local; senao direciona p/ a nuvem.
    do_ocr, caption_backend = _setup_annotation(do_ocr, want_describe, backend_pref)

    # Transparencia: antes de tocar qualquer servico, diz o que vai ser acessado
    # e o custo - ja refletindo o backend resolvido (local x Groq vision).
    _print_services(source, do_ocr, caption_backend)

    # Probe best-effort: mostra a duracao e barra videos absurdamente longos
    # antes de comprometer download + processamento (mesmo guard rail do server).
    try:
        meta = processor.probe_source(source)
        dur = meta.get("duration") or 0
        if meta.get("title"):
            print(f"Video: {meta['title']}")
        if dur:
            print(f"Duracao: {processor.fmt_time(dur)}")
            if dur > processor.MAX_DURATION_HARD:
                print(f"Video muito longo. Limite de seguranca: "
                      f"{processor.fmt_time(processor.MAX_DURATION_HARD)}. Corte um trecho antes.")
                sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        pass  # probe e best-effort; se falhar, segue direto para o processamento

    print("\nProcessando (pode levar alguns minutos em videos longos)...\n")
    try:
        result = processor.process(source, screenshots, progress=_progress,
                                   do_ocr=do_ocr, caption_backend=caption_backend)
    except Exception as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)

    cache_dir = result["cache_dir"]
    n_frames = len(result["frames"])
    print("\nPronto!")
    print(f"  Transcricao:   {os.path.join(cache_dir, 'transcript.md')}")
    print(f"  Prints ({n_frames}):  {os.path.join(cache_dir, 'frames')}")
    print(f"  cache_id:      {result['cache_id']}")
    print("\nAbra o transcript.md no seu editor e os .jpg da pasta frames/.")


if __name__ == "__main__":
    main()

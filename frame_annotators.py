"""Anotacao OPCIONAL dos prints: OCR (texto na tela) + descricao (VLM).

Dois passos opcionais, DESLIGADOS por padrao, rodados depois do dedup e so nos
frames que sobraram (conjunto minimo):

  - OCR via **RapidOCR** (ONNX, leve, local, gratuito) -> le o texto na tela.
  - Descricao via **Florence-2** (Microsoft, MIT, 0.23B, roda em CPU) -> legenda
    do conteudo visual.

Ambos sao LOCAIS e offline; o Florence-2 baixa os pesos (~0.5GB) do Hugging Face
na 1a vez. As dependencias sao pesadas (o Florence-2 puxa torch), entao NAO estao
no requirements principal: se faltarem, o passo e PULADO com aviso claro — nunca
derruba o pipeline. Os imports caros ficam dentro das funcoes (lazy).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import unicodedata

log = logging.getLogger("video_transcriber.annotators")

FLORENCE_MODEL = "microsoft/Florence-2-base"   # 0.23B, MIT, roda em CPU
FLORENCE_TASK = "<CAPTION>"                      # legenda curta (<MORE_DETAILED_CAPTION> = longa)

_ocr_engine = None        # singletons carregados sob demanda (caros de inicializar)
_florence = None


# --- disponibilidade (sem importar as libs pesadas) ------------------------
def _installed(name: str) -> bool:
    """True so se o pacote esta REALMENTE instalado (nao um namespace vazio).

    `find_spec` acha ate um diretorio-namespace sem conteudo (ex.: um torch meio
    instalado), cujo `origin` e None — isso daria falso-positivo. Exigir origin
    real evita prometer um engine que vai estourar ImportError na hora de usar.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def ocr_available() -> bool:
    return _installed("rapidocr_onnxruntime") or _installed("rapidocr")


def caption_available() -> bool:
    return all(_installed(m) for m in ("torch", "transformers", "PIL"))


# --- OCR (RapidOCR) --------------------------------------------------------
def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR     # pacote classico, estavel
        except ImportError:
            from rapidocr import RapidOCR                 # pacote novo (2.x+)
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _meaningful(text: str) -> bool:
    """True se o texto tem ao menos uma letra LATINA ou um digito.

    Filtra ruido de OCR em icones/graficos (medido: pecas de xadrez viram '天'/'口'
    com confianca ALTA, entao o score nao separa). Preserva PT/EN com acentos,
    numeros e URLs; descarta deteccoes puramente CJK/simbolos.
    """
    for c in text:
        if c.isdigit():
            return True
        if c.isalpha() and unicodedata.name(c, "").startswith("LATIN"):
            return True
    return False


def _ocr_text(out) -> str:
    """Junta o texto legivel de uma saida do RapidOCR (API classica ou nova)."""
    # API classica: (result, elapse) com result = [[box, text, score], ...] | None
    if isinstance(out, tuple):
        result = out[0]
        if not result:
            return ""
        lines = sorted(result, key=lambda r: min(p[1] for p in r[0]))  # por y do topo
        return " ".join(t for r in lines if _meaningful(t := (r[1] or "").strip())).strip()
    # API nova: objeto com .txts ja em ordem de deteccao
    txts = getattr(out, "txts", None)
    if not txts:
        return ""
    return " ".join(t for raw in txts if _meaningful(t := (raw or "").strip())).strip()


def ocr_image(path: str) -> str:
    """Texto na tela em ordem de leitura (topo->baixo). '' se nada legivel."""
    return _ocr_text(_get_ocr()(path))


def ocr_region(frame, bbox) -> str:
    """Texto DENTRO de uma bbox (x0,y0,x1,y1) de um frame RGB (ndarray). '' se nada.

    Recorta a zona e roda o mesmo motor do ocr_image. Serve o screen_match: ler so o
    texto do que MUDOU numa tela, nao a tela toda."""
    import numpy as np
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    if crop.size == 0:
        return ""
    return _ocr_text(_get_ocr()(crop))


# --- Descricao: backend local (Florence-2) ou nuvem (Groq vision) ----------
# Local e pesado; o padrao e NAO rodar local sem o usuario querer. A CLI mede a
# RAM e decide: >= OK roda; entre MIN e OK pede permissao (cabe apertado); < MIN
# nao roda local -> nuvem. No nao-interativo (MCP) o 'auto' prefere a nuvem.
CAPTION_RAM_OK_GB = 2.0    # >= isto: roda local sem perguntar
CAPTION_RAM_MIN_GB = 1.5   # entre MIN e OK: pede permissao; abaixo: so nuvem

# Groq vision (nuvem): mesmo free-tier/chave do Whisper; Scout tem o maior TPM.
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_PROMPT = ("Descreva em uma unica frase curta, em portugues, o que se ve "
                      "nesta imagem: a cena, pessoas/objetos e qualquer texto em destaque.")
_groq_client = None


def _available_ram_gb() -> float | None:
    """RAM livre em GB, ou None se nao der p/ medir (a checagem entao nao bloqueia)."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = _MEM()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            return None
    try:  # Linux/Mac
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


def caption_local_readiness() -> tuple[str, "float | None", str]:
    """Avalia se da pra rodar o Florence-2 LOCAL agora. Retorna (tier, ram_gb, motivo).

    tier: 'ok'       -> roda direto (RAM >= OK);
          'tight'    -> cabe apertado (MIN <= RAM < OK) -> pedir permissao;
          'low'      -> RAM < MIN -> so nuvem;
          'no_torch' -> deps ausentes ou torch quebrado (namespace importa mas estoura).
    """
    if not caption_available():
        return "no_torch", None, "dependencias ausentes (torch/transformers/Pillow)"
    try:
        import torch
        torch.zeros(1)            # exercita o backend (namespace vazio estoura aqui)
    except Exception as exc:
        return "no_torch", None, f"torch nao funcional ({type(exc).__name__})"
    ram = _available_ram_gb()
    if ram is None:
        return "ok", None, "RAM nao medida (assumindo ok)"
    if ram >= CAPTION_RAM_OK_GB:
        return "ok", ram, f"{ram:.1f}GB livres"
    if ram >= CAPTION_RAM_MIN_GB:
        return "tight", ram, f"{ram:.1f}GB livres (apertado; min folgado {CAPTION_RAM_OK_GB:.0f}GB)"
    return "low", ram, f"{ram:.1f}GB livres < {CAPTION_RAM_MIN_GB:.1f}GB minimos"


# --- Descricao na nuvem (Groq vision) --------------------------------------
def groq_vision_available() -> bool:
    return _installed("groq") and bool(os.environ.get("GROQ_API_KEY"))


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def caption_image_groq(path: str) -> str:
    """Descricao do print via Groq vision (Llama-4). Levanta em erro (o chamador
    classifica via processor._describe_groq_error: limite/conexao/cobranca/auth)."""
    import base64
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = _get_groq().chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": GROQ_VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=120, temperature=0.2)
    return (resp.choices[0].message.content or "").strip()


# --- Descricao (Florence-2) ------------------------------------------------
def _get_florence():
    global _florence
    if _florence is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports
        from unittest.mock import patch

        # O codigo remoto do Florence-2 importa flash_attn incondicionalmente; em
        # CPU (sem flash-attn) isso quebra o load. Removemos esse import opcional.
        def _no_flash(filename):
            imp = get_imports(filename)
            return [m for m in imp if m != "flash_attn"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        with patch("transformers.dynamic_module_utils.get_imports", _no_flash):
            model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        proc = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
        _florence = (model, proc, device, dtype)
    return _florence


def caption_image(path: str) -> str:
    """Legenda do conteudo visual via Florence-2. '' em falha."""
    import torch
    from PIL import Image

    model, proc, device, dtype = _get_florence()
    image = Image.open(path).convert("RGB")
    inputs = proc(text=FLORENCE_TASK, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype)
    with torch.no_grad():
        ids = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                             max_new_tokens=256, num_beams=3, do_sample=False)
    text = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(
        text, task=FLORENCE_TASK, image_size=(image.width, image.height))
    cap = parsed.get(FLORENCE_TASK, "") if isinstance(parsed, dict) else str(parsed)
    return (cap or "").strip()


# --- driver ----------------------------------------------------------------
def _ensure(check, label: str, install_hint: str) -> bool:
    if check():
        return True
    log.warning("Anotacao '%s' pedida, mas a dependencia nao esta instalada — "
                "pulando este passo. Para habilitar: %s", label, install_hint)
    return False


def annotate_frames(frames: list[dict], do_ocr: bool = False,
                    caption_backend: "str | None" = None, progress=None) -> list[dict]:
    """Anexa 'ocr'/'caption' a cada frame (dict com 'path'). Mutaveis in-place.

    `caption_backend`: None (sem descricao) | 'florence' (local) | 'groq' (nuvem).
    A ESCOLHA do backend (medir RAM, pedir permissao) e do chamador (CLI/server);
    aqui so executa. `progress(i, n)`: andamento. Dependencia/engine ausente ->
    passo pulado com aviso. Falha pontual num frame -> '' naquele frame, segue.
    """
    if not (do_ocr or caption_backend) or not frames:
        return frames
    use_ocr = do_ocr and _ensure(ocr_available, "OCR (RapidOCR)",
                                  "pip install rapidocr-onnxruntime")
    cap = caption_backend  # 'florence' | 'groq' | None

    # Aquece cada engine UMA vez. Se a carga falhar (sem rede p/ baixar o modelo,
    # chave invalida, etc.), desliga aquele anotador e segue — um aviso so, em vez
    # de repetir o mesmo erro a cada frame.
    if use_ocr:
        try:
            _get_ocr()
        except Exception as exc:
            log.warning("OCR indisponivel na carga (%s) — pulando.", exc)
            use_ocr = False
    if cap == "florence":
        try:
            _get_florence()
        except Exception as exc:
            log.warning("Descricao local indisponivel na carga (%s) — pulando.", exc)
            cap = None
    elif cap == "groq":
        if not groq_vision_available():
            log.warning("Descricao online (Groq) pedida, mas falta o pacote groq ou "
                        "a GROQ_API_KEY — pulando.")
            cap = None
        else:
            try:
                _get_groq()
            except Exception as exc:
                log.warning("Descricao online (Groq) indisponivel na carga (%s) — pulando.", exc)
                cap = None
    if not (use_ocr or cap):
        return frames

    n = len(frames)
    for i, fr in enumerate(frames):
        path = fr.get("path")
        if use_ocr:
            try:
                fr["ocr"] = ocr_image(path)
            except Exception as exc:
                log.warning("OCR falhou em %s: %s", path, exc)
                fr["ocr"] = ""
        if cap == "florence":
            try:
                fr["caption"] = caption_image(path)
            except Exception as exc:
                log.warning("Descricao local falhou em %s: %s", path, exc)
                fr["caption"] = ""
        elif cap == "groq":
            try:
                fr["caption"] = caption_image_groq(path)
            except Exception as exc:
                # Reusa a classificacao de erro do Groq (limite/conexao/cobranca/auth).
                from processor import _describe_groq_error
                kind, msg = _describe_groq_error(exc)
                fr["caption"] = ""
                if kind in ("limite", "conexao", "cobranca", "auth"):
                    # Nao adianta insistir nos demais prints: para a descricao online.
                    log.warning("Descricao online (Groq) interrompida — %s: %s. "
                                "Demais prints ficam sem descricao.", kind, msg)
                    cap = None
                else:
                    log.warning("Descricao online (Groq) falhou em %s: %s", path, msg)
        if progress:
            try:
                progress(i + 1, n)
            except Exception:
                pass
    log.info("Anotacao concluida em %d frame(s) (ocr=%s, descricao=%s)",
             n, use_ocr, caption_backend or "nao")
    return frames

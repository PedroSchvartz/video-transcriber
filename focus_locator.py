"""
focus_locator.py - localizador de foco por JANELA + sinal ESPACIAL.

Ao contrario do metrics_dump (1 escalar por video), este modulo trabalha por
TRECHO: desliza uma janela ao longo do video e emite "spans de foco" rotulados,
com a regiao estudavel (bbox). Motivacao (ver planning): foco e propriedade de
COMO a tela se comporta num intervalo, nao do video inteiro -- o escalar por
video erra em vidos que trocam de regime (aula = parte falada + parte de slides).

Tres sinais, em ordem de decisao:
  - rest(t) por janela  -> imobilidade temporal (discriminador principal)
  - static_blocks/bbox  -> estrutura ESPACIAL: distingue "canvas estudavel com um
    distrator pequeno" (ex.: slide + webcam num canto) de "dinamico de verdade".
    O rest GLOBAL e cego a isso; a webcam derruba o rest mesmo com a tela parada.
  - visao (opcional)    -> SEMANTICO: rest mede imobilidade, nao informatividade;
    so a visao separa "slide com conteudo" de "card de espera morto".

A ferramenta LOCALIZA e ROTULA candidatos; nao faz gatekeeping (a decisao "vale a
pena focar?" e do consumidor da timeline, nao do sinal).

Uso:
    python focus_locator.py <URL ou caminho.mp4> [--json out.json] [--vision]
                            [--window 12] [--hop 4] [--dwell 20] [--gate 0.45]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict

import numpy as np

# Reusa os primitivos ja validados do metrics_dump (1 fonte de verdade).
from metrics_dump import (
    read_frames, block_view, probe_container, _is_url,
    FPS, GX, GY, T_COR, BLOCK_HOT, REST_MAX, STATIC_MAX, MASS_KEEP,
)

# --- parametros (POLITICA, nao calibracao por video) ------------------------
# rest e absoluto e configuravel de proposito: auto-calibrar por video faria um
# video todo-estatico subir o proprio gate e rejeitar tudo (anti-tese).
GATE_REST = 0.45        # janela com rest >= isso = "estatico" (separacao bimodal n=31)
WINDOW_S = 12.0         # largura da janela (s)
HOP_S = 4.0             # passo entre janelas (s)
MIN_DWELL_S = 20.0      # histerese: span so vale se durar >= isso (mata micro-spans)
D_CALM = 0.03           # "calmo": deadzone de aula tem d_mean 0.008-0.02; dinamico >=0.06
BBOX_SMALL = 0.30       # atividade localizada: bbox <= 30% da area = distrator pequeno
CARD_DMEAN = 0.001      # span estatico com d_mean abaixo disso = praticamente congelado
CARD_MIN_S = 60.0       # ... e mais longo que isso, tocando inicio/fim = suspeito de card

SPARK = " .:-=+*#"      # sparkline ASCII (console cp1252-safe no Windows)
TOTAL_BLOCKS = GX * GY


@dataclass
class FocusSpan:
    """Um trecho candidato a foco, com a regiao estudavel e o porque."""
    t0: float
    t1: float
    regime: str                 # "estatico" | "slide_pip"
    rest: float
    d_mean: float
    bbox: tuple                 # (x0,y0,x1,y1) na grade GXxGY: onde a atividade se concentra
    bbox_area_frac: float       # estatico: ~tela toda; slide_pip: bbox = distrator a mascarar
    suspeito_card: bool         # estatico longo + congelado + na borda -> confirmar c/ visao
    vision: dict | None = None  # so com --vision


@dataclass
class LocateResult:
    """Resultado completo de um video: spans + series por janela (p/ console/debug)."""
    spans: list
    centers: list               # tempo (s) do centro de cada janela
    rests: list
    dmeans: list
    regimes: list               # rotulo por janela
    rest_global: float
    dmean_global: float
    duration: float
    n_frames: int


# ===========================================================================
# Scan: 1 passada de ffmpeg (2 fps) -> d(t) global + mapa de blocos por frame.
# ===========================================================================
def scan_video(path: str):
    """Retorna (d, hot_stack): d[k] = fracao global mudada no frame k; hot_stack
    [N,GY,GX] uint8 = blocos 'quentes' (>BLOCK_HOT mudados) por frame."""
    prev = None
    d_series: list[float] = []
    hot_list: list[np.ndarray] = []
    for f in read_frames(path):
        if prev is not None:
            diff2 = ((f - prev) ** 2).sum(axis=2)
            changed = diff2 > (T_COR * T_COR)
            d_series.append(float(changed.mean()))
            hot_list.append((block_view(changed) >= BLOCK_HOT).astype(np.uint8))
        prev = f
    if not d_series:
        return np.zeros(0), np.zeros((0, GY, GX), dtype=np.uint8)
    return np.array(d_series), np.stack(hot_list)


def _localize(block_hot_count: np.ndarray):
    """Menor retangulo que concentra MASS_KEEP da atividade -> (bbox, area_frac).
    Sem atividade (tela congelada) -> tela inteira (estudavel = canvas todo)."""
    flat = block_hot_count.flatten().astype(float)
    total = flat.sum()
    if total <= 0:
        return (0, 0, GX, GY), 1.0
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    k = int(np.searchsorted(csum, MASS_KEEP * total)) + 1
    ys, xs = np.unravel_index(order[:k], (GY, GX))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / TOTAL_BLOCKS
    return bbox, round(area, 3)


def _label(rest_w: float, dmean_w: float, bbox_area: float) -> str:
    """Rotulo de regime de uma janela (a hierarquia de sinais da tese-mae)."""
    if rest_w >= GATE_REST:
        return "estatico"                       # tela parada no todo
    if dmean_w <= D_CALM and bbox_area <= BBOX_SMALL:
        return "slide_pip"                      # calmo + distrator localizado (aula)
    return "dinamico"


def window_metrics(d: np.ndarray, hot_stack: np.ndarray,
                   window_s: float, hop_s: float):
    """Agrega d/blocos em janelas deslizantes -> listas paralelas por janela."""
    wn = max(1, int(window_s * FPS))
    hop = max(1, int(hop_s * FPS))
    t = (np.arange(len(d)) + 1) / FPS          # tempo da transicao d[k]
    centers, rests, dmeans, regimes = [], [], [], []
    i = 0
    while i + wn <= len(d):
        wd = d[i:i + wn]
        wh = hot_stack[i:i + wn]
        rest_w = float((wd < REST_MAX).mean())
        dmean_w = float(wd.mean())
        block_count = wh.sum(axis=0)            # (GY,GX): em quantos frames cada bloco ficou quente
        _, bbox_area = _localize(block_count)
        centers.append(float(t[i] + window_s / 2))
        rests.append(round(rest_w, 4))
        dmeans.append(round(dmean_w, 5))
        regimes.append(_label(rest_w, dmean_w, bbox_area))
        i += hop
    return centers, rests, dmeans, regimes


def _span_stats(d: np.ndarray, hot_stack: np.ndarray, t0: float, t1: float):
    """Recalcula rest/d_mean/bbox sobre TODOS os frames do span [t0,t1] (mais
    preciso que mediar janelas)."""
    lo = max(0, int(t0 * FPS) - 1)
    hi = min(len(d), int(t1 * FPS))
    seg_d = d[lo:hi]
    seg_h = hot_stack[lo:hi]
    if len(seg_d) == 0:
        return 0.0, 0.0, (0, 0, GX, GY), 1.0
    rest = float((seg_d < REST_MAX).mean())
    dmean = float(seg_d.mean())
    bbox, area = _localize(seg_h.sum(axis=0))
    return round(rest, 4), round(dmean, 5), bbox, area


def spans_from_windows(d, hot_stack, centers, regimes, duration,
                       window_s, dwell_s) -> list[FocusSpan]:
    """Mescla janelas de foco (estatico/slide_pip) consecutivas em spans, aplica
    histerese (dwell minimo) e rotula cada span pelo regime dominante."""
    focus = [r in ("estatico", "slide_pip") for r in regimes]
    spans: list[FocusSpan] = []
    half = window_s / 2
    j = 0
    n = len(regimes)
    while j < n:
        if not focus[j]:
            j += 1
            continue
        k = j
        while k < n and focus[k]:
            k += 1
        # span cobre do inicio da 1a janela ao fim da ultima
        t0 = max(0.0, centers[j] - half)
        t1 = min(duration, centers[k - 1] + half)
        if t1 - t0 >= dwell_s:
            sub = regimes[j:k]
            regime = "slide_pip" if sub.count("slide_pip") > sub.count("estatico") else "estatico"
            rest, dmean, bbox, area = _span_stats(d, hot_stack, t0, t1)
            card = (regime == "estatico" and dmean < CARD_DMEAN
                    and (t1 - t0) >= CARD_MIN_S
                    and (t0 <= window_s or t1 >= duration - window_s))
            spans.append(FocusSpan(
                t0=round(t0, 1), t1=round(t1, 1), regime=regime,
                rest=rest, d_mean=dmean, bbox=bbox, bbox_area_frac=area,
                suspeito_card=bool(card)))
        j = k
    return spans


# ===========================================================================
# Orquestracao: resolve fonte (URL->temp), scan, janelas, spans, visao opcional.
# ===========================================================================
def _resolve_source(source: str):
    """(local_path, duration, cleanup_dir). URL -> baixa p/ temp (limpar depois)."""
    if _is_url(source):
        import processor
        dl = tempfile.mkdtemp(prefix="vtf_dl_")
        local, _info = processor.download_or_locate(source, dl)
        cont = probe_container(local)
        return local, cont.get("duration_s") or 0.0, dl
    cont = probe_container(source)
    return source, cont.get("duration_s") or 0.0, None


def analyze_local(local: str, duration: float, do_vision: bool,
                  window_s: float, hop_s: float, dwell_s: float) -> LocateResult:
    """Pipeline sobre um arquivo LOCAL ja resolvido."""
    d, hot_stack = scan_video(local)
    n = len(d) + 1
    if len(d) == 0:
        return LocateResult([], [], [], [], [], 0.0, 0.0, duration, n)
    rest_global = round(float((d < REST_MAX).mean()), 4)
    dmean_global = round(float(d.mean()), 5)
    centers, rests, dmeans, regimes = window_metrics(d, hot_stack, window_s, hop_s)
    spans = spans_from_windows(d, hot_stack, centers, regimes, duration, window_s, dwell_s)
    if do_vision and spans:
        _annotate_vision(local, duration, spans)
    return LocateResult(spans, centers, rests, dmeans, regimes,
                        rest_global, dmean_global, duration, n)


def _annotate_vision(local: str, duration: float, spans: list[FocusSpan]) -> None:
    """Preenche span.vision com categoria/rosto/texto amostrando DENTRO do span.
    So aqui se importa a visao (envia ao Groq). Falha vira {'vision_error': ...}."""
    import metrics_dump
    for sp in spans:
        try:
            sp.vision = metrics_dump.vision_metrics(local, duration, start=sp.t0, end=sp.t1)
        except Exception as e:  # nao derruba o resto dos spans
            sp.vision = {"vision_error": f"{type(e).__name__}: {e}"}


def locate_focus_spans(source: str, do_vision: bool = False,
                       window_s: float = WINDOW_S, hop_s: float = HOP_S,
                       dwell_s: float = MIN_DWELL_S) -> list[FocusSpan]:
    """API de biblioteca: spans de foco de UM video (URL ou arquivo)."""
    local, duration, cleanup = _resolve_source(source)
    try:
        return analyze_local(local, duration, do_vision, window_s, hop_s, dwell_s).spans
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


# ===========================================================================
# CLI: imprime escalar (contraste) + sparkline + tabela + spans; --json salva.
# ===========================================================================
def _spark(vals, lo=0.0, hi=1.0) -> str:
    out = []
    for v in vals:
        f = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
        out.append(SPARK[min(len(SPARK) - 1, int(f * (len(SPARK) - 1) + 0.5))])
    return "".join(out)


def _print_report(res: LocateResult) -> None:
    print(f"\nduracao ~{res.duration:.0f}s | frames {res.n_frames} @ {FPS}fps")
    print(f"ESCALAR (video inteiro): rest_frac={res.rest_global}  "
          f"d_mean={res.dmean_global}  "
          f"-> {'FOCO' if res.rest_global >= GATE_REST else 'nao-foco'}")
    if not res.centers:
        print("  (poucos frames p/ janela)")
        return
    rests = np.array(res.rests)
    dmeans = np.array(res.dmeans)
    print(f"\nJANELA {WINDOW_S:.0f}s / passo {HOP_S:.0f}s  ({len(rests)} janelas)")
    print(f"  rest(t)   {_spark(rests, 0, 1)}   [{rests.min():.2f}..{rests.max():.2f}]")
    print(f"  d_mean(t) {_spark(dmeans, 0, max(float(dmeans.max()), 1e-6))}   "
          f"[{dmeans.min():.4f}..{dmeans.max():.4f}]")
    if res.spans:
        print(f"\nSPANS DE FOCO ({len(res.spans)}):")
        for sp in res.spans:
            tag = "  [SUSPEITO CARD]" if sp.suspeito_card else ""
            vis = f"  visao={sp.vision.get('vision_category')}" if sp.vision else ""
            print(f"  {sp.t0:6.0f}s -> {sp.t1:6.0f}s  ({sp.t1 - sp.t0:4.0f}s)  "
                  f"{sp.regime:9s} rest={sp.rest:.2f} d_mean={sp.d_mean:.4f} "
                  f"bbox={sp.bbox} area={sp.bbox_area_frac}{tag}{vis}")
    else:
        print("\nSPANS DE FOCO: nenhum")


def main() -> None:
    global GATE_REST
    ap = argparse.ArgumentParser(description="Localizador de foco por janela -> spans")
    ap.add_argument("source", help="URL (YouTube/Instagram) ou caminho de video")
    ap.add_argument("--json", help="salva os spans nesse arquivo .json")
    ap.add_argument("--vision", action="store_true",
                    help="confirma cada span via Groq vision (ENVIA ao Groq; off por padrao)")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--hop", type=float, default=HOP_S)
    ap.add_argument("--dwell", type=float, default=MIN_DWELL_S)
    ap.add_argument("--gate", type=float, default=GATE_REST,
                    help="limiar de rest para 'estatico' (politica, nao calibrado)")
    args = ap.parse_args()
    GATE_REST = args.gate

    if args.vision:  # Fase semantica precisa da GROQ_API_KEY do .env (como metrics_dump)
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except Exception:
            pass

    print(f"fonte: {args.source}")
    local, duration, cleanup = _resolve_source(args.source)
    try:
        res = analyze_local(local, duration, args.vision, args.window, args.hop, args.dwell)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
    _print_report(res)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(s) for s in res.spans], fh, ensure_ascii=False, indent=2)
        print(f"\nspans -> {args.json}")


if __name__ == "__main__":
    main()

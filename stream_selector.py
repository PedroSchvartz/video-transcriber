"""
Selecao de frames ao vivo (modo online / stream) — versao incremental do
`frame_selector`, para uma fonte SEM FIM (ex.: a tela do celular via scrcpy).

O `frame_selector` e batch: varre o arquivo inteiro, deriva o limiar da distribuicao
do video todo e so entao marca. Aqui a fonte e um stream: os limiares tem que ser
estimados ONLINE e o frame emitido a medida que a sessao acontece.

Desenho (ver planning/modo_online.md, "Decisoes 2026-07-03"):
  - Detecção de mudança AUTO-RECALIBRANTE: a fracao de pixels "sujos" entre quadros
    consecutivos alimenta uma media/desvio EXPONENCIAL (EMA) — o limiar de area
    acompanha o padrao da PROPRIA tela ao vivo. Tela que muda pouco fica sensivel;
    regime de movimento continuo (video tocando) sobe o proprio limiar e para de
    disparar a cada quadro. Piso absoluto (EDIT_AREA_FLOOR) ancora o regime
    morto-estatico, onde a baseline relativa -> 0.
  - Warmup: nao emite nada nos primeiros WARMUP_S — so acumula a baseline.
  - Emit RETROSPECTIVO (~MIN_GAP atras da borda): ao detectar mudanca nao emite na
    hora; agrupa a rajada de transicao e, quando a tela ASSENTA (MIN_GAP de quietude),
    escolhe o frame mais NITIDO ja-assentado do grupo e emite. Numa UI o conteudo
    interessante e a tela DEPOIS da transicao, nao a de saida.
  - Dedup por JANELA limitada (nao O(n^2)): compara so contra as ultimas emissoes,
    reusando o predicado do frame_selector.

Fonte-agnostico: consome um iteravel de (t_pc, frame_bgr_ou_rgb ndarray HxWx3). O
adaptador scrcpy (produzir esses frames ao vivo) e camada fina por cima e validavel
por replay de um clipe/sintetico, sem telefone.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import frame_selector as fs

log = logging.getLogger("video_transcriber.stream")

# --- Parametros do modo online --------------------------------------------
DETECT_SIZE = fs.EDIT_SIZE          # downscale quadrado p/ medir mudanca (128)
T_COR_LIVE = fs.DEDUP_T_COR         # piso de cor por pixel. Screencast e sinal DIGITAL
                                    # limpo: ruido por pixel ~constante e baixo, ao
                                    # contrario de camera. Atalho consciente: fixo aqui
                                    # (adaptar seria YAGNI ate uma fonte ruidosa provar
                                    # o contrario; a receita auto-medida do batch,
                                    # T_cor do p99-dos-quietos, segue no caminho gravado).
WARMUP_S = 2.0                      # nao emite antes disso (baseline fria)
TAU_CALIB = 8.0                     # constante de tempo (s) da EMA de calibracao de area
AREA_K = fs.EDIT_AREA_K             # limiar de area = EMA_media + K*EMA_desvio
AREA_FLOOR = fs.EDIT_AREA_FLOOR     # piso absoluto (ancora o regime morto-estatico)
MIN_GAP = fs.MIN_GAP_SECONDS        # quietude que fecha um grupo de mudanca (assentou)
DEDUP_WINDOW = 8                    # quantas emissoes recentes o dedup compara


@dataclass
class StreamEvent:
    """Um frame que 'venceu' a selecao online e esta pronto para virar print."""
    t: float                 # tempo-PC do frame emitido (relogio comum, ver plano)
    frame: np.ndarray        # o quadro escolhido (resolucao como foi recebida)
    reason: str              # "mudanca" (por ora; "acao" quando o loop interleavar eventos)
    dirty: float             # fracao suja que disparou o grupo (diagnostico)


def marks_from_events(actions: list[dict]) -> list[fs.FrameMark]:
    """Marca de frame a partir de uma trilha de ACOES (taps/swipes/logcat/console).

    Espelha `from_chapters`/`from_words`: cada acao com timestamp `t` (tempo-PC, ver
    o contrato em ponte_android/planning/plano.md) vira uma marca. E o PAPEL DE FRAME
    do `from_events`; a outra metade — a acao virar TRILHA TEXTUAL da timeline, no
    lugar da fala — mora no montador (server.py/processor.py), nao aqui: o FrameMark
    nao carrega texto.
    """
    marks: list[fs.FrameMark] = []
    for a in actions or []:
        t = a.get("t")
        if t is not None:
            marks.append(fs.FrameMark(time=float(t), reason="acao"))
    return marks


def _laplacian_var(gray: np.ndarray) -> float:
    """Nitidez = variancia do laplaciano (foco). Maior = mais nitido/assentado."""
    lap = (-4.0 * gray[1:-1, 1:-1]
           + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


class OnlineSelector:
    """Seletor incremental: alimente `feed(t, frame)` quadro a quadro; devolve os
    StreamEvent que ficaram PRONTOS (assentaram) naquele passo. `flush()` no fim
    fecha um grupo pendente.
    """

    def __init__(self, *, t_cor: float = T_COR_LIVE, warmup_s: float = WARMUP_S,
                 tau: float = TAU_CALIB, area_k: float = AREA_K,
                 area_floor: float = AREA_FLOOR, min_gap: float = MIN_GAP,
                 dedup_window: int = DEDUP_WINDOW):
        self.t_cor = t_cor
        self.warmup_s = warmup_s
        self.tau = tau
        self.area_k = area_k
        self.area_floor = area_floor
        self.min_gap = min_gap

        self._t0: float | None = None
        self._prev_small: np.ndarray | None = None   # frame anterior downscale (float)
        # EMA de media/variancia da fracao suja (calibracao de area rolling)
        self._ema_mean = 0.0
        self._ema_var = 0.0
        self._ema_init = False
        # Grupo de mudanca em aberto (histerese): abre pelo limiar adaptativo
        # (sensivel) e so FECHA quando a tela fica genuinamente quieta — julgado por
        # barra ABSOLUTA (o piso), nao pelo limiar adaptativo, que infla no movimento
        # continuo e falsamente "assentaria" no meio da rajada.
        self._group_open = False
        self._last_motion = 0.0   # ultimo t com dirty acima da barra de quietude
        self._pending_dirty = 0.0
        # Buffer recente p/ escolher o frame assentado mais nitido (bounded no tempo)
        self._buf: deque[tuple[float, np.ndarray, float]] = deque()  # (t, frame, nitidez)
        # Janela de dedup: assinaturas das ultimas emissoes
        self._sigs: deque = deque(maxlen=dedup_window)

    # -- calibracao ---------------------------------------------------------
    def _area_threshold(self) -> float:
        std = math.sqrt(max(self._ema_var, 0.0))
        return max(self._ema_mean + self.area_k * std, self.area_floor)

    def _update_calib(self, dirty: float, dt: float) -> None:
        # alpha dependente do tempo: janela ~tau segundos, robusto a fps variavel.
        alpha = 1.0 - math.exp(-dt / self.tau) if dt > 0 else 0.0
        if not self._ema_init:
            self._ema_mean, self._ema_var, self._ema_init = dirty, 0.0, True
            return
        delta = dirty - self._ema_mean
        self._ema_mean += alpha * delta
        self._ema_var = (1.0 - alpha) * (self._ema_var + alpha * delta * delta)

    # -- ingestao ----------------------------------------------------------
    def feed(self, t: float, frame: np.ndarray) -> list[StreamEvent]:
        if self._t0 is None:
            self._t0 = t
        small = _downscale(frame, DETECT_SIZE)          # HxWx3 float
        events: list[StreamEvent] = []

        if self._prev_small is not None:
            dist = np.sqrt(((small - self._prev_small) ** 2).sum(axis=2))
            dirty = float((dist > self.t_cor).mean())
            dt = t - self._buf[-1][0] if self._buf else 0.0

            warmed = (t - self._t0) >= self.warmup_s
            thr = self._area_threshold()
            self._update_calib(dirty, dt)               # baseline inclui todos os quadros

            gray = small.mean(axis=2)
            self._buf.append((t, frame, _laplacian_var(gray)))
            self._prune_buffer(t)

            quiet_bar = self.area_floor
            if not self._group_open:
                if warmed and dirty >= thr:               # abre pelo limiar adaptativo
                    self._group_open = True
                    self._last_motion = t
                    self._pending_dirty = dirty
            else:
                if dirty > quiet_bar:                     # ainda se mexendo (barra absoluta)
                    self._last_motion = t
                    self._pending_dirty = max(self._pending_dirty, dirty)
                if (t - self._last_motion) >= self.min_gap:   # assentou de verdade
                    ev = self._emit_settled(self._last_motion)
                    if ev is not None:
                        events.append(ev)
                    self._group_open = False
                    self._pending_dirty = 0.0
        else:
            self._buf.append((t, frame, 0.0))

        self._prev_small = small
        return events

    def flush(self) -> list[StreamEvent]:
        """Fim de stream: fecha um grupo pendente escolhendo o frame ja assentado."""
        if not self._group_open:
            return []
        ev = self._emit_settled(self._last_motion)
        self._group_open = False
        return [ev] if ev is not None else []

    # -- emissao -----------------------------------------------------------
    def _emit_settled(self, change_t: float) -> StreamEvent | None:
        # Frames ASSENTADOS = os que vieram DEPOIS da ultima mudanca (a tela nova ja
        # composta). Escolhe o mais nitido entre eles.
        settled = [(t, fr, sh) for (t, fr, sh) in self._buf if t >= change_t]
        if not settled:
            return None
        t_best, frame_best, _ = max(settled, key=lambda x: x[2])

        if self._is_dup(frame_best):
            log.info("Stream: frame assentado em %.2fs e duplicata recente; descartado", t_best)
            return None
        return StreamEvent(t=t_best, frame=frame_best, reason="mudanca",
                           dirty=self._pending_dirty)

    def _is_dup(self, frame: np.ndarray) -> bool:
        import imagehash
        from PIL import Image

        img = Image.fromarray(_as_rgb_uint8(frame))
        ph, ch = imagehash.phash(img), imagehash.colorhash(img)
        small = np.asarray(img.resize((64, 64)), dtype=np.int32)
        if fs._is_dup_of(ph, ch, small, self._sigs):
            return True
        self._sigs.append((ph, ch, small))
        return False

    def _prune_buffer(self, now: float) -> None:
        horizon = self.min_gap + 0.6     # cobre [pending, now] + folga
        while self._buf and now - self._buf[0][0] > horizon:
            self._buf.popleft()


def _downscale(frame: np.ndarray, size: int) -> np.ndarray:
    """Downscale quadrado para medir mudanca — barato e robusto a baixa resolucao."""
    from PIL import Image

    img = Image.fromarray(_as_rgb_uint8(frame)).resize((size, size))
    return np.asarray(img, dtype=np.float32)


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Normaliza um frame para HxWx3 uint8 RGB (aceita RGBA e tipos != uint8)."""
    arr = frame
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def select_stream(source, selector: OnlineSelector | None = None):
    """Conveniencia: consome um iteravel de (t, frame) e gera StreamEvent.

    Serve tanto para o adaptador scrcpy quanto para replay offline (validacao).
    """
    sel = selector or OnlineSelector()
    for t, frame in source:
        yield from sel.feed(float(t), frame)
    yield from sel.flush()

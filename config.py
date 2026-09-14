"""
Configuracao do pipeline — parametros numericos ajustaveis.

Precedencia (maior ganha):
  1. Variaveis de ambiente VT_* (ex.: VT_MAX_FRAMES_PER_MINUTE=9999 no .env)
  2. config.json na pasta da ferramenta (opcional; versionavel, pode ser commitado)
  3. Defaults hard-coded

Para persistir uma configuracao, edite o .env (vars VT_*) ou crie um config.json.
9999 em max_frames_per_minute desativa o teto (sem limite de prints por minuto).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_HERE = Path(__file__).parent

_DEFAULTS: dict = {
    "min_frames_per_minute":   1,      # piso de prints por minuto de video
    "max_frames_per_minute":  12,      # teto de prints por minuto (9999 = sem limite)
    "max_duration_hours":      4.0,    # recusa videos mais longos que isso
    "max_cache_entries":      10,      # quantos videos manter no cache (LRU)
    "silence_threshold_dbfs": -50.0,   # mean_volume abaixo disso = sem fala; pula transcricao

    # --- Modo online / stream (perfil TELA: PC/Android) ---------------------
    # A selecao ao vivo AUTO-RECALIBRA o limiar de area ao padrao da propria tela
    # (EMA de media/desvio da fracao suja). Por isso estes NAO sao constantes
    # cravadas de um corpus de UI (que nao temos) — sao os META-parametros que
    # governam a auto-calibracao, mais o piso absoluto que ancora o regime
    # morto-estatico. Defaults herdados da calibracao de VIDEO (frame_selector),
    # exceto onde a diferenca de regime justifica (ver stream_min_gap_s).
    # Racional detalhado em planning/config_parametros.md.
    "stream_t_cor":          50.0,   # distancia de cor p/ pixel "diferente" (screencast digital: ruido baixo e constante)
    "stream_warmup_s":        2.0,   # nao emite antes disso — baseline fria (spec: 2-5s)
    "stream_tau_calib_s":     8.0,   # constante de tempo (s) da EMA de calibracao de area
    "stream_area_k":          2.0,   # limiar de area = EMA_media + K*EMA_desvio
    "stream_area_floor":      0.02,  # piso absoluto (fracao da tela) p/ contar como mudanca; UI real pode querer 0.03-0.05 (MEDIR antes de cravar)
    "stream_min_gap_s":       0.8,   # quietude que fecha um grupo; < que os 1.5 do video: transicao de UI e discreta e rapida
    "stream_dedup_window":    8,     # quantas emissoes recentes o dedup compara (bound de memoria)
}

_ENV_MAP: dict[str, tuple[str, type]] = {
    "VT_MIN_FRAMES_PER_MINUTE":  ("min_frames_per_minute",  int),
    "VT_MAX_FRAMES_PER_MINUTE":  ("max_frames_per_minute",  int),
    "VT_MAX_DURATION_HOURS":     ("max_duration_hours",     float),
    "VT_MAX_CACHE_ENTRIES":      ("max_cache_entries",      int),
    "VT_SILENCE_THRESHOLD_DBFS": ("silence_threshold_dbfs", float),

    "VT_STREAM_T_COR":           ("stream_t_cor",           float),
    "VT_STREAM_WARMUP_S":        ("stream_warmup_s",        float),
    "VT_STREAM_TAU_CALIB_S":     ("stream_tau_calib_s",     float),
    "VT_STREAM_AREA_K":          ("stream_area_k",          float),
    "VT_STREAM_AREA_FLOOR":      ("stream_area_floor",      float),
    "VT_STREAM_MIN_GAP_S":       ("stream_min_gap_s",       float),
    "VT_STREAM_DEDUP_WINDOW":    ("stream_dedup_window",    int),
}


def load() -> dict:
    """Carrega a configuracao aplicando a precedencia correta."""
    cfg = dict(_DEFAULTS)
    cfg_path = _HERE / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    for env_key, (cfg_key, typ) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = typ(val)
            except ValueError:
                pass
    return cfg


def get() -> dict:
    """Retorna a configuracao atual.

    Sem cache intencional: le sempre os env vars para suportar overrides em
    runtime (ex.: CLI que seta VT_MAX_FRAMES_PER_MINUTE antes de rodar o pipeline).
    """
    return load()

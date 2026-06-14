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
}

_ENV_MAP: dict[str, tuple[str, type]] = {
    "VT_MIN_FRAMES_PER_MINUTE":  ("min_frames_per_minute",  int),
    "VT_MAX_FRAMES_PER_MINUTE":  ("max_frames_per_minute",  int),
    "VT_MAX_DURATION_HOURS":     ("max_duration_hours",     float),
    "VT_MAX_CACHE_ENTRIES":      ("max_cache_entries",      int),
    "VT_SILENCE_THRESHOLD_DBFS": ("silence_threshold_dbfs", float),
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

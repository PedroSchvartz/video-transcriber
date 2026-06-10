"""
Garante que ffmpeg/ffprobe estejam acessiveis, mesmo quando o winget acabou
de instalar e o PATH do processo ainda nao foi atualizado.

Ordem de busca:
  1. Variavel de ambiente FFMPEG_DIR (pasta com ffmpeg.exe/ffprobe.exe)
  2. PATH atual
  3. Instalacao do winget (Gyan.FFmpeg) em %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages

Ao encontrar, prefixa a pasta no PATH do processo, de modo que todas as
chamadas subprocess a "ffmpeg"/"ffprobe" passem a funcionar sem alteracao.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil

log = logging.getLogger("video_transcriber.ffmpeg")

_ensured = False


def ensure_on_path() -> bool:
    """Coloca o diretorio do ffmpeg no PATH se necessario. Idempotente."""
    global _ensured
    if _ensured:
        return True

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        _ensured = True
        return True

    candidates: list[str] = []

    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir:
        candidates.append(env_dir)

    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates += glob.glob(
            os.path.join(local, "Microsoft", "WinGet", "Packages",
                         "Gyan.FFmpeg*", "**", "bin"),
            recursive=True,
        )

    for d in candidates:
        if os.path.isfile(os.path.join(d, "ffmpeg.exe")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            log.info("ffmpeg localizado e adicionado ao PATH: %s", d)
            _ensured = True
            return True

    log.warning("ffmpeg nao encontrado. Instale com: winget install Gyan.FFmpeg")
    return False

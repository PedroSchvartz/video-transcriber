"""Worker em subprocesso: roda UM job de transcricao, atualizando o status.

Roda como processo proprio (lancado por server.process_video) para que o
cancel possa mata-lo — e aos ffmpeg/yt-dlp filhos — sem derrubar o server MCP.

Uso:  python worker.py <cache_id> <source> [max_screenshots] [ocr 0|1] [caption_backend] [force 0|1]
      caption_backend: '' (sem descricao) | 'florence' (local) | 'groq' (nuvem)
      force: '1' reprocessa mesmo que a fonte ja tenha sido vista
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import processor

if __name__ == "__main__":
    cache_id = sys.argv[1]
    source = sys.argv[2]
    max_screenshots = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    do_ocr = len(sys.argv) > 4 and sys.argv[4] == "1"
    caption_backend = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    force = len(sys.argv) > 6 and sys.argv[6] == "1"

    # O server lanca o worker destacado, com stdout/stderr -> DEVNULL. Sem um log
    # em arquivo, uma morte do worker no meio do job nao deixaria rastro nenhum
    # para diagnosticar. Logamos um arquivo por job, irmao da pasta do cache.
    os.makedirs(processor.CACHE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(processor.CACHE_DIR, f"{cache_id}.log"),
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    processor.run_job(cache_id, source, max_screenshots, do_ocr=do_ocr,
                      caption_backend=caption_backend, force=force)

# video_transcriber — orientação para agentes

Servidor MCP + CLI de transcrição de vídeo. Recebe uma URL (YouTube/Instagram) ou um
`.mp4` e devolve uma **timeline**: a fala (com timestamps) e os **prints** dos momentos
de mudança intercalados em ordem cronológica — como se o Claude tivesse assistido.

## Pipeline

`fonte (URL ou .mp4)` → probe (lê duração/título sem baixar) → download (yt-dlp, até
1080p) → extração de áudio (ffmpeg, Opus) → transcrição (Groq Whisper; fallback
`faster-whisper` local) → seleção de frames (fala/cena/edição/áudio/capítulo/intervalo,
limiares auto-calibrados) → dedup (pHash/colorhash) → anotação opcional (OCR/descrição)
→ **timeline** (fala + prints intercalados).

Cada vídeo vira um cache em `_cache/<id>/` (mantém os 10 mais recentes), o que permite
remontar a timeline e capturar novos instantes sem reprocessar.

## Quando usar cada entry point

| Arquivo | Quando usar |
|---------|-------------|
| `server.py` | **Servidor MCP** — usado pelo Claude Code. Fluxo: `process_video(source)` inicia em background e retorna um `cache_id` na hora (não trava em vídeos longos); acompanhe com `get_status(cache_id)`; quando pronto, pegue a timeline com `get_result(cache_id)`; `cancel(cache_id)` interrompe de verdade; `get_frame(cache_id, timestamp)` "rebobina" para qualquer instante. |
| `transcribe.py` | **CLI standalone** — roda o pipeline direto no terminal, síncrono, sem IA. `python transcribe.py "<URL ou caminho .mp4>"` (sem argumento entra em modo interativo). Flags opcionais: `--screenshots/-s N`, `--ocr`, `--describe`, `--describe-backend auto|local|online`. Passo a passo completo em `USO_SEM_IA.md`. |

## Setup

```
pip install -r requirements.txt
winget install Gyan.FFmpeg          # uma vez só; o código localiza o binário sozinho
```

Chave Groq (opcional, recomendada — transcrição muito mais rápida e precisa): copie
`.env.example` para `.env` e preencha `GROQ_API_KEY` (crie de graça em
https://console.groq.com/keys). Sem a chave, usa `faster-whisper` local (mais lento,
funciona offline).

Registrar o MCP no Claude Code (escopo global) e reiniciar:

```
claude mcp add video-transcriber -s user -- python "C:\Codigos\claud_develop\tools\video_transcriber\server.py"
```

Confirme com `/mcp` que `video-transcriber` está ativo. Remover:
`claude mcp remove video-transcriber -s user`.

**Anotação dos prints** (opcional, desligada por padrão — `--ocr` / `--describe`). As
dependências são pesadas e ficam comentadas no `requirements.txt`; instale só se for
usar:

```
pip install rapidocr-onnxruntime           # OCR (texto na tela) — leve, ONNX
pip install torch transformers timm einops # descrição LOCAL (Florence-2) — ~2 GB (torch)
```

A descrição **na nuvem** (`--describe-backend online`) não exige nada disso: usa o
pacote `groq` (já instalado) e a `GROQ_API_KEY`.

## GROQ_API_KEY

| Variável | Obrigatória | Para quê |
|----------|-------------|----------|
| `GROQ_API_KEY` | Não | Transcrição via Groq Whisper (free tier) e descrição de prints via Groq vision. Sem ela, transcrição cai no `faster-whisper` local. |

Definida em `tools/video_transcriber/.env` — **nunca commitar** (está no `.gitignore`).

## Arquivos principais

| Arquivo | Papel |
|---------|-------|
| `server.py` | MCP server — tools `process_video`/`get_status`/`get_result`/`cancel`/`get_frame`; lança o worker e monta a timeline |
| `transcribe.py` | CLI sem IA — roda o pipeline direto no terminal |
| `worker.py` | Processo de background matável (um por job), atualiza o status |
| `processor.py` | Pipeline (probe, download, áudio, transcrição, frames) + cache + status + `run_job`/`cancel` |
| `frame_selector.py` | Seleção de timestamps (fala/cena/edição/áudio/capítulo/intervalo) + dedup |
| `frame_annotators.py` | Anotação opcional dos prints — OCR (RapidOCR) + descrição (Florence-2 local ou Groq vision) |
| `ffmpeg_tools.py` | Localiza ffmpeg/ffprobe no PATH |

Docs de apoio: `README.md` (visão geral da ferramenta) e `USO_SEM_IA.md` (uso da CLI no
terminal, sem IA).

## Análise / experimentação (fora do pipeline)

| Arquivo | Papel |
|---------|-------|
| `metrics_dump.py` | Batch de métricas sobre uma **coleção** de vídeos → 1 linha CSV por vídeo (~45 colunas), para tirar conclusões com dados. **Fase A** (sem chave, ~s/vídeo): container, áudio (volume/silêncio), assinatura de mudança por blocos (rest/static/localização/recorrência), brilho, proxy de texto (bordas) e de movimento global (correlação de fase). **Fase B** (`--transcript`/`--vision`/`--all`, usa `GROQ_API_KEY`, **envia ao Groq**): words/min + cobertura de fala; categoria + rosto + texto (1 chamada de visão/vídeo sobre colagem). Uso: `python metrics_dump.py <pasta \| lista.txt \| vídeos> [-o saida.csv] [--all]`. |

Serve para investigar **quando** uma futura "ferramenta de foco" seria útil: a classe
foco (ex.: xadrez) se identifica pela **convergência** de sinais (rest alto + tela
estática + câmera parada + categoria semântica), não por um limiar único.

## Não mexa

- **`_cache/`** — artefatos gerados (vídeos, frames, transcripts, `meta.json`,
  `result.json`, timings). Nunca commitar, nunca editar à mão.
- **`_test_*` / `_run_real.py` / `_prescan.py`** — scripts de teste/diagnóstico avulsos, fora do pipeline (no `.gitignore`).
- **`.env`** — contém `GROQ_API_KEY`. Nunca commitar.

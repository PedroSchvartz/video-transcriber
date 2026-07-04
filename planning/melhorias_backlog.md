# Backlog de melhorias — video_transcriber

Levantamento de oportunidades identificadas em 2026-06-14.  
Prioridade: **P0** (impacto alto, esforço baixo) · **P1** (alto impacto) · **P2** (nice-to-have)

---

## P0 — Quick wins

| # | Melhoria | Arquivo | Detalhe |
|---|---------|---------|---------|
| 1 | **Parâmetros configuráveis via `.env`/`config.json`** | `config.py` | ✅ **Implementado** — `VT_MIN/MAX_FRAMES_PER_MINUTE`, `VT_MAX_DURATION_HOURS`, `VT_MAX_CACHE_ENTRIES`, `VT_SILENCE_THRESHOLD_DBFS`; flags `--no-limit`, `--min-fpm`, `--max-fpm` na CLI |
| 2 | **Filtro pós-transcrição de alucinações do Whisper** | `processor.py` | Remover segmentos cujo texto bate com lista conhecida ("Thank you.", "Thanks for watching.", etc.) — complemento à verificação de silêncio já existente |
| 3 | **Verificação de espaço em disco antes de baixar** | `processor.py` | Antes de `download_or_locate()`, checar espaço livre estimado (vídeo em 1080p ≈ duração × 1 MB/s) e avisar se insuficiente |
| 4 | **`language` parameter para Groq Whisper** | `processor.py` | Adicionar flag `--language`/`VT_LANGUAGE` para forçar idioma da transcrição (ex.: `pt`, `en`) — evita switches de idioma em vídeos multilíngues |
| 5 | **Caption mais detalhada com Florence-2** | `frame_annotators.py` | Expor opção `--describe-detail full` que usa `<MORE_DETAILED_CAPTION>` em vez de `<CAPTION>` |

---

## P1 — Alto impacto

| # | Melhoria | Arquivo | Detalhe |
|---|---------|---------|---------|
| 6 | **Extração de legendas embutidas (closed captions)** | `processor.py` | Se o vídeo tiver `.srt`/`vtt` (yt-dlp já baixa com `--write-subs`), usar como fonte de transcrição em vez do Whisper — muito mais rápido e preciso |
| 7 | **OCR e descrição em paralelo** | `frame_annotators.py` | O loop de anotação é sequencial frame-a-frame. Rodar em `ThreadPoolExecutor` (I/O-bound para Groq, CPU-bound para OCR leve) pode ser 2–3× mais rápido em vídeos com muitos prints |
| 8 | **Retry em timeout de rede (Groq)** | `processor.py` | Falha de Groq por `ConnectionError`/`Timeout` cai para `faster-whisper` imediatamente. Um retry com backoff exponencial (2 tentativas, 2s + 4s) reduziria degradações desnecessárias em conexões instáveis |
| 9 | **Dedup melhorado para telas de texto** | `frame_selector.py` | `DIRTY_DUP_LOW = 0.02` (2% diferente = duplicata) pode fundir slides de texto que mudam pouco. Considerar threshold maior (5%) ou verificar se o pHash difere antes de descartar por "sujo" |
| 10 | **Fallback de modelo Groq vision** | `frame_annotators.py` | Se `llama-4-scout` não estiver disponível (limit/auth), tentar outro modelo vision do Groq em vez de silenciosamente pular a descrição |
| 11 | **Backend do descritor no resultado** | `result.json` / `meta.json` | Gravar qual backend de descrição foi usado (`florence` ou `groq`) no `meta.json`, para rastreabilidade ao remontar a timeline |
| 12 | **Validação de `config.json`** | `config.py` | Se `config.json` tiver chaves desconhecidas ou valores fora de range (ex.: `max_frames_per_minute: -5`), logar aviso em vez de silenciosamente usar valor inválido |
| 21 | **Modo online / ao vivo (stream)** _(2026-06-26)_ | `frame_selector.py` | Tornar a seleção de frames e o dedup **incrementais** para uma fonte ao vivo (calibração rolling + warmup, janela de dedup limitada, emit atrasado ~`MIN_GAP`, nova fonte `from_events`). Base para o `ponte_android`. Spec completa: [`planning/modo_online.md`](modo_online.md) |

---

## P2 — Nice-to-have

| # | Melhoria | Arquivo | Detalhe |
|---|---------|---------|---------|
| 13 | **Saída em formatos alternativos** | `processor.py` / CLI | Flag `--output-format vtt\|html\|json` para exportar a transcrição além do `.md` + `.jpg` |
| 14 | **Validação de integridade do cache** | `processor.py` | Ao carregar um cache, checar se `result.json` é JSON válido e se ao menos um frame existe; se não, remover e reprocessar |
| 15 | **Deinterlacing para vídeos antigos** | `processor.py` | Adicionar `-vf yadif` ao ffmpeg quando `meta["format"]` indicar vídeo interlaced (analógico/DVD) |
| 16 | **Progresso mais granular no `get_status`** | `server.py` | Além de etapa + ETA, expor `frames_done`/`frames_total` durante a extração para um progresso mais informativo no chat |
| 17 | **Aviso de ETA impreciso na 1ª execução** | `processor.py` | Se `_timings.json` não existir (primeira vez), avisar que o ETA é uma estimativa grosseira e deve ser ignorado |
| 18 | **`SPEECH_GAP_THRESHOLD` adaptativo** | `frame_selector.py` | Calcular o gap médio entre palavras do transcript e ajustar o threshold dinamicamente (voz rápida vs. apresentação pausada) |
| 19 | **Check de GPU antes de carregar Florence-2** | `frame_annotators.py` | Antes de `torch.cuda.is_available()`, verificar se o driver CUDA está funcional com um micro-tensor; capturar RuntimeError do driver quebrado separadamente de "GPU não existe" |
| 20 | **Validação de `GROQ_API_KEY` no startup** | `server.py` / `transcribe.py` | Fazer uma chamada `client.models.list()` no startup para validar a chave antes do primeiro job, em vez de descobrir só quando a transcrição falhar |

---

## Fora de escopo (por enquanto)

- Watermark/logo removal (complexidade de CV, impacto limitado)
- Suporte a múltiplos canais de áudio (mono já cobre todos os casos de uso)
- Download de playlists inteiras (cada vídeo é um job separado — isso é arquitetura)

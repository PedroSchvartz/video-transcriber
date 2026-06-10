# Video Transcriber (MCP tool)

Ferramenta MCP para o Claude Code: envie uma URL (YouTube/Instagram) ou um
`.mp4` no chat e o Claude transcreve o áudio e captura **prints relevantes** dos
momentos de mudança de fala/cena/áudio. A resposta é uma **timeline** com a fala
e os prints intercalados em ordem cronológica, como se o Claude tivesse assistido.

## Ferramentas expostas

O processamento roda em **background, num processo separado matável**:
`process_video` retorna na hora (some o "pensando" eterno) e você acompanha por
`get_status`. Uma chamada **síncrona** prenderia o turno do Claude Code por
minutos e ignoraria o botão de parar — o handler já despachado seguiria rodando
no processo do servidor MCP. Rodar o trabalho num processo à parte resolve isso
e ainda permite **cancelar de verdade** (`cancel` mata o processo e os ffmpeg
filhos).

| Ferramenta | O que faz |
|-----------|-----------|
| `process_video(source, max_screenshots=20, ocr=False, describe=False, describe_backend="auto")` | **Inicia** a transcrição num processo de background e retorna um `cache_id` na hora. `max_screenshots` é encaixado na faixa de 1–12 prints por minuto do vídeo. `ocr` extrai o **texto na tela** de cada print (RapidOCR, local) e `describe` gera uma **descrição** do conteúdo — ambos opcionais e desligados por padrão (em geral desnecessários quando o Claude já lê as imagens; úteis para tornar a timeline pesquisável). `describe_backend`: `auto` (no MCP prefere a **nuvem Groq**, não roda o modelo local pesado sem permissão), `groq` ou `local`. |
| `get_status(cache_id)` | Etapa atual (na fila / baixando / transcrevendo / capturando prints / pronto / cancelado). |
| `get_result(cache_id, start=0, count=0)` | Quando pronto, monta a **timeline** (fala + prints intercalados), **paginada por prints** — `count=0` devolve o máximo que cabe no limite de saída (~25k tokens); o rodapé informa "prints X–Y de N" e o `start` da próxima página. |
| `cancel(cache_id)` | **Interrompe** um processamento em andamento — mata o processo do job e os ffmpeg/yt-dlp filhos. |
| `get_frame(cache_id, timestamp)` | Captura **qualquer instante** de um vídeo já processado (`mm:ss`, `hh:mm:ss` ou segundos). Deixa o Claude "rebobinar" para olhar um momento que não virou print. |

## Como funciona

```
fonte (URL ou .mp4)
  → probe         lê duração/título sem baixar (valida antes de comprometer a máquina)
  → yt-dlp        baixa vídeo (até 1080p pela menor dimensão) + metadados/capítulos
  → ffmpeg        extrai áudio (Opus 24kbps 16kHz mono — ~16MB em 90min, 1 só request Groq)
  → Groq Whisper  transcreve com timestamps por palavra  (fallback: faster-whisper local)
  → seleção       início de fala + cena (auto-calibrada) + edição/pixels sujos (auto-calibrada) + onset de áudio + capítulos + rede uniforme
  → ffmpeg        extrai 1 frame por momento escolhido
  → filtros       descarta frames pretos/uniformes + dedup pHash/colorhash + pixels sujos
  → anotação      (opcional) OCR do texto na tela (RapidOCR) + descrição do conteúdo (Florence-2 local ou Groq vision)
  → timeline      fala e prints intercalados em ordem cronológica
```

O vídeo, a transcrição (`transcript.md`), os prints, `result.json` e `meta.json`
ficam num **cache** por vídeo (`_cache/<id>/`), mantendo os 10 mais recentes. É
isso que permite `get_result`/`get_frame` montarem a timeline e capturarem novos
instantes depois, sem reprocessar. O `meta.json` carimba a **versão do pipeline**
(`version`) que gerou aquele cache, e cada job grava um log próprio
(`_cache/<id>.log`) para diagnóstico. A poda do cache remove também os arquivos
irmãos (`.status.json`/`.log`) — nada de órfãos acumulando.

### Guard rails (não prender a sessão nem desperdiçar recursos)
- **Processo de background matável + `cancel`** — *o fix principal*: a sessão nunca fica refém de uma chamada longa, e dá para interromper de verdade (mata o processo e os ffmpeg filhos).
- **Progresso por etapa** — `get_status` mostra onde o job está, em vez de um "pensando" opaco.
- **ETA honesto (etapa atual + total)** — `get_status` informa *quanto falta para a etapa atual terminar* e o *total restante*, sempre presente e corrigido conforme o progresso real chega. A etapa pesada (`deteccao`: cena/edição/áudio) ganha um **sinal de progresso real** (o detector de edição reporta quantos segundos do vídeo já varreu) em vez de um *countdown* cego — então o ETA acompanha a realidade e nunca "trava em 0" fingindo que acabou. A `extracao` usa fração honesta `i/n`. As taxas de cada etapa se **auto-calibram** no histórico (`_timings.json`) a cada run.
- **`timeout` em todo subprocess** — nenhuma etapa pendura para sempre; em timeout ela degrada (segue sem aquela fonte) em vez de congelar.
- **Scene detection só por keyframes** (`-skip_frame nokey`) — num vídeo de 1h33 a etapa mais pesada de CPU caiu de *dezenas de minutos* para **~16s**.
- **Detecção de cena e de áudio em paralelo** — as duas passadas de ffmpeg são independentes e rodam concorrentes, encurtando a etapa de seleção de frames.
- **Budget de prints proporcional à duração** — entre **1 e 12 prints por minuto** de vídeo (no mínimo 1): cobertura garantida em vídeos longos sem inundar de prints os curtos.
- **Um job por vídeo** — `process_video` não dispara um segundo worker para um vídeo que já está sendo processado (evita sobrescrever o cache em andamento).
- **Áudio Opus** em vez de FLAC — cabe num único request Groq, sem fatiamento.
- **Download até 1080p pela menor dimensão** (`format_sort res:1080`) — arquivo ainda pequeno e decodificação barata, sem degradar vídeo vertical (o antigo `height<=720` baixava reels a 360p).
- **Limite duro de duração** (4h) + **probe** antes de baixar — recusa vídeos absurdamente longos sem comprometer nada.
- **Erros de serviços de rede classificados** — falhas dos serviços *não-locais* (download via yt-dlp, Groq) viram mensagens claras por categoria, sem se confundirem: **conexão** (sem internet/DNS/timeout — yt-dlp não reintenta com cookies à toa e o local assume), **limite** (free tier da Groq / HTTP 429 da plataforma — *não é erro*: traz o teto/restante/retry e degrada para o `faster-whisper` local), **cobrança** (cota paga esgotada — informa o que houve, distinto de conexão) e **conteúdo** (vídeo privado/removido/restrito). O único ponto de rede do modo "local" — baixar o modelo do `faster-whisper` na 1ª vez — também dá mensagem clara se faltar conexão.

### Fontes de prints
- **fala** — início de novas frases (word-timestamps do Whisper)
- **cena** — mudança visual (detector de cena do ffmpeg), com limiar **auto-calibrado** à distribuição de scores do próprio vídeo — pega cortes duros e também overlays/b-roll/texto na tela, sem afogar em ruído. Dentro do shot, escolhe o frame **mais nítido** (variância do Laplaciano) numa pequena janela antes da troca — evita borrão de transição e overlays em fade, pegando a tela já inteira e estável
- **edicao** — fração de pixels "sujos" entre quadros (estilo *dirty rectangles*), com **dois limiares auto-calibrados**: a *resistência à cor* por pixel (`T_cor`, derivada do ruído do próprio vídeo — grão/compressão não contam, preto→branco sim) e a *área* mínima que precisa mudar (média + desvio, adapta à velocidade de edição). Complementa a cena pegando edições que o `scene` do ffmpeg perde
- **audio** — onset sonoro após pausa, com limiar **adaptativo** ao volume do vídeo (efeitos, música, aplausos)
- **capitulo** — marcadores do YouTube, quando existirem
- **intervalo** — rede de segurança uniforme, garante cobertura quando falta fala/cena

### Anotação opcional dos prints (OCR + descrição)
Passo **opcional, desligado por padrão** (`ocr=True` / `describe=True` na tool, ou
`--ocr` / `--describe` na CLI), rodado depois do dedup e **só nos prints que
sobraram**. Transforma a imagem em **texto** — útil para tornar a timeline/`transcript.md`
**pesquisável** e essencial no **uso sem IA** (onde ninguém "lê" os `.jpg`):
- **OCR** — texto na tela via **[RapidOCR](https://github.com/RapidAI/RapidOCR)** (ONNX, local, gratuito, leve).
- **descrição** — legenda do conteúdo, com **dois backends**:
  - **local** — **[Florence-2](https://huggingface.co/microsoft/Florence-2-base)** (Microsoft, MIT, 0.23B, CPU; baixa ~0.5GB na 1ª vez). Offline e grátis, mas puxa `torch` e RAM.
  - **nuvem** — **[Groq vision (Llama-4)](https://console.groq.com/docs/vision)**, reusando a mesma `GROQ_API_KEY` e o tratamento de limite/erro do Whisper. **1 requisição por print**; free-tier limitado (sai da máquina).

Ambos são **locais e offline**. As dependências são pesadas (o Florence-2 puxa
`torch`), então **não** entram no `requirements.txt` principal — ficam comentadas
lá. Se faltarem, o passo é **pulado com aviso claro**, nunca derruba o pipeline. O
resultado vai para o `result.json`, o `transcript.md` e a timeline (linhas
`descrição:` / `texto na tela:` sob cada print).

Salvaguardas dessa etapa:
- **Escolha de backend por hardware (descrição):** o padrão é **não rodar o modelo
  local sem o usuário querer**. A CLI **mede a RAM** e decide — `≥2 GB` roda **local**;
  `1,5–2 GB` **pede permissão** (cabe apertado); `<1,5 GB` ou `torch` quebrado → **não
  roda local** e **direciona para a nuvem (Groq)**. No MCP (não-interativo, sem como
  pedir permissão) o `auto` **prefere a nuvem**. Backends: `auto` / `local` / `online`
  (`--describe-backend` na CLI; `describe_backend` na tool).
- **Pré-check de hardware** antes do Florence-2: confere que o `torch` está funcional
  (um install pela metade importa mas estoura ao usar) e a RAM disponível.
- **Confirmação antecipada (CLI)**: ao pedir `--ocr`/`--describe`, a CLI diz logo se
  cada anotador está `PRONTO`, vai pra `NUVEM` ou `SERÁ PULADO` (com o motivo) —
  *antes* de baixar/transcrever, para você não esperar até o fim para descobrir.
- **Erro/limite da descrição na nuvem** reusa a classificação do Groq: um 429 vira
  `limite` (não-erro), e a descrição online é interrompida nos prints restantes em
  vez de insistir e queimar a cota.
- **Filtro de ruído no OCR**: descarta detecções sem letra latina nem dígito (ícones
  e gráficos viram caracteres CJK com confiança alta no OCR), preservando PT/EN com
  acentos, números e URLs.
- O `rapidocr-onnxruntime` traz `numpy>=2` (o pipeline é compatível).

## Setup

1. **ffmpeg** (já instalado via `winget install Gyan.FFmpeg`). O código localiza
   o binário automaticamente mesmo se ainda não estiver no PATH.
2. **Dependências Python**: `pip install -r requirements.txt`
3. **Chave Groq** (recomendado — transcrição ~200x mais rápida e muito mais precisa):
   - Crie de graça em https://console.groq.com/keys (sem cartão)
   - Copie `.env.example` para `.env` e cole a chave em `GROQ_API_KEY=`
   - **Sem chave**, a ferramenta usa `faster-whisper` local (mais lento, já funciona).

## Registro no Claude Code

Registrado **globalmente** (escopo `user`, em `~/.claude.json`), disponível em
qualquer projeto:

```
claude mcp add video-transcriber -s user -- python "C:\Codigos\claud_develop\tools\video_transcriber\server.py"
```

Após **reiniciar** o Claude Code, rode `/mcp` para confirmar que
`video-transcriber` aparece ativo (com `process_video`, `get_status`,
`get_result`, `cancel` e `get_frame`). Para remover:
`claude mcp remove video-transcriber -s user`.

## Uso

No chat, é só pedir — o Claude orquestra `process_video` → `get_status` → `get_result`:

```
Transcreve esse vídeo: https://www.youtube.com/watch?v=...
Transcreve C:\caminho\para\video.mp4
```

Como roda em background, em vídeos longos o Claude inicia, vai checando o status
e entrega a timeline quando termina. Para investigar um momento específico:

```
Mostra o que aparece em 2:15 desse vídeo
```

(o Claude usa `get_frame` com o `cache_id` da resposta anterior).

## Uso sem IA (direto no terminal)

Dá para rodar o pipeline sozinho, sem o Claude/MCP:

```
python transcribe.py "<URL ou caminho .mp4>"
```

Roda síncrono, mostra o progresso e salva `transcript.md` + `frames/` no cache.
Passo a passo completo (pré-requisitos, opções, onde fica o resultado) em
[`USO_SEM_IA.md`](USO_SEM_IA.md).

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `server.py` | MCP server — tools `process_video`/`get_status`/`get_result`/`cancel`/`get_frame`; lança o worker e monta a timeline |
| `transcribe.py` | **CLI sem IA** — roda o pipeline direto no terminal (ver [`USO_SEM_IA.md`](USO_SEM_IA.md)) |
| `worker.py` | Processo de background matável — roda um job de transcrição atualizando o status |
| `processor.py` | Pipeline (probe, download, áudio, transcrição, frames) + cache + status + `run_job`/`cancel` |
| `frame_selector.py` | Seleção de timestamps (fala/cena/edição/áudio/capítulo/intervalo) + dedup |
| `frame_annotators.py` | **Anotação opcional dos prints** — OCR (RapidOCR) + descrição (Florence-2 local ou Groq vision), com pré-check de hardware |
| `ffmpeg_tools.py` | Localiza ffmpeg/ffprobe no PATH |

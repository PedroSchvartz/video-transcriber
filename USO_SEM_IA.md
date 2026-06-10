# Usar o Video Transcriber **sem IA**

Normalmente é o Claude que dirige esta ferramenta (via MCP): você manda a URL no
chat e ele monta a timeline. Mas o motor — **baixar → extrair áudio → transcrever
→ capturar prints** — roda sozinho. Este guia mostra como usá-lo direto no
terminal, **sem nenhuma IA envolvida**.

---

## 1. Pré-requisitos (uma vez só)

1. **Python 3.10+** e as dependências:
   ```
   pip install -r requirements.txt
   ```
2. **ffmpeg** — instale com `winget install Gyan.FFmpeg`. A ferramenta localiza o
   binário sozinha, mesmo que ainda não esteja no PATH.
3. **Chave Groq** *(opcional, recomendado)* — transcrição muito mais rápida e
   precisa. Crie de graça em <https://console.groq.com/keys>, copie
   `.env.example` para `.env` e cole a chave em `GROQ_API_KEY=`.
   - **Sem chave**, a transcrição usa o `faster-whisper` local: funciona offline,
     porém mais lento.
4. **Anotação dos prints** *(opcional — só se for usar `--ocr` / `--describe`)*:
   ```
   pip install rapidocr-onnxruntime          # OCR (texto na tela) — leve, ONNX
   pip install torch transformers timm einops # descrição LOCAL (Florence-2) — pesado (~torch)
   ```
   São **locais e gratuitas**; o Florence-2 baixa ~0.5GB do modelo na 1ª vez. Ficam
   de fora do `requirements.txt` principal de propósito (são pesadas). **A descrição
   na nuvem (`--describe-backend online`) não exige nada disso** — usa o pacote `groq`
   (já instalado) e a sua `GROQ_API_KEY`. Sem RAM nem `torch`.

---

## 2. Rodar

```
python transcribe.py "<URL ou caminho do arquivo>"
```

Ou **sem argumento nenhum** — aí ele pergunta a fonte (e quantos prints) no próprio
terminal, repetindo até você colar algo válido:

```
python transcribe.py
> Cole a URL (YouTube/Instagram) ou o caminho do .mp4: <você cola aqui>
> Quantos prints? [Enter = 20]:
```

Exemplos:

```bash
# YouTube
python transcribe.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Instagram (reel/post)
python transcribe.py "https://www.instagram.com/reel/XXXXXXXXX/"

# Arquivo local
python transcribe.py "C:\videos\minha_aula.mp4"
```

Quer mais ou menos prints? Use `--screenshots` (ou `-s`):

```
python transcribe.py "..." --screenshots 30
```

> O número é **encaixado entre 1 e 12 prints por minuto** de vídeo. Ex.: num vídeo
> de 5 min, o intervalo válido é 5 a 60; pedir 30 mantém 30, pedir 200 vira 60.

**Transformar os prints em texto (opcional):** sem IA, os prints são só `.jpg` que
você abre na mão. Estas flags extraem o conteúdo das imagens **para texto** (vai pro
`transcript.md`, deixando-o pesquisável):

```
python transcribe.py "..." --ocr                              # texto na tela (RapidOCR, local)
python transcribe.py "..." --describe                          # descrição (backend automático)
python transcribe.py "..." --describe --describe-backend online # descrição na NUVEM (Groq)
python transcribe.py "..." --ocr --describe
```

> No modo interativo ele pergunta por essas opções.

**A descrição escolhe o backend pela sua RAM** (o padrão é **não rodar o modelo local
pesado sem você querer**):
- **≥ 2 GB livres** → roda **local** (Florence-2).
- **1,5–2 GB** → **pede permissão** (cabe apertado, pode deixar a máquina lenta).
- **< 1,5 GB** (ou `torch` ausente/quebrado) → **não roda local** e **vai pra nuvem**
  (Groq vision), se houver `GROQ_API_KEY`; senão é pulada com aviso.

Force com `--describe-backend local` (só local; nunca vai pra nuvem sozinho) ou
`online` (direto pro Groq, sem depender da sua máquina).

**Confirmação antes do trabalho pesado:** a CLI imprime logo no início o que cada
anotador vai fazer — e *por quê* — antes de baixar/transcrever:

```
Anotacao dos prints:
  - OCR (RapidOCR) ........... PRONTO
  - Descricao: ONLINE (Groq vision) — nao da pra rodar local (1.6GB livres).
```

> O **local** (Florence-2) precisa de `torch` (veja o passo 1); a **nuvem** (Groq) não
> precisa instalar nada além do `groq` (já vem) e reusa sua `GROQ_API_KEY`, mas envia
> cada print pra nuvem (free-tier limitado, 1 requisição por print). O OCR ainda
> **filtra ruído** (ícones/gráficos que viram caracteres estranhos), mantendo texto
> latino, números e URLs.

O comando roda **de forma síncrona**: ele mostra o progresso (baixando,
transcrevendo, detectando cenas...) e termina sozinho. Em vídeos longos pode levar
alguns minutos — é só esperar no terminal.

---

## 3. Serviços acessados (o que sai da sua máquina)

Ao rodar, **antes de tocar qualquer coisa**, o comando lista os serviços que aquela
execução vai acessar e o custo de cada um — sensível ao contexto (URL vs. arquivo
local, com vs. sem chave Groq):

| Serviço | Quando | Natureza |
|---------|--------|----------|
| **yt-dlp** → YouTube/Instagram/etc | só com URL | **Gratuito** — ferramenta local, mas **baixa da internet** (sujeito a bloqueio/limite da plataforma) |
| **ffmpeg / ffprobe** (áudio, frames, detecção) | sempre | **Local** — offline, gratuito |
| **Groq Whisper API** (`api.groq.com`) | se houver `GROQ_API_KEY` | **Gratuito porém limitado** — free tier; **envia o áudio para a nuvem**; tem limite por minuto/dia |
| **faster-whisper** (modelo `base`) | sem `GROQ_API_KEY` (ou como fallback) | **Local** — offline, gratuito; **na 1ª vez baixa ~140 MB** do modelo (Hugging Face) |
| **RapidOCR** (texto na tela) | só com `--ocr` | **Local** — offline, gratuito (ONNX) |
| **Florence-2** (descrição **local**) | `--describe` com RAM folgada | **Local** — offline, gratuito (MIT); **na 1ª vez baixa ~0.5 GB** do modelo (Hugging Face) |
| **Groq vision / Llama-4** (descrição **na nuvem**) | `--describe` com RAM baixa ou `--describe-backend online` | **Gratuito porém limitado** — free tier; **envia cada print para a nuvem**; 1 requisição por print |

**Nenhum serviço é pago:** o único de nuvem é o Groq, que tem free tier. Com arquivo
local, a linha de download some (`NÃO USADO` — nada sai pela rede para obter o vídeo).

### Instagram / conteúdo logado
Não há código específico de Instagram — é o extractor do `yt-dlp`. Conteúdo **público**
baixa direto. Conteúdo **privado / que exige login** (reel/post) cai num fallback que
**lê os cookies do seu Chrome logado** (acesso *local* ao navegador) e baixa
**autenticado como você**, reusando a sua sessão. Só funciona se você **usa o Chrome**
e está **logado no site**. Em versões recentes do Chrome no Windows, a leitura pode
falhar se o navegador estiver **aberto** (lock do banco de cookies).

---

## 4. Quando um serviço de rede falha

Serviços **locais** (ffmpeg, faster-whisper já baixado) não dependem de rede. Os
**não-locais** (download, Groq) podem falhar — e a ferramenta **classifica o motivo**
em vez de cuspir um stack trace, sem confundir as causas:

| Categoria | O que é | O que acontece |
|-----------|---------|----------------|
| **Conexão** | sem internet/DNS/timeout (não chegou no servidor) | mensagem clara; no download não reinsiste com cookies à toa; na Groq, **transcreve offline** no local |
| **Limite** | free tier da Groq (HTTP 429) ou 429 da plataforma | **não é erro** — informa o **teto / restante / quando libera** e degrada para o `faster-whisper` local |
| **Cobrança** | cota paga / crédito esgotado | informa o que houve (distinto de "sem conexão") |
| **Autenticação** | `GROQ_API_KEY` inválida ou sem permissão | avisa que a chave não vale e **transcreve no local** — não trava |
| **Conteúdo** | vídeo privado/removido/restrito | diz que o vídeo está indisponível, não "erro de rede" |
| **Cookies** | falha ao ler os cookies do Chrome (Chrome **aberto**, DPAPI) para conteúdo logado | pede para **fechar o Chrome** e confirmar que você está logado no site — não trata como "erro de rede" |

Pontos importantes:
- **Passar do limite ≠ falha.** Se estourar o free tier da Groq, ele **não** trata como
  erro: avisa o limite e termina pelo motor local — você ainda recebe a transcrição.
- **O modo local também tem 1 ponto de rede:** baixar o modelo `faster-whisper` na 1ª
  vez. Sem conexão (e sem `GROQ_API_KEY`), a mensagem diz exatamente isso, em vez de
  estourar uma stack.
- **Antes de tocar a rede**, a fonte é validada: um link que **não é de vídeo** ou um
  arquivo **corrompido/incompleto** param na hora com mensagem clara ("o arquivo não
  parece ser um vídeo válido… tente outro ou baixe de novo"), sem disparar download
  nem cuspir erro do ffmpeg.

---

## 5. Onde fica o resultado

Tudo é salvo em `_cache/<id-do-video>/`. Ao terminar, o próprio comando imprime os
caminhos. Dentro da pasta:

| Arquivo | O que é |
|---------|---------|
| `transcript.md` | A **transcrição** com timestamps + a lista de prints capturados |
| `frames/` | Os **prints** em `.jpg`, nomeados com o instante e o motivo (`fala`/`cena`/`edicao`/`audio`/`capitulo`/`intervalo`) |
| `result.json` | Segmentos da fala + prints, em formato estruturado |
| `meta.json` | Título, fonte, duração, motor de transcrição e a **versão** do pipeline |

**Como ler:** abra o `transcript.md` no seu editor — ele tem a fala com horários e
aponta cada print. Os `.jpg` ficam na pasta `frames/`, na ordem cronológica.

---

## 6. Diferença para o uso *com* IA

| | **Com IA** (Claude/MCP) | **Sem IA** (este CLI) |
|---|---|---|
| Como roda | em background, o Claude acompanha | síncrono, você espera no terminal |
| Resultado | "timeline" (fala + imagens) montada **no chat** | você abre o `transcript.md` e os `.jpg` |
| "Rebobinar" um instante | Claude usa `get_frame` | você roda um comando (veja abaixo) |

É o **mesmo conteúdo** — a diferença é que, sem IA, é você quem lê o arquivo e abre
as imagens, em vez de o chat montar tudo para você.

### Capturar um instante específico depois

Para pegar um frame de um segundo qualquer de um vídeo **já processado** (o que o
`get_frame` faz):

```
python -c "import processor; print(processor.get_frame_from_cache('<cache_id>', 75))"
```

(`75` = o segundo desejado; o comando imprime o caminho do `.jpg` gerado.)

---

## 7. Detalhes úteis

- **Instagram / conteúdo logado e cookies do Chrome:** veja a **seção 3** (serviços).
- **Erros de rede (sem conexão, limite, cobrança):** veja a **seção 4**.
- **Limite de duração:** 4 h (trava de segurança). Corte um trecho antes se for
  maior.
- **Qualidade do download:** até **1080p** pela menor dimensão (arquivo ainda
  pequeno; não degrada vídeo vertical/reels).
- **Cache:** mantém os **10 vídeos mais recentes**; os mais antigos são removidos
  automaticamente (junto com os arquivos irmãos `.status.json`/`.log`).
- **Sem áudio falado?** O vídeo ainda gera prints (cena/intervalo) — a transcrição
  fica vazia, mas você vê o conteúdo visual.

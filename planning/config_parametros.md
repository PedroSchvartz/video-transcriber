# Spec: Parâmetros configuráveis via `.env` / `config.json` / CLI

**Status:** Implementado  
**Versão do pipeline:** 1.10.x

---

## Problema

Constantes numéricas relevantes (`min/max frames por minuto`, duração máxima, tamanho
do cache, limiar de silêncio) estavam hard-coded nos módulos Python. Não havia forma
de ajustá-los sem editar o código.

## Solução

Novo módulo `config.py` com precedência:

```
CLI args  >  variáveis de ambiente VT_*  >  config.json  >  defaults hard-coded
```

## Parâmetros configuráveis

### Batch (pipeline sobre arquivo)

| Chave (`config.json`) | Variável de ambiente | Default | Descrição |
|-----------------------|---------------------|---------|-----------|
| `min_frames_per_minute` | `VT_MIN_FRAMES_PER_MINUTE` | `1` | Piso de prints/min de vídeo |
| `max_frames_per_minute` | `VT_MAX_FRAMES_PER_MINUTE` | `12` | Teto de prints/min (`9999` = sem limite) |
| `max_duration_hours` | `VT_MAX_DURATION_HOURS` | `4.0` | Duração máxima aceita (horas) |
| `max_cache_entries` | `VT_MAX_CACHE_ENTRIES` | `10` | Vídeos mantidos no cache (LRU) |
| `silence_threshold_dbfs` | `VT_SILENCE_THRESHOLD_DBFS` | `-50.0` | Limiar de silêncio (dBFS) |

### Modo online / stream — perfil **TELA** (PC/Android)

Governam a seleção de frames ao vivo (`stream_selector.OnlineSelector`). Lidos no
`__init__` (não no import), então overrides de runtime valem; um argumento explícito
ao construtor vence o config.

| Chave (`config.json`) | Variável de ambiente | Default | Descrição |
|-----------------------|---------------------|---------|-----------|
| `stream_t_cor` | `VT_STREAM_T_COR` | `50.0` | Distância de cor p/ um pixel contar como "diferente" |
| `stream_warmup_s` | `VT_STREAM_WARMUP_S` | `2.0` | Segundos de baseline fria antes do 1º emit |
| `stream_tau_calib_s` | `VT_STREAM_TAU_CALIB_S` | `8.0` | Constante de tempo (s) da EMA de calibração de área |
| `stream_area_k` | `VT_STREAM_AREA_K` | `2.0` | Limiar de área = `EMA_média + K·EMA_desvio` |
| `stream_area_floor` | `VT_STREAM_AREA_FLOOR` | `0.02` | Piso absoluto: fração mínima da tela p/ contar como mudança |
| `stream_min_gap_s` | `VT_STREAM_MIN_GAP_S` | `0.8` | Quietude (s) que fecha um grupo de mudança (assentou) |
| `stream_dedup_window` | `VT_STREAM_DEDUP_WINDOW` | `8` | Emissões recentes comparadas no dedup |

**De onde vêm estes defaults (a calibração que já tínhamos).** O modo online
**auto-recalibra** o limiar de área ao padrão da própria tela ao vivo (EMA da fração
suja) — decisão de projeto registrada em [`modo_online.md`](modo_online.md) ("#5
dissolvida na #1"). Por isso os valores abaixo **não são constantes cravadas de um
corpus de UI** (que não temos): são os meta-parâmetros da auto-calibração + o piso
absoluto que ancora o regime morto-estático. A origem de cada um:

- **Herdados da calibração de vídeo** (`frame_selector`): `t_cor` (`DEDUP_T_COR=50`),
  `area_k` (`EDIT_AREA_K=2.0`), `area_floor` (`EDIT_AREA_FLOOR=0.02`). No regime
  estático a baseline relativa → 0, então quem decide é o **piso** — daí ele ser a
  âncora que herdamos direto do batch.
- **Ajustado pela diferença de regime:** `min_gap` **0.8 s** (vs. 1.5 s do vídeo). O
  vídeo precisa de 1.5 s porque o movimento é contínuo; a transição de UI é **discreta
  e rápida** (~0.2–0.4 s), então a tela assenta antes e o emit pode sair mais cedo. Único
  default que mudei do valor de vídeo, e por raciocínio de regime, não por chute.
- **Próprios do online:** `warmup_s` (2 s, faixa 2–5 s da spec) e `dedup_window` (8
  emissões — bound de memória que o batch O(n²) não tinha).

**O que ainda falta MEDIR (não cravar sem dado).** `area_floor` é o parâmetro que a
spec (#5) marca como sensível a UI: numa gravação de tela real ele pode querer subir
para **0.03–0.05** (evitar disparo em cursor/tooltip/micro-animação). Mantive a âncora
de vídeo (0.02) porque não há corpus de UI commitado; quando houver uma gravação de
tela real, medir a distribuição da fração suja e ajustar por aqui — sem tocar código.

## Como usar

### Via `.env` (persiste entre execuções)
```
# em .env:
VT_MAX_FRAMES_PER_MINUTE=5      # máximo de 5 prints por minuto
VT_MAX_FRAMES_PER_MINUTE=9999   # sem limite
```

### Via `config.json` (opcional; versionável)
```json
{ "max_frames_per_minute": 5 }
```

### Via CLI (só para a execução atual)
```bash
python transcribe.py video.mp4 -s 200 --no-limit     # sem cap de prints/min
python transcribe.py video.mp4 --max-fpm 5            # máximo 5/min
python transcribe.py video.mp4 --min-fpm 2 --max-fpm 8
```

### Modo interativo
Ao rodar `python transcribe.py` sem argumentos, o CLI pergunta os limites:
```
Limites de prints por minuto de video (Enter = usar valor atual):
  Min por minuto [1]:
  Max por minuto [12] (0 = sem limite):
```

## Arquivos modificados

| Arquivo | O que mudou |
|---------|------------|
| `config.py` | **Criado** — módulo central de configuração |
| `config.json.example` | **Criado** — exemplo de config.json |
| `.env.example` | Adicionado bloco `VT_*` comentado |
| `frame_selector.py` | `screenshot_budget()` lê `config.get()` em runtime |
| `processor.py` | `MAX_DURATION_HARD`, `MAX_CACHE_ENTRIES`, `SILENCE_THRESHOLD_DBFS` lidos de `config.load()` |
| `transcribe.py` | Flags `--no-limit`, `--min-fpm`, `--max-fpm`; bloco interativo de limites |

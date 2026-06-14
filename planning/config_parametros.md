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

| Chave (`config.json`) | Variável de ambiente | Default | Descrição |
|-----------------------|---------------------|---------|-----------|
| `min_frames_per_minute` | `VT_MIN_FRAMES_PER_MINUTE` | `1` | Piso de prints/min de vídeo |
| `max_frames_per_minute` | `VT_MAX_FRAMES_PER_MINUTE` | `12` | Teto de prints/min (`9999` = sem limite) |
| `max_duration_hours` | `VT_MAX_DURATION_HOURS` | `4.0` | Duração máxima aceita (horas) |
| `max_cache_entries` | `VT_MAX_CACHE_ENTRIES` | `10` | Vídeos mantidos no cache (LRU) |
| `silence_threshold_dbfs` | `VT_SILENCE_THRESHOLD_DBFS` | `-50.0` | Limiar de silêncio (dBFS) |

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

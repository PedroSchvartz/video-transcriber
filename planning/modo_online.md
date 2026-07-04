# Spec: Modo online / ao vivo (stream)

**Status:** Planejado  
**Origem:** necessidade do `ponte_android` (controle de Android pelo PC) — ver
[`../../ponte_android/planning/plano.md`](../../ponte_android/planning/plano.md).  
**Data:** 2026-06-26

---

## Problema

O pipeline atual é **batch sobre um arquivo fechado**: a auto-calibração de
`detect_scenes` e `detect_edits` deriva o limiar de `média + K·desvio` sobre a
distribuição do **vídeo inteiro** — precisa do todo. Para o `ponte_android` queremos a
mesma seleção de frames + dedup sobre uma fonte **ao vivo** (a tela do celular),
emitindo à medida que a sessão acontece.

## O que NÃO precisa mudar

- **Detecção-de-mudança crua** pode vir da própria captura: o
  [minicap](https://github.com/DeviceFarmer/minicap) (Android 4.2+) já manda frame **só
  quando a tela muda**. O valor único do `video_transcriber` é a camada **semântica** —
  não a detecção.
- **Tela muda = transcrição vazia** já é tratado com elegância (guard de silêncio em
  `processor.py`; `_empty_transcript`).

## Mudanças

| # | Mudança | Onde | Detalhe |
|---|---------|------|---------|
| 1 | **Calibração online** | `frame_selector.py` | Trocar `média + K·desvio` batch por estatística incremental (Welford) ou EMA, com **janela de warmup** (~2–5 s) antes do primeiro emit. Você não precisa da distribuição *final*, só de uma estimativa corrente boa o suficiente. |
| 2 | **Janela de dedup limitada** | `frame_selector.py` (`dedup_visual`) | Hoje compara cada frame com **todos** os já aceitos (`sigs` cresce sem limite → O(n²) + memória). Ao vivo: dedup contra uma **janela recente** (últimas K telas / últimos T s). Crítico numa máquina de 8 GB. |
| 3 | **Emit atrasado ~`MIN_GAP`** | `frame_selector.py` (`_marks_from_changes`, `pick_sharpest_time`) | A escolha de frame é retrospectiva de propósito (prefere a mudança mais tardia do grupo + frame mais nítido da janela). Ao vivo, o emit fica ~`MIN_GAP` atrás da borda para a tela assentar — parte do delay vem daqui, não da rede. |
| 4 | **Nova fonte `from_events`** | `frame_selector.py` | Marca de frame a partir de uma **lista de ações** (taps/swipes/logcat/console) com timestamp — mesmo formato de `from_chapters`. Substitui a fala como trilha textual da timeline. No celular a intenção é **observável** (você emitiu o tap), não inferida. |
| 5 | **Re-tunar constantes p/ UI estática** | `frame_selector.py` | `SCENE_SCORE_FLOOR=0.06`, fatores de ruído etc. foram calibrados p/ **vídeo** (movimento contínuo). UI é o regime oposto (estática + saltos discretos): no estático o termo adaptativo some e quem decide é o **FLOOR**. **Medir a distribuição numa gravação de UI real antes de cravar** (relaciona com o item 9 do backlog). |

## Forma de uso (esboço)

- **Timeline pós-sessão:** grava a sessão (`screenrecord` / `scrcpy --record`) →
  pipeline com `from_events(actions_log)` no lugar da fala. Reusa quase tudo como está.
- **Stream disparado por mudança:** detectores em modo incremental sobre a fonte ao
  vivo; emite frame deduplicado quando a tela muda. As 5 mudanças acima habilitam isto.

> **Ambos os modos consomem o mesmo schema de ação.** O schema do `from_events` e a
> sincronização de relógio (tempo-PC + `t0` na gravação) estão no contrato em
> [`../../ponte_android/planning/plano.md`](../../ponte_android/planning/plano.md).

## Princípio

O `video_transcriber` continua dono dos **primitivos** (`dedup_visual`, `drop_blank`,
`pick_sharpest_time`, `_marks_from_changes`); o modo online os torna **incrementais**. O
`ponte_android` consome — não duplica.

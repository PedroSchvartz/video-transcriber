# Vigilância por anomalia — plano de desenvolvimento

**Status:** Plano / proposta — **nada construído**. Direção nova, derivada da investigação de foco.
**Data:** 2026-06-23
**Linhagem:** `focus_locator.py` → [discriminador handheld](handheld_discriminator.md) → isto.

> **Ressalva de escopo (convenção do hub):** isto é um **produto novo** (detector de
> anomalia em câmera ao vivo), **não** uma peça do `video_transcriber`. Reusa os
> *primitivos de análise* (correlação de fase, `change_signature` por blocos,
> deriva/jitter), não o pipeline de transcrição. Se graduar de plano para código, vira
> **projeto próprio na raiz** (`claud_develop/<projeto>/`). Por ora o *plano* mora aqui
> pela linhagem conceitual.

---

## Tese (a chave de abóbada)

**A anomalia está nos pixels — relativa ao normal aprendido (local + temporal). A ameaça
não está. Quem julga ameaça é o humano.** A câmera detecta *desvio*; a pessoa interpreta
*perigo*. É a tese-mãe do foco vestida de vigilância: **localizar e rotular** o que
desvia, **não fazer gatekeeping**.

O objetivo **não** é alarmar só em emergência — é alarmar quando **algo transborda do
previsto**. O "todo mundo levantou fora de hora pra cantar parabéns" alarmando
*corretamente* é o teste de que o conceito está certo: benigno **e** anômalo, e o alarme
é o comportamento desejado.

## O que é / o que não é

**É:** um *front-end* de detecção de anomalia robusto a tremor — ingere stream ao vivo,
cancela ego-movimento da câmera, e dispara sobre desvio do normal aprendido por região,
com `bbox` e razão. O valor distintivo é estreito e real: **gating robusto a tremor +
normal aprendido**, justo o que CCTV barato faz mal (afoga em falso-alarme de vento/IR,
ou sobe o limiar e fica cego).

**Não é:** classificação (pessoa vs gato vs sacola), rastreamento de identidade,
zonas/regras (linha de intrusão, loitering), gravação/retenção, infra de alerta. Isso já
existe maduro (Frigate, ZoneMinder, Blue Iris, analytics ONVIF, YOLO em Coral). Reinventar
seria YAGNI. Este sistema é **um estágio** antes de classificação/tracking — não o sistema.

## Arquitetura

### 1. Ingestão ao vivo (loop RTSP)
Troca o pipeline batch (arquivo finito) por **stream sem fim**. ffmpeg já lê RTSP nativo
(o `read_frames` aponta pra `rtsp://...`), com **drop-on-lag** (descarta frame se atrasar,
pra não acumular). Sem download, sem duração conhecida, sem seleção guiada por transcript.

### 2. Ego-movimento global barato + gate de perfil de tremor
Correlação de fase do **quadro inteiro** (sub-pixel, refino parabólico) → cancela o
balanço da câmera. O **gate de perfil de tremor** recusa tratar como ego-movimento um
deslocamento súbito/grande/*fora do perfil* oscilatório conhecido da câmera — é o que
defende o caso "objeto chegou perto e sequestrou a estimativa" (o falso-negativo perigoso).
Lógica a **reencarnar** do scratch (`phase_vec_sub`, `frac_shift`, `best_register`,
deriva/jitter) como código de verdade — hoje só vive em `_test_*` gitignored.

### 3. O mapa — duas faces
- **Global-temporal:** o ritmo da *própria câmera* (quanto treme) e a agenda de luz
  (dia/crepúsculo/noite-IR). Transição global previsível = **degrau simultâneo no quadro
  todo** (luminância ou colapso de croma no IR), sem localização → o `bbox`/espalhamento
  já separa global de local. Disciplina: **segura o alarme durante a transição,
  re-baseline depois** (janela cega curta, mas anunciada).
- **Espacial-zonal:** modelo do *normal de cada região* — a **assinatura** da mudança
  esperada (oscilatória-recorrente = árvore/bandeira, ignorável; transitória-persistente =
  algo chegou). **Dois botões distintos, não confundir:**
  - **precisão** (sub-pixel) — gasta onde movimento pequeno importa (zona *distante*);
  - **tensão / tolerância** — definida pelo *comportamento aprendido* (a árvore baixa a
    tensão, mesmo distante). Distante-e-quieto e distante-e-balançando querem tratamentos
    opostos; o mapa é o **produto** geometria × comportamento.
- **Prior de cor/aparência por ponto**, **condicionado ao estado de luz** (morre no IR
  cinza — tem que saber em qual paleta está). Camada **mole**: "essa paleta não é a daqui"
  (ex.: empresa de uniforme verde-e-amarelo vs pessoa toda de preto). Serve **vestuário
  controlado**; forjável; não generaliza pra vestuário livre.
- **Saída = surpresa relativa ao normal local+temporal**, não mudança absoluta.

### 4. Calibração (warm-up declarado)
**Janela cega assumida** — não há como calibrar sem abrir mão de vigiar; a espera é
planeável. Declará-la converte o *envenenamento silencioso de fundo* em **procedimento
operacional** (o humano reforça a vigilância no período; a responsabilidade volta a quem
já a tem). ~**72h ≈ 3 ciclos diurnos** (mínimo pra aprender que o normal *varia com a
hora*). Na largada o usuário escolhe o preset: **mais sensível + mais falsos** /
**menos sensível + mais preciso** / **geralzão** (padrão).

Amostrador (reusa o filtro que já existe): a cada **15 min**, grava **1 min @ 2fps**,
escolhe o `d` padrão pelo preset, extrai frames; o `change_signature` **mantém só os
relevantes**. Constrói o mapa rico em contexto, barato. *Resíduo honesto:* a calibração
**define** o normal → é uma superfície que pode ser **encenada** por quem a observa.

### 5. Detecção ao vivo
Por trio/cadência: estima ego-movimento, registra, mede `change_signature` por zona,
compara ao normal aprendido da zona (assinatura de movimento + prior de cor, condicionado
ao estado temporal) → emite **evento de anomalia com `bbox` + razão**. Gating barato
(scan de baixa cadência → rajada-trio só nas janelas de `d` alto), reusando a ideia do
gate do `focus_locator`.

### 6. Feedback humano (active learning) — com disciplina anti-catraca
A cada alarme o usuário marca **útil / inútil**. **Dois tipos de edição, opostos em risco:**
- **Cirúrgica (segura):** "esse funcionário também vai ao banheiro" → *adiciona um padrão
  específico* (rota/zona/horário/aparência) como normal. Pontual.
- **Geral (perigosa):** "inútil → baixa a sensibilidade" → *afrouxa a zona inteira*.

Preferir sempre a cirúrgica; **racionar** a geral. *Limite honesto:* o mapa **não conhece
o funcionário** (sem classificação de identidade — a parede semântica). Aprende um
*padrão*, então **cada exceção é um buraco com formato de padrão, não de pessoa** — um
intruso que **casa o padrão** herda a isenção. Padrão é forjável.

## Princípio de segurança transversal

**Governar a assimetria do observável.** O falso-**positivo** é *visto* (toca à toa, o
usuário corrige); o falso-**negativo** é *cego* (o intruso perdido não gera sinal). Todos
os canais que aprendem — calibração, mapa de cor, downweight adaptativo, feedback humano —
têm a **mesma deriva embutida pra cegueira**, porque aprendem *do que aparece*, e o que
mais importa (o que foi perdido) é o que não aparece.

A disciplina é **adaptação assimétrica**: **lenta pra relaxar, rápida pra re-armar**; teto
em quanto o feedback pode baixar; re-sensibilização periódica forçada; a tensão de uma zona
**decai de volta pro sensível** a menos que o movimento habitual a re-justifique. É o
"preguiçoso, não negligente" aplicado ao tempo. **Essa contra-pressão é a engenharia de
verdade — não os limiares.**

## Limites honestos (carry-forward)

- **Parede semântica:** detecta desvio, não ameaça nem identidade. O humano julga.
- **Noite / IR:** piso de ruído sobe; o prior de cor morre no cinza. Precisa de denoise /
  piso modelado e de condicionar tudo ao estado de luz.
- **Ciclo semanal:** 72h pega o diário, é **cego ao semanal** (sábado vazio). Quer ~1
  semana, dia-da-semana como *feature*.
- **Normal raro sub-amostrado:** 1 min a cada 15 aprende bem o comum, **mal o raro** (a
  entrega esporádica vira "anomalia"). Molda o perfil de falso-alarme; o feedback conserta.
- **Prior de cor** só serve a vestuário controlado; forjável.
- **Calibração encenável:** quem molda a janela molda o normal.
- **Frequência real não medida:** todo o premiss depende de footage de câmera **fixa que
  treme** (vento/vibração) — raro e ainda não validado (mesmo gargalo do handheld).

## Faseamento (com gate de validação por fase)

- **Fase 0 — fechar o buraco empírico.** Conseguir footage real de câmera fixa que **treme**
  (estabilização off, vento/vibração). Sem isso, o premiss é só sintético. **Gate:** o
  ego-movimento + gate de jitter derrubam o `d` de tremor real num feed fixo.
- **Fase 1 — primitivos reencarnados (offline).** Tirar do scratch pra código real:
  leitor de stream com drop-on-lag, `phase_vec_sub`/`frac_shift`/`best_register`,
  deriva/jitter, `change_signature` por zona. Roda em clipes gravados primeiro.
- **Fase 2 — o mapa** (espacial-zonal + global-temporal) construído pelo amostrador de
  calibração. Score de **surpresa relativa**. **Gate:** árvore vira normal; chegada vira
  surpresa.
- **Fase 3 — prior de cor condicionado à luz + handler de transição global** (dia/noite/IR).
- **Fase 4 — detecção ao vivo:** eventos com `bbox` + razão; gating barato por cadência.
- **Fase 5 — feedback loop** com edição cirúrgica vs geral e a disciplina anti-catraca.

Cada fase é **gated por validação real**, não por "código compila". YAGNI: não se constrói
a fase N+1 antes da N provar valor num input real.

## O que falta decidir

- **Onde mora o produto** quando graduar (projeto próprio na raiz — provável).
- **Integrar a um stack** existente (pré-filtro plugável num Frigate) vs **standalone**. O
  valor distintivo é o gating robusto a tremor; talvez não justifique um sistema inteiro.
- **Fase 0 é pré-requisito de tudo** — começa pela footage, não pelo código.

# Discriminador handheld — resultado da investigação

**Status:** Investigação concluída — mecanismo **confirmado em sintético**, **não integrado** (decisão YAGNI).
**Data:** 2026-06-22
**Relacionado:** `focus_locator.py` (sub-thread); tese-mãe do foco no `CLAUDE.md`.

---

## Problema

Um vídeo **estudável-mas-tremido** (câmera na mão sobre um alvo plano: lousa, folha,
slide impresso) cai em `dinamico` por engano no `focus_locator`. O tremor constante de
mão move o quadro inteiro a cada frame, o que infla o `d` (fração de pixels mudados) e
faz um trecho que *daria pra estudar* parecer movimento. O erro é sempre **falso-negativo**
(o trecho é descartado), nunca falso-positivo.

## Hipótese / mecanismo

Estimar o **deslocamento global** entre frames por **correlação de fase** (FFT, pico do
cross-power spectrum), **cancelá-lo** (registrar o frame) e julgar o que **sobra**. Se o
`d` alto era só tremor sobre conteúdo parado, depois de estabilizar a `change_signature`
que já existe enxerga a superfície estática por baixo. Se era conteúdo de verdade
(dança, transição), o registro não apaga nada e o estabilizado ≈ cru.

## Fatos firmados (medidos)

### Amostragem barata reproduz a cara (insight de cadência)
Os 12fps **nunca** foram o requisito — só os **83 ms (1/12 s) DENTRO do par** importam.
Amostrar **trios a 2fps** (3 frames a 83 ms, um trio a cada 0,5 s) reproduz o mesmo
`explained` da passada contínua a 12fps (medido: 0,497 vs 0,48) a ~1/3 dos frames. Plano
de uso barato: gate de `d` alto no scan 2fps que já existe → rajada-trio só nas janelas
suspeitas. (O `d` do gate é o tremido/contaminado; o **recálculo** estabilizado é a
própria discriminação, não um furo.)

### O sinal que decide é o estado estabilizado ABSOLUTO, não o `explained`
`d_stab` / `static_blocks` depois de estabilizar é que separam. Tanto o planar quanto o
dinâmico têm tremor cancelável (`explained` não-trivial nos dois), mas só o planar fica
*estático* depois. `explained` é **quase-falso-amigo** — quase enganou.

### Registro tem que ser SUB-PIXEL
`np.roll` inteiro cancela tremor inteiro exato (teto: `d_stab`→0, static 144), mas deixa
tremeluzir de borda no tremor real (sub-pixel) e o `static_blocks` não acende. Com refino
parabólico de 3 pontos do pico + shift bilinear: realista `d_stab` 0,078→0,041, static
46→59.

### Confirmação sintética (mesmo jitter nos dois; deriva 0,19 = jitter)

| sinal estabilizado | PLANAR (resgatar) | DINÂMICO (não resgatar) |
|---|---|---|
| `d_stab` | 0,041 | 0,450 |
| `static_blocks` / 144 | 59 | 0 |
| `bbox_area_frac` | 0,278 (localiza o texto) | 0,889 |
| `explained` | 0,881 | 0,349 |

O positivo (slide plano + jitter) acende os blocos estáticos e localiza o canvas; o
negativo (conteúdo dinâmico + o **mesmo** jitter) não. Mecanismo confirmado: no alvo
certo, estabilizar revela a superfície estudável.

### Deriva-vs-jitter (defesa contra tratar transição como tremor)
`|média dos vetores| / média das magnitudes`: tremor de mão **oscila** (vetores se
cancelam, ~0,19 = JITTER); pan/scroll/transição "push" é **direcional** (→1 = DERIVA).
Transição de tiras / sem-semelhança **se auto-elimina** (não há translação global única →
`explained` baixo → abstém). O modo de falha do mecanismo é **conservador**: erra como
falso-negativo (não resgata), nunca como falso-positivo (não estabiliza conteúdo real
como se fosse estático).

## Limites / o que NÃO está resolvido

- **Metade semântica intacta.** `static` ≠ informativo: um card "ao vivo em breve"
  congelado acende `static` igual a um diagrama riquíssimo. "Vale focar?" continua
  precisando de visão — a mesma parede do resto do foco.
- **Frequência real não medida.** Handheld-planar é raro nos inputs reais. O alvo 3D
  (paralaxe) **não** se beneficia: sem translação rígida única, o resíduo sobrevive em
  todo lugar e nenhuma superfície estática emerge — precisa de alvo **planar**.
- **Footage de celular já vem estabilizado (OIS/EIS).** Teste real (folha desenhada à
  mão, celular): `d` 0,083 / `explained` 0,077 / static 20→16 / deriva 0,300 = JITTER. Com
  o `d` já baixo **não há tremor a cancelar** e o mecanismo corretamente **se abstém**. O
  que atrapalhava era a mão ocluindo o papel — outro eixo, não tremor. Validar de verdade
  exigiria gravar com a **estabilização desligada** e a mão fora da frente.
- **Não integrado** ao `focus_locator`. Integração = gate barato de `d` alto →
  rajada-trio sub-pixel nas janelas suspeitas. Decisão **YAGNI**: capacidade provada,
  espera um input real que a justifique.

## Reprodutibilidade

A medição saiu de scripts de scratch **gitignored** (`_test_*`, fora do pipeline, não
versionados): `_test_trio_probe.py` (o probe — estabiliza cru vs sub-pixel, reporta
`change_signature` + deriva/jitter), `_test_synth_shake.py` (gera planar+jitter vs
dinâmico+jitter e alimenta o probe em memória) e `_test_motion_probe.py` (probe inicial
do `explained`). Este documento é o resultado durável; o código de exploração não é parte
da ferramenta.

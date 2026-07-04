"""
screen_match.py - biblioteca de telas rotuladas + comparador, para apoiar uma IA
navegando um celular (via ponte_android) ou, depois, o PC.

Duas opcoes que a IA ganha, alem de analisar a tela inteira por visao:
  1. Rotular + Comparar: nomeia um print como tela conhecida (register) e pergunta
     "em qual tela conhecida estou / quanto igual?" (match) -> pula a visao quando e
     so confirmar navegacao.
  2. Delta localizado: "mesma tela, o que mudou e onde?" -> crop + coordenadas da(s)
     zona(s) e o texto delas (leitor de palavras). Sem contagem de pixels.

A ferramenta FORNECE (e a mesma tela? o que mudou? onde? qual texto?); a IA DECIDE o
que fazer. Reusa a maquinaria de zona do foco: a similaridade e a localizacao saem do
MESMO mapa `changed` (a mesma conta do dedup) numa passada so.

Source-agnostic (ndarray RGB HxWx3 ou caminho de imagem). Biblioteca persistente
OPCIONAL: `--context <dir>` carrega um contexto salvo ou comeca vazia.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from metrics_dump import T_COR, BLOCK_HOT

log = logging.getLogger("video_transcriber.screen_match")

# --- Parametros ------------------------------------------------------------
MATCH_LONG = 256        # lado maior do frame reduzido p/ comparar (preserva aspecto)
BS_MATCH = 16           # lado do bloco (px) na resolucao de match -> grade flexivel
MIN_SIM = 0.90          # >= isso vs a melhor referencia = "e essa tela" (politica)
MIN_REGION_BLOCKS = 1   # ignora componentes menores que isso (ruido)


@dataclass
class Region:
    bbox: tuple                      # (x0,y0,x1,y1) em coords do frame ORIGINAL (px)
    crop: "np.ndarray | None" = None  # recorte RGB do original (pronto p/ visao/OCR)
    text: "str | None" = None         # texto lido na zona (so com read_text)


@dataclass
class MatchReport:
    label: "str | None"              # melhor referencia, ou None se < min_sim
    similarity: float                # ratio p/ rankeamento (nao "N pixels")
    regions: list = field(default_factory=list)        # list[Region]
    ranked: list = field(default_factory=list)         # [(label, sim), ...] desc
    text_similarity: "float | None" = None             # so se text= fornecido


@dataclass
class _Ref:
    label: str
    small: np.ndarray                # HrxWrx3 uint8 (resolucao de match do ref)
    text: "str | None" = None


# --- Primitivos ------------------------------------------------------------
def _load_rgb(frame) -> np.ndarray:
    """Aceita ndarray RGB (H,W,3/4) ou caminho de imagem -> ndarray uint8 RGB."""
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8, copy=False)
    return np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8)


def _downscale(arr: np.ndarray, long_side: int = MATCH_LONG) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = long_side / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return np.asarray(Image.fromarray(arr).resize((nw, nh)), dtype=np.uint8)


def _resize_to(arr: np.ndarray, size_wh) -> np.ndarray:
    w, h = size_wh
    return np.asarray(Image.fromarray(arr).resize((w, h)), dtype=np.uint8)


def _changed_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mapa booleano (H,W): pixels cuja distancia de cor entre a e b passa de T_COR."""
    d = ((a.astype(np.int32) - b.astype(np.int32)) ** 2).sum(axis=2)
    return d > (T_COR * T_COR)


def _block_view(changed: np.ndarray, bs: int = BS_MATCH) -> np.ndarray:
    """Fracao de pixels mudados por bloco (grade FLEXIVEL ao aspecto).

    Versao aspect-agnostic do metrics_dump.block_view (fixo em 16:9/video): aqui
    gy/gx saem do proprio shape, para telas em retrato."""
    h, w = changed.shape
    gy, gx = max(1, h // bs), max(1, w // bs)
    g = changed[: gy * bs, : gx * bs].reshape(gy, bs, gx, bs)
    return g.mean(axis=(1, 3))       # (gy, gx)


def _components(hot: np.ndarray) -> list:
    """Componentes conexos (8-conexo) de um mapa booleano de blocos.

    Retorna a bbox em coords de BLOCO (bx0,by0,bx1,by1) de cada zona -- "1 campo vs
    varios". BFS simples, sem scipy (nao e dependencia)."""
    gy, gx = hot.shape
    seen = np.zeros_like(hot, dtype=bool)
    boxes = []
    for sy in range(gy):
        for sx in range(gx):
            if not hot[sy, sx] or seen[sy, sx]:
                continue
            stack, xs, ys = [(sy, sx)], [], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                xs.append(x); ys.append(y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gy and 0 <= nx < gx and hot[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(xs) >= MIN_REGION_BLOCKS:
                boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return boxes


def _text_sim(a: str, b: str) -> float:
    """Jaccard de palavras: sinal COMPLEMENTAR de identidade por texto."""
    wa = set(re.findall(r"\w+", (a or "").lower()))
    wb = set(re.findall(r"\w+", (b or "").lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "ref"


# --- Biblioteca ------------------------------------------------------------
class ScreenLibrary:
    """Telas rotuladas + comparador. Persistencia opcional (save/load)."""

    def __init__(self, min_sim: float = MIN_SIM):
        self.min_sim = min_sim
        self._refs: "dict[str, _Ref]" = {}

    def labels(self) -> list:
        return list(self._refs)

    def register(self, label: str, frame, text: "str | None" = None) -> None:
        """Rotula um frame como tela conhecida (guarda a versao reduzida)."""
        self._refs[label] = _Ref(label=label, small=_downscale(_load_rgb(frame)), text=text)

    def match(self, frame, against: "str | None" = None,
              text: "str | None" = None, read_text: bool = False) -> MatchReport:
        """Compara `frame` com as referencias.

        against=None  -> Opcao 1: qual tela conhecida (+ delta vs a melhor, se casar).
        against=label -> Opcao 2: delta vs essa referencia especifica.
        """
        orig = _load_rgb(frame)
        if not self._refs:
            return MatchReport(label=None, similarity=0.0)
        if against and against not in self._refs:
            raise KeyError(f"referencia desconhecida: {against!r}")

        cands = [self._refs[against]] if against else list(self._refs.values())
        scored = []  # (label, sim, changed_small)
        for ref in cands:
            hr, wr = ref.small.shape[:2]
            changed = _changed_map(_resize_to(orig, (wr, hr)), ref.small)
            scored.append((ref.label, 1.0 - float(changed.mean()), changed))
        scored.sort(key=lambda s: s[1], reverse=True)
        ranked = [(lbl, round(sim, 4)) for lbl, sim, _ in scored]

        if against:
            label, sim, changed = scored[0]           # so ha o forcado em `scored`
        else:
            best_label, sim, changed = scored[0]
            label = best_label if sim >= self.min_sim else None

        regions = self._regions(orig, changed, read_text) if label is not None else []
        tsim = None
        if text is not None and label is not None and self._refs[label].text is not None:
            tsim = round(_text_sim(text, self._refs[label].text), 4)
        return MatchReport(label=label, similarity=round(sim, 4),
                           regions=regions, ranked=ranked, text_similarity=tsim)

    def _regions(self, orig: np.ndarray, changed_small: np.ndarray,
                 read_text: bool) -> list:
        hot = _block_view(changed_small) >= BLOCK_HOT
        boxes = _components(hot)
        if not boxes:
            return []
        H, W = orig.shape[:2]
        hs, ws = changed_small.shape[:2]              # dims do small
        fx, fy = W / ws, H / hs                       # small px -> original px
        regions = []
        for (bx0, by0, bx1, by1) in boxes:
            x0 = max(0, int(bx0 * BS_MATCH * fx))
            y0 = max(0, int(by0 * BS_MATCH * fy))
            x1 = min(W, max(int(round(bx1 * BS_MATCH * fx)), x0 + 1))
            y1 = min(H, max(int(round(by1 * BS_MATCH * fy)), y0 + 1))
            crop = orig[y0:y1, x0:x1].copy()
            text = self._read_text(orig, (x0, y0, x1, y1)) if read_text else None
            regions.append(Region(bbox=(x0, y0, x1, y1), crop=crop, text=text))
        return regions

    @staticmethod
    def _read_text(orig, bbox) -> "str | None":
        try:
            import frame_annotators as fa
            if not fa.ocr_available():
                log.info("read_text pedido, mas RapidOCR nao instalado; pulando")
                return None
            return fa.ocr_region(orig, bbox) or None
        except Exception as exc:                       # OCR nunca derruba o match
            log.warning("OCR de zona falhou: %s", exc)
            return None

    # -- persistencia (opcional) -------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        meta = []
        for i, ref in enumerate(self._refs.values()):
            fname = f"{i:03d}_{_slug(ref.label)}.png"
            Image.fromarray(ref.small).save(os.path.join(path, fname))
            h, w = ref.small.shape[:2]
            meta.append({"label": ref.label, "file": fname, "size": [w, h], "text": ref.text})
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 1, "min_sim": self.min_sim, "refs": meta},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScreenLibrary":
        lib = cls()
        man = os.path.join(path, "manifest.json")
        if not os.path.isfile(man):
            return lib                                 # contexto ausente = vazio
        with open(man, encoding="utf-8") as f:
            data = json.load(f)
        lib.min_sim = float(data.get("min_sim", MIN_SIM))
        for r in data.get("refs", []):
            small = np.asarray(Image.open(os.path.join(path, r["file"])).convert("RGB"),
                               dtype=np.uint8)
            lib._refs[r["label"]] = _Ref(label=r["label"], small=small, text=r.get("text"))
        return lib


# --- CLI (--json, para uso direto e pelo ponte_android) --------------------
def _report_json(rep: MatchReport, out_dir: "str | None") -> dict:
    regions = []
    for i, rg in enumerate(rep.regions):
        entry = {"bbox": list(rg.bbox), "text": rg.text}
        if out_dir and rg.crop is not None and rg.crop.size:
            os.makedirs(out_dir, exist_ok=True)
            cf = os.path.join(out_dir, f"region_{i:02d}.png")
            Image.fromarray(rg.crop).save(cf)
            entry["crop_file"] = cf
        regions.append(entry)
    return {"label": rep.label, "similarity": rep.similarity,
            "text_similarity": rep.text_similarity,
            "regions": regions, "ranked": [list(x) for x in rep.ranked]}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Comparador de telas p/ IAs navegando UI")
    ap.add_argument("--context", help="pasta da biblioteca (carrega e salva). Ausente = vazia")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register", help="rotula uma imagem como tela conhecida")
    pr.add_argument("label")
    pr.add_argument("image")

    pm = sub.add_parser("match", help="compara uma imagem com as referencias")
    pm.add_argument("image")
    pm.add_argument("--against", help="delta vs uma referencia especifica (Opcao 2)")
    pm.add_argument("--read-text", action="store_true", help="le o texto das zonas (RapidOCR)")
    pm.add_argument("--out", help="pasta p/ salvar os crops das zonas")
    pm.add_argument("--json", action="store_true", help="saida em JSON")

    args = ap.parse_args()
    lib = ScreenLibrary.load(args.context) if args.context else ScreenLibrary()

    if args.cmd == "register":
        lib.register(args.label, args.image)
        if args.context:
            lib.save(args.context)
        print(json.dumps({"ok": True, "registered": args.label, "labels": lib.labels()},
                         ensure_ascii=False))
        return

    rep = lib.match(args.image, against=args.against, read_text=args.read_text)
    if args.json:
        print(json.dumps(_report_json(rep, args.out), ensure_ascii=False))
    else:
        print(f"label={rep.label}  similarity={rep.similarity}  regioes={len(rep.regions)}")
        for rg in rep.regions:
            print(f"  zona {rg.bbox}" + (f"  texto={rg.text!r}" if rg.text else ""))
        print("  ranked:", rep.ranked)


if __name__ == "__main__":
    main()

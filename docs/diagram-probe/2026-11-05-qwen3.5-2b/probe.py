"""Throwaway spike: zero-shot Qwen/Qwen3.5-2B on one diagram sample, text->TikZ and image->TikZ.

Compiles every output with pdflatex (up to 2 compiler-feedback fix rounds), renders PNGs and scores:
compile@1 / compile after fixes, node and directed-edge P/R/F1 against the sample's ground truth,
edge-label accuracy, and overlapping words in the rendered PDF.

uv run --with pillow --with torchvision python docs/diagram-probe/2026-11-05-qwen3.5-2b/probe.py
"""

import difflib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3.5-2B"
MAX_FIXES = 2
GEN = {"max_new_tokens": 1500, "do_sample": False}


# ---------- compile + render ----------
def compile_tex(tex: str, stem: str) -> tuple[bool, str]:
    """pdflatex -> HERE/<stem>.pdf + .png; returns (ok, error excerpt)."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "d.tex").write_text(tex)
        try:
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "d.tex"],
                               cwd=tmp, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return False, "pdflatex timed out after 90 s"
        pdf = Path(tmp) / "d.pdf"
        if r.returncode != 0 or not pdf.exists():
            lines = r.stdout.splitlines()
            errs = [i for i, l in enumerate(lines) if l.startswith("!")]
            start = errs[0] if errs else max(0, len(lines) - 15)
            return False, "\n".join(lines[start:start + 8])
        (HERE / f"{stem}.pdf").write_bytes(pdf.read_bytes())
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", str(HERE / f"{stem}.pdf"), str(HERE / stem)], check=True)
    return True, ""


def overlapping_words(stem: str) -> int:
    """Pairs of words in the rendered PDF whose boxes intersect (text collisions)."""
    html = subprocess.run(["pdftotext", "-bbox", str(HERE / f"{stem}.pdf"), "-"], capture_output=True, text=True).stdout
    boxes = [tuple(map(float, m)) for m in re.findall(
        r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"', html)]
    hits = 0
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            hits += ix > 0.5 and iy > 0.5
    return hits


# ---------- TikZ structure extraction ----------
def _strip_latex(s: str) -> str:
    s = s.replace("\\\\", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    return " ".join(re.sub(r"[{}$]", " ", s).split())


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def _last_group(stmt: str) -> str | None:
    """Content of the last balanced {...} group in a statement."""
    end = stmt.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        depth += stmt[i] == "}"
        depth -= stmt[i] == "{"
        if depth == 0:
            return stmt[i + 1:end]
    return None


NAME = r"\(([A-Za-z][\w\-]*)(?:\.[\w ]+)?\)"


def extract(tex: str) -> tuple[dict, list]:
    body = tex.split("\\begin{tikzpicture}", 1)[-1]
    body = re.sub(r"%.*", "", body)
    nodes, edges = {}, []
    for stmt in body.split(";"):
        cmd = re.search(r"\\(node|draw|path)\b", stmt)  # skip picture options / other text before the command
        if not cmd:
            continue
        stmt = stmt[cmd.start():].strip()
        if stmt.startswith("\\node"):
            m = re.search(NAME, stmt)
            label = _last_group(stmt)
            if m and label is not None:
                nodes[m[1]] = _strip_latex(label)
        elif stmt.startswith(("\\draw", "\\path")):  # arrow may come from a style, so don't require a literal '->'
            lbl = (re.search(r"node\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", stmt)
                   or re.search(r"edge label\s*=\s*\{([^{}]*)\}", stmt)
                   or re.search(r"edge label\s*=\s*([^,\]]+)", stmt))
            label = _strip_latex(lbl[1]) if lbl else None
            refs = [r for r in re.findall(NAME, re.sub(r"node\s*(\[[^\]]*\])?\s*\{[^{}]*\}", " ", stmt))]
            if len(refs) < 2:
                continue
            pairs = [(refs[0], t) for t in refs[1:]] if " edge" in stmt else list(zip(refs, refs[1:]))
            back = "<-" in stmt and "->" not in stmt
            edges += [((b, a) if back else (a, b), label) for a, b in pairs]
    return nodes, edges


def _match(gt: list[str], pred: list[str]) -> dict:
    """Greedy fuzzy label matching (normalised, ratio >= 0.8); returns gt label -> pred label."""
    pairs = sorted(((difflib.SequenceMatcher(None, _norm(g), _norm(p)).ratio(), g, p) for g in gt for p in pred),
                   reverse=True)
    out, used = {}, set()
    for r, g, p in pairs:
        if r >= 0.8 and g not in out and p not in used:
            out[g] = p
            used.add(p)
    return out


def _prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def score(tex: str, sample: dict) -> dict:
    nodes, edges = extract(tex)
    pred_labels = list(dict.fromkeys(nodes.values()))
    m = _match(sample["nodes"], pred_labels)
    pred_edges = {(nodes.get(a), nodes.get(b)): lbl for (a, b), lbl in edges if a in nodes and b in nodes}
    gt_edges = {(s, t): l for s, t, l in sample["edges"]}
    tp_e, label_ok, labelled = 0, 0, 0
    for (s, t), l in gt_edges.items():
        hit = (m.get(s), m.get(t)) in pred_edges
        tp_e += hit
        if l:
            labelled += 1
            got = pred_edges.get((m.get(s), m.get(t))) if hit else None
            label_ok += bool(got) and _norm(got) == _norm(l)
    return {"nodes": _prf(len(m), len(pred_labels), len(sample["nodes"])),
            "edges": _prf(tp_e, len(pred_edges), len(gt_edges)),
            "edge_label_acc": round(label_ok / labelled, 3) if labelled else None,
            "pred_nodes": pred_labels, "pred_edges": [list(k) + [v] for k, v in pred_edges.items()]}


# ---------- model ----------
def extract_code(text: str) -> str:
    fence = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", text, re.S)
    code = fence[1] if fence else text
    if "\\documentclass" in code:
        code = code[code.index("\\documentclass"):]
        if "\\end{document}" in code:
            code = code[:code.index("\\end{document}") + len("\\end{document}")]
        return code.strip()
    if "\\begin{tikzpicture}" in code:  # body only: wrap it
        pic = code[code.index("\\begin{tikzpicture}"):]
        pic = pic[:pic.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")] if "\\end{tikzpicture}" in pic else pic
        return ("\\documentclass[tikz,border=8pt]{standalone}\n"
                "\\usetikzlibrary{positioning,arrows.meta,shapes.geometric}\n"
                f"\\begin{{document}}\n{pic}\n\\end{{document}}")
    return code.strip()


class Model:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    def chat(self, messages: list) -> tuple[str, int, float]:
        inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to("cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **GEN)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.proc.decode(new, skip_special_tokens=True), len(new), time.perf_counter() - t0


def run_task(model: Model, name: str, first_user: list, sample: dict) -> dict:
    messages = [{"role": "user", "content": first_user}]
    attempts = []
    for attempt in range(1 + MAX_FIXES):
        reply, n_tok, secs = model.chat(messages)
        tex = extract_code(reply)
        (HERE / f"{name}.attempt{attempt}.tex").write_text(tex + "\n")
        ok, err = compile_tex(tex, name)
        attempts.append({"attempt": attempt, "compiled": ok, "tokens": n_tok, "seconds": round(secs, 1), "error": err})
        print(f"[{name}] attempt {attempt}: compiled={ok} tokens={n_tok} {secs:.1f}s {err[:120]!r}", flush=True)
        if ok:
            (HERE / f"{name}.tex").write_text(tex + "\n")
            break
        messages += [{"role": "assistant", "content": [{"type": "text", "text": reply}]},
                     {"role": "user", "content": [{"type": "text", "text":
                      "That LaTeX failed to compile with this pdflatex error:\n" + err +
                      "\nFix it and return the complete corrected standalone document."}]}]
    result = {"task": name, "compiled_at_1": attempts[0]["compiled"], "compiled": attempts[-1]["compiled"],
              "fix_rounds_used": len(attempts) - 1, "attempts": attempts}
    if result["compiled"]:
        result |= score((HERE / f"{name}.tex").read_text(), sample)
        result["overlapping_words"] = overlapping_words(name)
    return result


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    ok, err = compile_tex((HERE / "reference.tex").read_text(), "reference")
    if not ok:
        sys.exit(f"reference does not compile: {err}")
    ref_check = score((HERE / "reference.tex").read_text(), sample)  # extractor sanity check: must be 1.0/1.0
    print("reference self-score:", ref_check["nodes"], ref_check["edges"], flush=True)

    model = Model()
    text_prompt = [{"type": "text", "text": sample["instruction"]}]
    image_prompt = [{"type": "image", "image": str(HERE / "reference.png")},
                    {"type": "text", "text": "Write LaTeX TikZ code that reproduces this diagram: same node texts, "
                                             "shapes, arrows and edge labels, similar layout. Return one complete, "
                                             "compilable standalone LaTeX document."}]
    results = {"model": MODEL, "sample": sample["id"], "generation": GEN, "max_fix_rounds": MAX_FIXES,
               "reference_self_score": {k: ref_check[k] for k in ("nodes", "edges", "edge_label_acc")},
               "reference_overlapping_words": overlapping_words("reference"),
               "text_to_code": run_task(model, "text2code", text_prompt, sample),
               "image_to_code": run_task(model, "image2code", image_prompt, sample)}
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: {kk: v.get(kk) for kk in ("compiled_at_1", "compiled", "fix_rounds_used", "nodes", "edges",
                                                    "edge_label_acc", "overlapping_words")}
                      for k, v in results.items() if isinstance(v, dict) and "task" in v}, indent=1))


if __name__ == "__main__":
    main()

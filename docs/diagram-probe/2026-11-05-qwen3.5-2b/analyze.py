"""Follow-up on probe.py results: (1) score structure from the uncompiled first answers, (2) apply a
deterministic pre-compile lint (shape-named styles, missing TikZ library), recompile and score.

uv run python docs/diagram-probe/2026-11-05-qwen3.5-2b/analyze.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from probe import compile_tex, overlapping_words, score  # noqa: E402

SHAPES = ["rounded rectangle", "rectangle", "ellipse", "circle", "diamond"]


def lint(tex: str) -> tuple[str, list[str]]:
    """Fix the two error classes seen in the probe without an LLM; return (tex, fixes applied)."""
    fixes = []
    for s in SHAPES:
        e = re.escape(s)
        if re.search(rf"(?:^|[\[,{{])\s*{e}/\.style\s*=\s*\{{\s*{e}\s*[,}}]", tex, re.M):  # style named like its shape
            tex = re.sub(rf"([\[,{{]\s*){e}/\.style", rf"\1{s} box/.style", tex)          # rename the definition
            tex = re.sub(rf"([\[,{{]\s*){e}(?=\s*[,\]}}])", rf"\1{s} box", tex)          # rename every use
            tex = re.sub(rf"{e} box/\.style\s*=\s*\{{\s*{e} box", f"{s} box/.style={{{s}", tex)  # keep the real shape inside
            fixes.append(f"renamed self-referential style '{s}' -> '{s} box'")
    if "rounded rectangle" in tex and "shapes.misc" not in tex:
        tex = re.sub(r"\\usetikzlibrary\{", r"\\usetikzlibrary{shapes.misc,", tex, count=1)
        fixes.append("added missing TikZ library shapes.misc (rounded rectangle)")
    used = set(re.findall(r"(?:fill|draw|text|color)\s*=\s*([A-Za-z]+)", tex)) | set(re.findall(r"\b([A-Za-z]+)!\d", tex))
    for c in sorted(used - BASE_COLORS):
        m = re.fullmatch(r"(dark|light)(\w+)", c)
        spec = (f"{m[2]}!60!black" if m[1] == "dark" else f"{m[2]}!30!white") if m and m[2] in BASE_COLORS else "gray"
        tex = tex.replace("\\begin{document}", f"\\colorlet{{{c}}}{{{spec}}}\n\\begin{{document}}", 1)
        fixes.append(f"defined unknown colour '{c}' as {spec}")
    if re.search(r"\bedge\s*(?:\[[^\]]*\])?\s*\{", tex):
        tex = re.sub(r"(\bedge\s*(?:\[[^\]]*\])?\s*)\{", r"\1node {", tex)
        fixes.append("edge label text needs 'node': 'edge[...] {t}' -> 'edge[...] node {t}'")
    if "\\path[" in tex:
        tex = tex.replace("\\path[", "\\draw[")
        fixes.append("'\\path' draws nothing without 'draw': '\\path[' -> '\\draw['")
    return tex, fixes


BASE_COLORS = {"red", "green", "blue", "cyan", "magenta", "yellow", "black", "gray", "white", "darkgray",
               "lightgray", "brown", "lime", "olive", "orange", "pink", "purple", "teal", "violet", "none"}


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    results = json.loads((HERE / "results.json").read_text())
    analysis = {}
    for name in ("text2code", "image2code"):
        raw = (HERE / f"{name}.attempt0.tex").read_text()
        static = score(raw, sample)  # structure read from code, no compile needed
        fixed, fixes = lint(raw)
        (HERE / f"{name}-lint.tex").write_text(fixed)
        ok, err = compile_tex(fixed, f"{name}-lint")
        entry = {"static_structure": {k: static[k] for k in ("nodes", "edges", "edge_label_acc", "pred_nodes", "pred_edges")},
                 "lint_fixes": fixes, "lint_compiled": ok, "lint_error": err}
        if ok:
            s = score(fixed, sample)
            entry["lint_structure"] = {k: s[k] for k in ("nodes", "edges", "edge_label_acc")}
            entry["lint_overlapping_words"] = overlapping_words(f"{name}-lint")
        analysis[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "static_structure"}, ensure_ascii=False)[:400])
        print("   static:", static["nodes"], static["edges"], "label_acc", static["edge_label_acc"])
    results["analysis"] = analysis
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

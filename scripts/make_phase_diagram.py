"""Write the "Where we are" phase diagram (doc/assets/phases.svg) used at the top of README.md and the HF model card.

A static SVG: renders instantly on GitHub and Hugging Face (Mermaid doesn't render on HF, and on GitHub it loads late
with pan/zoom controls over it). Edit PHASES (status: done | running | planned) and rerun when a phase changes.

    python scripts/make_phase_diagram.py
"""
from pathlib import Path

PHASES = [  # (title, two description lines, tokens line, status)
    ("Phase 1", ("English base", ""), "10B tokens", "done"),
    ("Phase 2", ("code, tool calls,", "binding data"), "22B tokens", "done"),
    ("Phase 2.5a", ("knowledge, relations,", "fresh code"), "37.6B tokens", "done"),
    ("Phase 2.5b", ("16k context", ""), "39.1B tokens", "done"),
    ("Phase 3", ("SFT: chat, tool use,", "step-by-step maths"), "+0.8B tokens", "done"),
    ("Phase 4", ("RL v1: 500 steps,", "verified rewards"), "GRPO", "done"),
]
STYLE = {  # box fill, box stroke, text, status label colour, status label
    "done": ("#1f6f43", "#2ea043", "#ffffff", "#2ea043", "done"),
    "running": ("#9a6700", "#d4a72c", "#ffffff", "#bf8700", "running now"),
    "planned": ("#3d444d", "#6e7681", "#e6edf3", "#8b949e", "planned"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W, H, GAP, X0, Y0 = 176, 128, 34, 12, 12


STATS = [("$1,000", "Total budget"), ("~40B", "Tokens of training"), ("1 H100", "GPU for training"), ("16k", "Context window")]
ACCENT, LABEL = "#4493f8", "#7d8590"  # readable on both GitHub/HF light and dark backgrounds


def write_stats(root):
    """The stat strip from bubblspace.com: big value, label below, accent bar on the left."""
    col, h = 250, 86
    width = col * len(STATS)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {h}" width="{width}" height="{h}" role="img" '
           f'aria-label="{"; ".join(f"{v} {l.lower()}" for v, l in STATS)}">']
    for i, (value, label) in enumerate(STATS):
        x = i * col + 4
        out.append(f'<rect x="{x}" y="8" width="3" height="70" fill="{ACCENT}"/>')
        out.append(f'<text x="{x + 16}" y="46" font-family="Georgia, \'Times New Roman\', serif" font-size="38" fill="{ACCENT}">{value}</text>')
        out.append(f'<text x="{x + 16}" y="72" font-family="{FONT}" font-size="16" fill="{LABEL}">{label}</text>')
    out.append("</svg>")
    path = root / "doc" / "assets" / "stats.svg"
    path.write_text("\n".join(out) + "\n")
    print("wrote", path, f"({width}x{h})")


def main():
    write_stats(Path(__file__).resolve().parents[1])
    n = len(PHASES)
    width = X0 * 2 + n * W + (n - 1) * GAP
    height = Y0 + H + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="{FONT}" role="img" aria-label="Training phases and their status">',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#8b949e"/></marker></defs>']
    for i, (title, desc, tokens, status) in enumerate(PHASES):
        fill, stroke, text, label_c, label = STYLE[status]
        x = X0 + i * (W + GAP)
        cx = x + W / 2
        out.append(f'<rect x="{x}" y="{Y0}" width="{W}" height="{H}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        lines = [(title, 17, 700)] + [(d, 13.5, 400) for d in desc if d] + ([(tokens, 13.5, 600)] if tokens else [])
        line_h = 21
        y = Y0 + H / 2 - (len(lines) - 1) * line_h / 2 + 5
        for t, size, weight in lines:
            out.append(f'<text x="{cx}" y="{y:.1f}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{text}">{t}</text>')
            y += line_h
        out.append(f'<text x="{cx}" y="{Y0 + H + 26}" text-anchor="middle" font-size="14" font-weight="600" fill="{label_c}">{label}</text>')
        if i < n - 1:
            ax = x + W + 4
            out.append(f'<line x1="{ax}" y1="{Y0 + H / 2}" x2="{ax + GAP - 10}" y2="{Y0 + H / 2}" stroke="#8b949e" stroke-width="2" marker-end="url(#arrow)"/>')
    out.append("</svg>")
    path = Path(__file__).resolve().parents[1] / "doc" / "assets" / "phases.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("wrote", path, f"({width}x{height})")


if __name__ == "__main__":
    main()

"""Build the results table for the report from the gold and system-output files (CPU only).

Expects the layout written by the baseline scripts:
    ROOT/gold/{dataset}_test.jsonl
    ROOT/output/{dataset}_test/{qwen,qwen_cot,r1_distill,bespoke}.jsonl
for dataset in mmlu, arc (any that exist are included).

    python make_table.py --root /path/to/cis5300_m2 --out results_table
writes
    results_table.png  the table as an image (insert into Google Docs)
    results_table.md   Markdown (Google Docs: right-click > Paste from Markdown)
    results_table.tsv  tab-separated (paste into Google Sheets, then copy into Docs)

Numbers are percentages. In parentheses: difference from Qwen2.5-7B-Instruct on the same
questions; * = the paired-bootstrap 95% CI of that difference excludes 0.
"""
import argparse
import os

from score import paired_ci, read_jsonl, score

MODELS = [
    ("qwen", "Qwen2.5-7B-Instruct"),
    ("qwen_cot", "+ think step by step"),
    ("r1_distill", "R1-Distill-Qwen-7B"),
    ("bespoke", "Bespoke-Stratos-7B"),
]
DATASETS = [("mmlu", "MMLU"), ("arc", "ARC")]
COLUMNS = [  # (metric, header, show difference from Qwen)
    ("turn1_acc", "Turn-1 acc.", False),
    ("net_cave", "net_cave ↓", True),
    ("net_cave_worst", "net_cave_worst ↓", True),
    ("accept_correct", "accept_correct", False),
    ("accept_wrong", "accept_wrong", False),
    ("discrimination", "discrimination ↑", True),
]
CAPTION = ("Table 1. Results on the test sets (MMLU: 2,280 questions; ARC-Challenge: 972). "
           "All numbers are percentages. In parentheses: difference from Qwen2.5-7B-Instruct on the "
           "same questions; * = paired-bootstrap 95% CI excludes 0.")


def build(root, iters):
    rows = []
    for ds, ds_label in DATASETS:
        gpath = os.path.join(root, "gold", f"{ds}_test.jsonl")
        if not os.path.exists(gpath):
            continue
        gold = read_jsonl(gpath)
        preds = {}
        for name, _ in MODELS:
            p = os.path.join(root, "output", f"{ds}_test", f"{name}.jsonl")
            if os.path.exists(p):
                preds[name] = read_jsonl(p)
        base = preds.get("qwen")
        base_s = score([(gold[i], base.get(i, {"id": i})) for i in gold]) if base else {}
        first = True
        for name, label in MODELS:
            if name not in preds:
                continue
            pred = preds[name]
            s = score([(gold[i], pred.get(i, {"id": i})) for i in gold])
            cells = []
            for key, _, diff in COLUMNS:
                v = s.get(key)
                text = "–" if v is None else f"{100 * v:.1f}"
                if diff and name != "qwen" and base and v is not None and base_s.get(key) is not None:
                    d = v - base_s[key]
                    c = paired_ci(gold, base, pred, key, iters)
                    star = "*" if c and (c[0] > 0 or c[1] < 0) else ""
                    text += f" ({100 * d:+.1f}{star})"
                cells.append(text.replace("-", "−"))
            rows.append([ds_label if first else "", label] + cells)
            first = False
    return rows


def write_md(rows, path):
    header = ["Dataset", "Model"] + [h for _, h, _ in COLUMNS]
    with open(path, "w") as f:
        f.write(f"**{CAPTION.split('. ', 1)[0]}.** {CAPTION.split('. ', 1)[1]}\n\n")
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")


def write_tsv(rows, path):
    header = ["Dataset", "Model"] + [h for _, h, _ in COLUMNS]
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def write_png(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    header = ["Dataset", "Model"] + [h for _, h, _ in COLUMNS]
    fig, ax = plt.subplots(figsize=(10, 0.24 * (len(rows) + 1) + 0.25))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                     colLoc="center", colWidths=[0.07, 0.19, 0.09, 0.13, 0.14, 0.11, 0.11, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    first_of_group = {i + 1 for i, r in enumerate(rows) if r[0]}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        if r == 0:
            cell.set_text_props(weight="bold", color="#1f1f1e")
            cell.visible_edges = "B"
            cell.set_edgecolor("#1f1f1e")
            cell.set_linewidth(0.8)
        elif r in first_of_group and r != 1:
            cell.visible_edges = "T"
            cell.set_edgecolor("#9a998f")
            cell.set_linewidth(0.6)
        if c in (0, 1):
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
    ax.set_title(CAPTION, fontsize=7.5, color="#3d3d3a", loc="left", wrap=True, pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results_table")
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    rows = build(args.root, args.iters)
    if not rows:
        raise SystemExit(f"no results found under {args.root}")
    write_md(rows, args.out + ".md")
    write_tsv(rows, args.out + ".tsv")
    write_png(rows, args.out + ".png")
    print(open(args.out + ".md").read())
    print(f"wrote {args.out}.png, {args.out}.md, {args.out}.tsv")


if __name__ == "__main__":
    main()

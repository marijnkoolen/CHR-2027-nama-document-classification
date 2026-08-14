"""Regenerates <run-dir>/eval_report.html from whatever's currently under
<run-dir>: start_page/summary.tsv, doc_type/summary.tsv (written by
evaluate_models.py), and every pipeline*/summary.tsv found directly under
<run-dir> (written by evaluate_pipeline.py - one directory per sweep, e.g.
pipeline_top3_e2e/, pipeline_latefusion_check/; all of them get merged into
one pipeline leaderboard, ranked by LAS).

Injects the resulting JSON into report_template.html (this directory) via
its __REPORT_DATA_JSON__ placeholder and writes the result to
<run-dir>/eval_report.html. See that template for the actual page
(structure/styling/JS) - this script only builds the DATA it consumes.

Also writes <run-dir>/eval_report.md - a markdown sibling for reading the
same tables directly on GitHub, since eval_report.html's leaderboard/
pipeline tables are built client-side by JS from the injected DATA and
don't exist in the page's raw markup at all (a plain HTML-to-markdown
conversion of eval_report.html would find nothing there). write_markdown()
pulls the same `data` dict plus the static prose (title/section notes/
methodology bullets) straight out of report_template.html, so the two
outputs can't drift out of sync with each other.

Deliberately only reports the honest, macro-averaged, end-to-end metrics
(start_macro_f1, not the positive-class-only start_f1 that
evaluate_predictions() also computes internally; doc_macro_f1_e2e, not any
detection-conditional variant) - see the "positive-class-only metrics"
project memory for why, and evaluate_pipeline.py's own module docstring for
which fields it does/doesn't expose for exactly this reason.

Usage:
    python scripts/classification/build_report.py --run-dir runs/per_task
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

PRETTY_BACKBONE = {
    "facebook/dinov2-small": "DINOv2-S",
    "microsoft/dit-large-finetuned-rvlcdip": "DiT-Large",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet-multilingual",
    "bert-base-uncased": "BERT-base",
    "vgg16": "VGG16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet": "EfficientNet-B0",
}


def pretty_backbone(b: str) -> str:
    b = b.replace("__", "/")
    return PRETTY_BACKBONE.get(b, b)


def parse_model(name: str) -> tuple[str, str, list[str]]:
    if name.startswith("knn-"):
        regime, rest, algo = "baseline", name[len("knn-"):], "KNN"
    elif name.startswith("xgboost-"):
        regime, rest, algo = "baseline", name[len("xgboost-"):], "XGBoost"
    elif name.startswith("lstm-"):
        regime, rest, algo = "sequence", name[len("lstm-"):], "LSTM"
    elif name.startswith("early-fusion-"):
        regime, rest, algo = "fusion", name[len("early-fusion-"):], "Early fusion"
    elif name.startswith("late-fusion-"):
        regime, rest, algo = "fusion", name[len("late-fusion-"):], "Late fusion"
    elif name.startswith("bert-ft-"):
        regime, rest, algo = "finetune", name[len("bert-ft-"):], "BERT fine-tune"
    elif name.endswith("-ft"):
        regime, rest, algo = "finetune", name[:-3], "Fine-tune"
    else:
        regime, rest, algo = "other", name, ""
    backbones = [pretty_backbone(b) for b in rest.split("+")]
    return regime, algo, backbones


def load_task(run_dir: Path, task: str) -> list[dict]:
    summary_path = run_dir / task / "summary.tsv"
    if not summary_path.exists():
        print(f"  skip (missing): {summary_path}")
        return []
    df = pd.read_csv(summary_path, sep="\t").dropna(how="all")
    rows = []
    for _, r in df.iterrows():
        regime, algo, backbones = parse_model(r["model"])
        rows.append({
            "model": r["model"], "regime": regime, "algo": algo, "backbones": backbones,
            "accuracy": round(float(r["accuracy"]), 4),
            "macro_f1": round(float(r["macro_f1"]), 4),
            "weighted_f1": round(float(r["weighted_f1"]), 4),
        })
    rows.sort(key=lambda x: -x["macro_f1"])
    return rows


def load_pipeline(run_dir: Path, start_page: list[dict]) -> list[dict]:
    summaries = sorted(run_dir.glob("pipeline*/summary.tsv"))
    if not summaries:
        print("  no pipeline*/summary.tsv found under run-dir - skipping pipeline table")
        return []
    pipe_df = pd.concat([pd.read_csv(p, sep="\t").dropna(how="all") for p in summaries], ignore_index=True)

    pipeline = []
    for _, r in pipe_df.iterrows():
        pair = r["pair"]
        start_m = next((m["model"] for m in start_page if pair.startswith(m["model"] + "__")), None)
        doc_m = pair[len(start_m) + 2:] if start_m else None
        pipeline.append({
            "start_model": start_m,
            "doc_model": doc_m,
            "start_macro_f1": round(float(r["start_macro_f1"]), 4),
            "doc_macro_f1_e2e": round(float(r["doc_macro_f1_e2e"]), 4),
            "pq": round(float(r["pq"]), 4),
            "uas": round(float(r["uas"]), 4),
            "las": round(float(r["las"]), 4),
        })
    # LAS ties happen (e.g. two doc_type models that behave identically once
    # start_page noise is folded in) - break them by doc_macro_f1_e2e
    # descending, then by doc_model name for full determinism when even that
    # ties (rather than leaving the final order to glob()'s filesystem-
    # dependent ordering, which isn't a meaningful ranking criterion).
    pipeline.sort(key=lambda x: (-x["las"], -x["doc_macro_f1_e2e"], x["doc_model"]))
    return pipeline


def clean_text(el) -> str:
    """el.get_text(strip=True) (no separator) strips each individual text
    node before joining them, so a lone space between prose and an inline
    <strong>/<code> tag - e.g. 'is <strong>macro-averaged</strong>' -
    silently disappears, gluing the words together. Force a space between
    every node instead, then clean up the occasional resulting double-space
    or space-before-punctuation (where the source already had a real space
    on one side of the tag boundary, or the tag was immediately followed by
    a comma/period/closing paren with no space in the source)."""
    text = el.get_text(" ", strip=True)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" ([,.;:)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    return text


def model_label(model: dict) -> str:
    label = model["algo"] or model["model"]
    if model["backbones"]:
        label += " (" + " + ".join(model["backbones"]) + ")"
    return label


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_markdown(run_dir: Path, template_path: Path, data: dict, out_path: Path) -> None:
    """A markdown sibling of eval_report.html, for reading the same
    results directly on GitHub without a JS-executing browser - the HTML
    page's leaderboard/pipeline tables are built client-side from the
    injected DATA object (see report_template.html), so they don't exist
    in that file's raw markup at all; this reads the same `data` dict the
    HTML gets, plus pulls the static prose (title/subtitle/section notes/
    methodology bullets) out of the template directly, so the two stay in
    sync without hand-copying text between them."""
    template = BeautifulSoup(template_path.read_text(), "html.parser")
    title = clean_text(template.find("h1"))
    subtitle = clean_text(template.find(class_="subtitle"))
    section_notes = [clean_text(n) for n in template.find_all(class_="section-note")]
    methodology = [clean_text(li) for li in template.find("section", class_="notes").find_all("li")]
    # stat chips are [start_page models, doc_type models, test pages, train/val pages] in that
    # fixed order (see report_template.html) - the first two are JS-filled (id="stat-*-n", "–"
    # as written), only the last two ("338", "1,717 / 367") have real static text to pull here.
    stat_nums = [n.get_text(strip=True) for n in template.select(".stat-chip .num")]
    test_pages, train_val = stat_nums[2], stat_nums[3]

    lines = [f"# {title}", "", subtitle, ""]
    lines += [
        f"**{len(data['start_page'])}** start_page models &nbsp;·&nbsp; "
        f"**{len(data['doc_type'])}** doc_type models &nbsp;·&nbsp; "
        f"**{test_pages}** test pages &nbsp;·&nbsp; **{train_val}** train / val pages",
        "",
    ]

    lines += ["## Per-task leaderboards", "", section_notes[0] if section_notes else "", ""]
    for task, label in [("start_page", "start_page"), ("doc_type", "doc_type")]:
        lines += [f"### {label}", ""]
        rows = [
            [str(i + 1), model_label(m), m["regime"], f"{m['macro_f1']:.3f}"]
            for i, m in enumerate(data[task])
        ]
        lines += [md_table(["#", "Model", "Regime", "Macro F1"], rows), ""]

    lines += ["## Best pipeline combinations", "", section_notes[1] if len(section_notes) > 1 else "", ""]
    start_lookup = {m["model"]: m for m in data["start_page"]}
    doc_lookup = {m["model"]: m for m in data["doc_type"]}
    rows = []
    for i, p in enumerate(data["pipeline"]):
        start_name = model_label(start_lookup[p["start_model"]]) if p["start_model"] in start_lookup else p["start_model"]
        doc_name = model_label(doc_lookup[p["doc_model"]]) if p["doc_model"] in doc_lookup else p["doc_model"]
        pipeline_cell = f"{start_name} → {doc_name}" + (" **(best)**" if i == 0 else "")
        rows.append([
            str(i + 1), pipeline_cell, f"{p['start_macro_f1']:.3f}", f"{p['doc_macro_f1_e2e']:.3f}",
            f"{p['pq']:.3f}", f"{p['uas']:.3f}", f"{p['las']:.3f}",
        ])
    lines += [md_table(["#", "Pipeline (start_page → doc_type)", "start Macro F1", "doc Macro F1 (e2e)", "PQ", "UAS", "LAS"], rows), ""]

    lines += ["## Methodology & notes", ""]
    lines += [f"- {note}" for note in methodology]

    out_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/per_task"))
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "report_template.html")
    parser.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_report.html")
    args = parser.parse_args()

    print("Loading per-task leaderboards …")
    start_page = load_task(args.run_dir, "start_page")
    doc_type = load_task(args.run_dir, "doc_type")
    print(f"  start_page: {len(start_page)} models, doc_type: {len(doc_type)} models")

    print("Loading pipeline combinations …")
    pipeline = load_pipeline(args.run_dir, start_page)
    print(f"  {len(pipeline)} pipeline combinations")

    data = {"start_page": start_page, "doc_type": doc_type, "pipeline": pipeline}
    data_json = json.dumps(data, indent=None)

    template = args.template.read_text()
    if "__REPORT_DATA_JSON__" not in template:
        raise SystemExit(f"{args.template} has no __REPORT_DATA_JSON__ placeholder - was it edited by hand?")
    html = template.replace("__REPORT_DATA_JSON__", data_json)

    out_path = args.out or (args.run_dir / "eval_report.html")
    out_path.write_text(html)
    print(f"Wrote {out_path} ({len(html)} bytes)")

    write_markdown(args.run_dir, args.template, data, out_path.with_suffix(".md"))


if __name__ == "__main__":
    main()

"""
Runs a train.py --mode sequence --cached-embeddings checkpoint (as saved
under a runs/from_embeddings/<name>/ directory) back over its own precomputed
embeddings and writes one row per page with predicted Start page / Document
type / Layout Type / Functional Category, plus a predicted_segment_id -
matching predict.py's output schema exactly (and therefore directly
consumable by evaluate_segmentation.py), but without needing live
images/PageXML or re-deriving embeddings from a live backbone: the
checkpoint's projection weights are applied directly to the cached raw
embeddings instead.

Since --cached-embeddings sequence-mode training always freezes the
backbone (see train.py), the checkpoint's "embedder" state dict's backbone
weights are unchanged from the pretrained checkpoint precompute_embeddings.py
used - only its `projection` submodule (LayerNorm -> Linear -> GELU, the
same architecture PageEmbedder/TextEmbedder/MultimodalPageEmbedder build for
`project_to` - see models.py) was actually trained, so only those weights
are extracted and applied here.

Usage:
    python scripts/classification/predict_from_embeddings.py \\
        --run-dir runs/from_embeddings/vision_efficient \\
        --cached-embeddings data/embeddings_vision_dino_aug4 \\
        --split test --out runs/from_embeddings/vision_efficient/test_predictions.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from common import pick_device

from train_from_embeddings import EmbeddingSequenceDataset
from train_sequence_from_embeddings import ProjectedSequenceModel


def load_projection_state(embedder_state: dict) -> dict:
    """Extract just the `projection.*` keys from a full embedder checkpoint
    (backbone + projection) and strip the prefix, so they load directly into
    a bare `model.projection` nn.Sequential."""
    prefix = "projection."
    return {k[len(prefix):]: v for k, v in embedder_state.items() if k.startswith(prefix)}


@torch.no_grad()
def predict_split(model: ProjectedSequenceModel, ds: EmbeddingSequenceDataset, embeddings: np.ndarray,
                   classes: dict, device: torch.device) -> list[dict]:
    """One forward pass per PDF (batch size 1, no padding needed) - there are
    only a few dozen PDFs per split and no backbone forward pass is involved
    here (the embeddings are already precomputed), so batching for speed
    isn't worth the added complexity of tracking pdf/page identity through a
    padded, multi-PDF batch."""
    model.eval()
    doctype_classes, layout_classes, functional_classes = (
        classes["document_type"], classes["layout_type"], classes["functional_category"]
    )

    rows_out = []
    for pdf_df in ds.pdfs:
        emb = torch.from_numpy(embeddings[pdf_df["row_id"].to_numpy()]).float().unsqueeze(0).to(device)
        padding_mask = torch.zeros(1, len(pdf_df), dtype=torch.bool, device=device)
        out = model(emb, padding_mask, true_start_page=None)

        start_prob = torch.sigmoid(out["start_logits"][0].float()).cpu()
        doctype_prob = F.softmax(out["doctype_logits"][0].float(), dim=-1).cpu()
        layout_prob = F.softmax(out["layout_logits"][0].float(), dim=-1).cpu()
        functional_prob = F.softmax(out["functional_logits"][0].float(), dim=-1).cpu()
        segment_ids = out["segment_ids"][0].cpu()

        pdf_id = pdf_df["pdf_id"].iloc[0]
        for t in range(len(pdf_df)):
            row = pdf_df.iloc[t]
            doc_idx, lay_idx, func_idx = int(doctype_prob[t].argmax()), int(layout_prob[t].argmax()), int(functional_prob[t].argmax())
            start_p = float(start_prob[t])
            rows_out.append({
                "pdf_name": row["pdf_id"], "page_num": int(row["page_number"]), "split": row["split"],
                "start_page": row["start_page"], "document_type": row["document_type"],
                "layout_type": row["layout_type"], "functional_category": row["functional_category"],
                "predicted_start_page": "yes" if start_p > 0.5 else "no",
                "start_page_confidence": max(start_p, 1 - start_p),
                "predicted_document_type": doctype_classes[doc_idx],
                "document_type_confidence": float(doctype_prob[t, doc_idx]),
                "predicted_layout_type": layout_classes[lay_idx],
                "layout_type_confidence": float(layout_prob[t, lay_idx]),
                "predicted_functional_category": functional_classes[func_idx],
                "functional_category_confidence": float(functional_prob[t, func_idx]),
                "predicted_segment_id": f"{pdf_id}::seg{int(segment_ids[t])}",
            })
    return rows_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                         help="a train.py --mode sequence --cached-embeddings --out-dir, containing "
                              "model_config.json/classes.json/sequence_model.pt")
    parser.add_argument("--cached-embeddings", type=Path, required=True,
                         help="the same precompute_embeddings.py directory (or one with an identical manifest "
                              "schema/split assignment) the run was trained on")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    config = json.loads((args.run_dir / "model_config.json").read_text())
    classes = json.loads((args.run_dir / "classes.json").read_text())
    print(f"model config: {config}")

    embeddings = np.load(args.cached_embeddings / "embeddings.npy")
    manifest = pd.read_csv(args.cached_embeddings / "embeddings_manifest.tsv", sep="\t")
    raw_dim = embeddings.shape[1]
    print(f"{embeddings.shape[0]} pages, {manifest['pdf_id'].nunique()} PDFs, raw_dim={raw_dim}")

    doctype_classes, layout_classes, functional_classes = (
        classes["document_type"], classes["layout_type"], classes["functional_category"]
    )

    model = ProjectedSequenceModel(
        raw_dim, config["project_to"], len(doctype_classes), len(layout_classes), len(functional_classes),
        config["n_heads"], config["n_layers"], doc_classification=config.get("doc_classification", "late"),
    ).to(device)

    state = torch.load(args.run_dir / "sequence_model.pt", map_location=device)
    model.projection.load_state_dict(load_projection_state(state["embedder"]))
    model.seq_model.load_state_dict(state["seq_model"])
    print("loaded checkpoint")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_rows = []
    for split in splits:
        ds = EmbeddingSequenceDataset(embeddings, manifest, split, doctype_classes, layout_classes, functional_classes)
        print(f"{split}: {len(ds.pdfs)} PDFs")
        all_rows.extend(predict_split(model, ds, embeddings, classes, device))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out, sep="\t", index=False)
    print(f"\n{len(all_rows)} pages written to {args.out}")


if __name__ == "__main__":
    main()

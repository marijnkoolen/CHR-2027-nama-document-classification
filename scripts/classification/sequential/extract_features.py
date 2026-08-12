"""Extracts and caches page-level embeddings for one backbone, covering
every page in the corpus (not filtered to one task) - one cache per
backbone serves both start_page and doc_type training, and any other
future task built on the same labels TSV, since they only ever differ in
which *rows* they select from a cache afterwards, not in what the cache
itself contains.

Delegates the actual extraction to ../joint_legacy/precompute_embeddings.py
(run as a subprocess - it's a standalone CLI script, not a library module,
so a subprocess call is the natural way to reuse it rather than importing
its main() directly) - this is also why any HuggingFace checkpoint
(facebook/dinov2-small, microsoft/dit-large-finetuned-rvlcdip,
xlm-roberta-base, ...) works here alongside vgg16/efficientnet_b0, with no
extraction code of its own: it's exactly what precompute_embeddings.py
already supports via build_image_embedder()/TextEmbedder (see
../lib/models.py, shared by both scripts/classification/joint_legacy/ and
this directory). joint/ as a whole is an older, now-archived pipeline
variant that sequential/ supersedes - but this one file is still a live
dependency, not dead code, hence living under joint_legacy/ rather than
being deleted outright.

Two ways to build the manifest of pages to extract:
  - --data-root (default): every page in this project's own labels TSV (see
    lib/labels.py's build_extraction_manifest) - the usual case, for caching
    features over the labeled training/eval corpus.
  - --manifest: an arbitrary TSV/CSV of pages, column names configurable
    (--pdf-col etc., matching predict.py's own flags) - for caching features
    over a corpus that has no labels TSV at all, e.g. a large corpus of
    unlabeled pages you're about to run predict.py over. Column values in
    the cache's own embeddings_manifest.tsv that would normally come from
    the labels TSV (start_page/document_type/layout_type/functional_category)
    are left blank in this mode - harmless, since no consumer of the cache
    reads them back out (lib/embeddings.py joins by pdf_id/page_number only).

Usage:
    python scripts/classification/sequential/extract_features.py \\
        --data-root data --modality vision --image-backbone vgg16

    python scripts/classification/sequential/extract_features.py \\
        --data-root data --modality vision --image-backbone facebook/dinov2-small

    python scripts/classification/sequential/extract_features.py \\
        --data-root data --modality text --text-backbone bert-base-uncased

    python scripts/classification/sequential/extract_features.py \\
        --data-root data --modality multimodal \\
        --image-backbone microsoft/dit-large-finetuned-rvlcdip --text-backbone bert-base-uncased

    # a genuinely unlabeled corpus, ahead of predict.py
    python scripts/classification/sequential/extract_features.py \\
        --manifest new_corpus_manifest.tsv --image-root /path/to/new_corpus \\
        --cache-dir /path/to/new_corpus/embeddings --modality vision --image-backbone facebook/dinov2-small

Writes, under <cache-dir>/<backbone-slug>/ (see slugify_backbone()):
    embeddings.npy            (N_pages, D) float32
    embeddings_manifest.tsv   row_id, pdf_id (real dossier name - see
                               precompute_embeddings.py's --keep-real-ids),
                               page_number, split, start_page, document_type,
                               layout_type, functional_category
train_baseline.py/train_sequence.py/train_fusion.py/lib/predict.py all read
this format back via lib/embeddings.py, joining on (pdf_id, page_number).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from common import load_page_manifest
from labels import build_extraction_manifest
from tasks import get_task

PRECOMPUTE_EMBEDDINGS = Path(__file__).resolve().parent.parent / "joint_legacy" / "precompute_embeddings.py"


def slugify_backbone(name: str) -> str:
    """A HuggingFace checkpoint name (e.g. 'facebook/dinov2-small') isn't a
    valid single path component as-is; torchvision names (vgg16,
    efficientnet_b0) already are and pass through unchanged."""
    return name.replace("/", "__")


def backbone_cache_dir(cache_dir: Path, args) -> Path:
    if args.modality == "vision":
        return cache_dir / slugify_backbone(args.image_backbone)
    if args.modality == "text":
        return cache_dir / slugify_backbone(args.text_backbone)
    return cache_dir / f"{slugify_backbone(args.image_backbone)}+{slugify_backbone(args.text_backbone)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="required unless --manifest is given - the labels-TSV-driven corpus (see module docstring)",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to <data-root>/embeddings")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="extract over an arbitrary manifest instead of --data-root's labels TSV - for a corpus with no "
             "labels TSV at all, e.g. before predict.py on genuinely new data (see module docstring); "
             "--cache-dir is required in this mode, since there's no --data-root to default it from",
    )
    parser.add_argument("--pdf-col", default="pdf_name", help="--manifest only")
    parser.add_argument("--page-col", default="page_num", help="--manifest only")
    parser.add_argument("--image-col", default="img_path", help="--manifest only")
    parser.add_argument("--text-col", default="text_path", help="--manifest only")
    parser.add_argument(
        "--image-root", type=Path, default=Path(""),
        help="--manifest only - joined onto every --image-col/--text-col value; leave unset if --manifest "
             "already has fully-resolved paths",
    )
    parser.add_argument(
        "--limit-dossiers", type=int, default=None,
        help="--manifest only - extract only the first N dossiers in --manifest order, for timing a sample "
             "before committing to a large corpus",
    )
    parser.add_argument("--modality", choices=["vision", "text", "multimodal"], default="vision")
    parser.add_argument(
        "--image-backbone", default="vgg16",
        help="'vgg16'/'efficientnet_b0' (torchvision), or any HuggingFace checkpoint (e.g. "
             "facebook/dinov2-small, microsoft/dit-large-finetuned-rvlcdip) - used for --modality vision/multimodal",
    )
    parser.add_argument(
        "--text-backbone", default="bert-base-uncased",
        help="any HuggingFace checkpoint - used for --modality text/multimodal",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="re-extract even if this backbone's cache dir already has embeddings.npy")
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="don't error on a missing image (or transcription) file - drop rows with a missing image and "
             "continue instead (missing text always falls back to empty text, regardless). Off by default: "
             "a wrong --data-root or a path-formula mistake should fail fast, before any heavy lifting, not "
             "silently shrink the corpus every downstream task's cache is drawn from.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.manifest is not None:
        if args.cache_dir is None:
            raise SystemExit("--cache-dir is required when using --manifest (no --data-root to default it from)")
        cache_dir = args.cache_dir
    else:
        if args.data_root is None:
            raise SystemExit("--data-root is required unless --manifest is given")
        cache_dir = args.cache_dir or (args.data_root / "embeddings")

    out_dir = backbone_cache_dir(cache_dir, args)
    if not args.force and (out_dir / "embeddings.npy").exists():
        print(f"{out_dir / 'embeddings.npy'} already exists - skipping (pass --force to re-extract).")
        return

    if args.manifest is not None:
        manifest_df = load_page_manifest(
            args.manifest, args.pdf_col, args.page_col, args.image_col, args.text_col,
            image_root=args.image_root, limit_dossiers=args.limit_dossiers,
            allow_missing_files=args.allow_missing_files,
        )
    else:
        # Corpus-wide, unfiltered - always start_page's TaskConfig (see docstring
        # above and build_extraction_manifest's own docstring). The 'split'
        # column carried into embeddings_manifest.tsv here is just the labels
        # TSV's own raw value - purely informational, since every consumer in
        # this project builds its own train/val/test split via load_labels()
        # and joins against this cache by (dossier, page_num), not by reading
        # split back out of the cache itself.
        task = get_task("start_page")
        manifest_df = build_extraction_manifest(args.data_root, task, allow_missing_files=args.allow_missing_files)
    print(f"{len(manifest_df)} pages, {manifest_df['dossier'].nunique()} dossiers")

    manifest_path = cache_dir / "_extraction_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    cmd = [
        sys.executable, str(PRECOMPUTE_EMBEDDINGS),
        "--manifest", str(manifest_path), "--image-root", "",
        "--out-dir", str(out_dir), "--modality", args.modality,
        "--pdf-col", "dossier", "--page-col", "page_num", "--image-col", "img_path",
        "--markdown-col", "text_path", "--text-source", "markdown",
        "--doctype-col", "document_type", "--layout-col", "layout_type",
        "--functional-col", "functional_category", "--start-col", "start_page", "--split-col", "split",
        "--image-size", str(args.image_size), "--max-text-length", str(args.max_text_length),
        "--batch-size", str(args.batch_size), "--keep-real-ids",
    ]
    if args.modality in ("vision", "multimodal"):
        cmd += ["--image-backbone", args.image_backbone]
    if args.modality in ("text", "multimodal"):
        cmd += ["--text-backbone", args.text_backbone]
    if args.device is not None:
        cmd += ["--device", args.device]

    print(f"Extracting {args.modality} embeddings -> {out_dir} …")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"precompute_embeddings.py failed (exit {result.returncode})")


if __name__ == "__main__":
    main()

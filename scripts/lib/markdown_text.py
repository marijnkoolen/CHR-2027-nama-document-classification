"""
Extracts plain text from Qwen2.5-VL markdown-mode OCR output (see
scripts/ocr/run_qwen_vl.py/benchmark_vllm.py), as an alternative text
source to pagexml.py's PageXML transcriptions.

Strips markdown markup (headers, bold/italic, tables, lists) rather than
feeding it to the text backbone as-is: bert-base-uncased and this
project's other text backbones were pretrained on natural prose, not
markdown, so syntax characters mostly just consume --max-text-length
token budget without conveying learned signal, and the "structure" they
carry isn't a reliable feature anyway - different OCR runs render the same
kind of content differently (this project's own GOT-OCR2 output leaned
LaTeX-style headers, Qwen's leaned markdown tables) - real layout
structure is far better captured by the vision modality directly.

Renders markdown -> HTML -> plain text (rather than a regex-only strip)
so nested constructs (tables, lists) are handled correctly rather than
mangled.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup


def strip_markdown(text: str) -> str:
    """Renders *text* markdown -> HTML -> plain text (rather than a
    regex-only strip) so nested constructs (tables, lists) are handled
    correctly rather than mangled. Works equally on a whole document or a
    single line - extract_text uses it on a full file.

    Because this goes through full CommonMark block grammar, it is NOT
    safe for short OCR-derived fragments in isolation: a line starting
    "14. 3.1928" is read as an ordered-list item ("14." the marker, "3.1928"
    the body) and the marker is discarded, silently eating the day off a
    date. extract_dates.py needs markup gone without that reinterpretation
    risk, so it uses strip_inline_markdown below instead."""
    if not text or not text.strip():
        return ""
    html = markdown.markdown(text, extensions=["tables"])
    plain = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    # Collapse the blank-line runs markdown/BeautifulSoup's block-element
    # spacing tends to leave behind, without losing genuine paragraph breaks.
    lines = [line.strip() for line in plain.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_text(markdown_path: str | Path) -> str:
    """Returns "" if the path is empty/missing rather than raising, same
    convention as pagexml.py's extract_text - some pages may legitimately
    have no OCR output (e.g. blank pages)."""
    if not markdown_path or not Path(markdown_path).exists():
        return ""
    raw = Path(markdown_path).read_text(encoding="utf-8")
    return strip_markdown(raw)


# Matched emphasis/code delimiter pairs only: ***/___  (bold+italic),
# **/__ (bold), */_ (italic), ` (code). Longest markers first so "**" isn't
# consumed as two separate "*" matches. The backreference requires the
# closing delimiter to match the opening one exactly.
_INLINE_MARKUP = re.compile(r"(\*\*\*|___|\*\*|__|\*|_|`)(.+?)\1")

# A leading ATX heading marker ("# ", "## ", ... "###### ").
_HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}[ \t]+")


def strip_inline_markdown(text: str) -> str:
    """Strips matched inline markdown emphasis/code markers from *text* via
    targeted regexes, deliberately NOT going through full CommonMark
    block-level parsing (see strip_markdown's docstring for why that's
    unsafe here: it reinterprets a leading "<digits>. " or "- " as a list
    marker and discards it, which is exactly the shape a lot of dates
    start with). Only removes delimiters that appear in matched pairs, so
    an unpaired "*"/"_" (not uncommon in noisy OCR) is left alone rather
    than guessed at.

    Unlike strip_markdown, this never introduces line breaks that weren't
    already in *text* - safe to call per physical line."""
    if not text:
        return text
    prev = None
    while prev != text:
        prev = text
        text = _INLINE_MARKUP.sub(r"\2", text)
    return _HEADING_MARKER.sub("", text)


def strip_markdown_fencing(lines: list[str]) -> list[str]:
    """Removing lines markdown fencing."""
    content_lines = []
    if lines[0].startswith('```'):
        lines.pop(0)
    for line in lines:
        if line.startswith('```'):
            # break because Qwen sometimes adds notes at the end of the file
            # after the closing ``` line
            break
        content_lines.append(line)
    return content_lines


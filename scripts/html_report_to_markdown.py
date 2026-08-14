"""Converts one of this project's self-contained report.html files (base64-
embedded plots, narrative prose + data tables built by that directory's own
build_report.py) into a companion report.md that reads cleanly on GitHub.

Base64-embedded <img> tags are de-embedded back to references to the actual
PNG files sitting alongside report.html (matched by comparing decoded bytes,
not filename guessing, so this can't silently point at the wrong plot) -
inlining megabytes of base64 into a markdown file would defeat the point of
making it readable. Everything else (headings, paragraphs, tables) goes
through html2text.

Usage:
    python scripts/html_report_to_markdown.py data/dossier_size_model/report.html
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import html2text
from bs4 import BeautifulSoup


def force_block_breaks(soup: BeautifulSoup) -> None:
    """html2text doesn't reliably put blank lines around <figure> or
    <details> - both render fine visually in the original (CSS-positioned/
    collapsible), but without that CSS, consecutive figures and the
    <details><summary>caption</summary><table>...</table></details> tables
    run together into one unreadable paragraph. Force explicit breaks
    around both so the conversion reads as separate blocks."""
    for tag in soup.find_all(["figure", "details"]):
        tag.insert_before(soup.new_tag("br"))
        tag.insert_after(soup.new_tag("br"))
    # put each figcaption on its own italicised line below its image,
    # rather than glued onto the end of the image's markdown line.
    for cap in soup.find_all("figcaption"):
        cap.insert_before(soup.new_tag("br"))
        em = soup.new_tag("em")
        em.string = cap.get_text(strip=True)
        cap.replace_with(em)
    # "callout-label"/"tag" spans (e.g. "Superseded by §6", or a bullet's
    # lead-in like "The corrected headline") sit directly before the prose
    # they flag, same CSS-positioning story as above - bold and break
    # before the text that follows, rather than leaving it glued on.
    for label in soup.find_all("span", class_=("callout-label", "tag")):
        label.name = "strong"
        label.insert_after(soup.new_tag("br"))
    # the decorative "§N" section-number span sits as its own element right
    # before the heading it labels (positioned there by CSS) - fold it into
    # the heading text itself instead of leaving it as an orphaned line.
    for span in soup.find_all("span", class_="section-num"):
        heading = span.find_next_sibling(["h1", "h2", "h3"])
        if heading is not None:
            heading.insert(0, f"{span.get_text(strip=True)} ")
        span.decompose()


def deembed_images(soup: BeautifulSoup, report_dir: Path) -> None:
    png_files = sorted(report_dir.glob("*.png"))
    png_bytes = {p: p.read_bytes() for p in png_files}

    for img in soup.find_all("img"):
        src = img.get("src", "")
        m = re.match(r"^data:image/png;base64,(.+)$", src)
        if not m:
            continue
        decoded = base64.b64decode(m.group(1))
        match = next((p for p, b in png_bytes.items() if b == decoded), None)
        if match is None:
            raise SystemExit(
                f"embedded image (alt={img.get('alt')!r}) doesn't match any PNG in {report_dir} - "
                "was a plot regenerated without report.html being rebuilt too?"
            )
        img["src"] = match.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html_path", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to <html_path> with .md extension")
    args = parser.parse_args()

    html = args.html_path.read_text()
    soup = BeautifulSoup(html, "html.parser")

    # Only the <body> content matters - drop <head>/<style>/<script>.
    body = soup.body or soup
    deembed_images(body, args.html_path.parent)
    force_block_breaks(body)

    h2t = html2text.HTML2Text()
    h2t.body_width = 0  # don't hard-wrap prose - GitHub renders it fine either way, wrapping just adds noisy diffs
    h2t.ignore_images = False
    h2t.unicode_snob = True
    h2t.single_line_break = False
    md = h2t.handle(str(body))

    # html2text leaves occasional runs of 3+ blank lines around tables/headings; tidy without touching content.
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    out_path = args.out or args.html_path.with_suffix(".md")
    out_path.write_text(md)
    print(f"Wrote {out_path} ({len(md)} bytes, from {args.html_path})")


if __name__ == "__main__":
    main()

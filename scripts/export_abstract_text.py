"""Emit the abstract as plain text for the submission form.

The abstract has to be pasted into OpenReview by hand. Copying it out of the
built PDF looks like the safe move and is not: page 1's text layer carries the
line numbers the submission style adds, hyphens from justified line breaks, and
fi/fl ligature codepoints. Copying it out of ``paper/main.tex`` instead means
copying unexpanded macros. So generate it from both, and let ``make verify``
diff the result -- the same guard that already protects ``paper/numbers.tex``.

Usage:
    python3 scripts/export_abstract_text.py

Reads:
    paper/main.tex
    paper/numbers.tex
Writes:
    docs/abstract.txt
    docs/tldr.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"
NUMBERS = ROOT / "paper" / "numbers.tex"
DEFAULT_OUT = ROOT / "docs" / "abstract.txt"
DEFAULT_TLDR = ROOT / "docs" / "tldr.txt"

# OpenReview's Submission invitation caps the abstract field.
MAX_CHARS = 5000
MAX_TLDR_CHARS = 250
TLDR_TEX = (r"At \SeedsHopper{} seeds, selected replay rankings remain unresolved, "
            r"coarse paired contrasts survive in \ContrastBH{}/\ContrastNTests{} "
            r"environments, and metric disagreement needs a pair-specific noise "
            r"floor with model-dependent inference.")

_NEWCOMMAND = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def macros(*sources: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for src in sources:
        for line in src.splitlines():
            m = _NEWCOMMAND.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def expand(text: str, table: dict[str, str]) -> str:
    """Substitute \\Foo{} and \\Foo until nothing resolves, then strip markup."""
    for _ in range(10):
        before = text
        text = re.sub(r"\\([A-Za-z]+)\{\}",
                      lambda m: table.get(m.group(1), m.group(0)), text)
        text = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])",
                      lambda m: table.get(m.group(1), m.group(0)), text)
        if text == before:
            break
    for pattern in (r"\\textsf\{([^{}]*)\}", r"\\texttt\{([^{}]*)\}",
                    r"\\emph\{([^{}]*)\}", r"\\text\{([^{}]*)\}"):
        text = re.sub(pattern, r"\1", text)
    return text


def to_plain(tex: str) -> str:
    text = tex.replace("---", "\u2014").replace("--", "\u2013")
    text = text.replace(r"$\times$", "\u00d7").replace(r"\times", "\u00d7")
    text = (text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
                .replace(r"\{", "{").replace(r"\}", "}"))
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("~", " ").replace("$", "")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the abstract as plain text.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--tldr-output", default=str(DEFAULT_TLDR))
    args = ap.parse_args()

    main_tex = MAIN.read_text()
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S)
    if not body:
        sys.exit("no abstract environment in paper/main.tex")

    table = macros(NUMBERS.read_text(), main_tex)
    text = to_plain(expand(body.group(1), table))

    leftover = sorted(set(re.findall(r"\\[A-Za-z]+", text)))
    if leftover:
        sys.exit(f"unresolved macros in the abstract: {leftover}")
    if len(text) > MAX_CHARS:
        sys.exit(f"abstract is {len(text)} characters, over OpenReview's {MAX_CHARS}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"Wrote {out} ({len(text)} chars, {len(text.split())} words)")
    tldr = to_plain(expand(TLDR_TEX, table))
    if re.search(r"\\[A-Za-z]+", tldr) or len(tldr) > MAX_TLDR_CHARS:
        sys.exit(f"TLDR has unresolved macros or exceeds {MAX_TLDR_CHARS} chars")
    tldr_out = Path(args.tldr_output)
    tldr_out.parent.mkdir(parents=True, exist_ok=True)
    tldr_out.write_text(tldr + "\n")
    print(f"Wrote {tldr_out} ({len(tldr)} chars)")


if __name__ == "__main__":
    main()

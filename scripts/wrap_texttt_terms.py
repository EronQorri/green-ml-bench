#!/usr/bin/env python3
"""Wrap standalone 'Optuna'/'optuna' and 'CodeCarbon'/'codecarbon' occurrences
in \\texttt{}, preserving capitalization.

Left untouched:
  * occurrences already inside a \\texttt{} block (compound names like
    optuna.create_study or co2eq_codecarbon_kg)
  * occurrences inside a heading command (\\paragraph, \\section, ...), where
    \\texttt would break the header formatting. Any such terms that were
    previously wrapped are unwrapped again (idempotent cleanup)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "thesis_latex"

TERMS = "Optuna|optuna|CodeCarbon|codecarbon"

# A \texttt{...} block (one level of nested braces allowed).
TEXTTT_RE = re.compile(r"\\texttt\{(?:[^{}]|\{[^{}]*\})*\}")
# A sectioning/heading command and its argument (one level of nesting allowed).
HEADING_RE = re.compile(
    r"\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph|title)"
    r"\*?\{(?:[^{}]|\{[^{}]*\})*\}"
)
# Standalone word, not preceded by a backslash/word char, not followed by word char.
WORD_RE = re.compile(rf"(?<![\w\\])({TERMS})(?!\w)")
# A wrapped term to undo when it sits inside a heading.
WRAPPED_TERM_RE = re.compile(rf"\\texttt\{{({TERMS})\}}")


def transform(text: str) -> tuple[str, int]:
    # 1. Unwrap any term previously wrapped inside a heading.
    text = HEADING_RE.sub(lambda m: WRAPPED_TERM_RE.sub(r"\1", m.group(0)), text)

    protected: list[str] = []

    def stash(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    # 2. Shield existing \texttt{} blocks, then whole heading arguments.
    masked = TEXTTT_RE.sub(stash, text)
    masked = HEADING_RE.sub(stash, masked)

    # 3. Wrap remaining standalone terms.
    masked, n = WORD_RE.subn(lambda m: f"\\texttt{{{m.group(1)}}}", masked)

    restored = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], masked)
    return restored, n


def main() -> None:
    total = 0
    for path in sorted(ROOT.rglob("*.tex")):
        original = path.read_text(encoding="utf-8")
        new, n = transform(original)
        if new != original:
            path.write_text(new, encoding="utf-8")
            print(f"{path.relative_to(ROOT.parent)}: {n} wrap(s), headers cleaned")
            total += n
    print(f"Total: {total} wrap(s)")


if __name__ == "__main__":
    main()

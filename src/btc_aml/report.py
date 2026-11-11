"""Copy the generated result tables into README.md.

The README never contains hand-typed numbers in its results section. The
block between the two markers below is replaced with results/tables.md,
which analysis.py builds from results/metrics.csv:

    <!-- RESULTS:START -->
    ... generated, do not edit ...
    <!-- RESULTS:END -->
"""

from __future__ import annotations

from .utils import ROOT, get_logger, results_dir

log = get_logger()

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(cfg: dict) -> bool:
    """Replace the marked block in README.md. Returns True if the file was updated."""
    readme = ROOT / "README.md"
    tables = results_dir(cfg) / "tables.md"
    if not readme.exists() or not tables.exists():
        log.info("README.md or results/tables.md missing; nothing to update")
        return False
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        log.info("README.md has no %s ... %s block; nothing to update", START, END)
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    note = ("_Everything between these markers is generated from `results/metrics.csv` by "
            "`python scripts/make_readme_table.py`. Do not edit by hand._")
    body = tables.read_text(encoding="utf-8").strip()
    readme.write_text(f"{head}{START}\n{note}\n\n{body}\n{END}{tail}", encoding="utf-8")
    log.info("updated the results block of %s", readme)
    return True

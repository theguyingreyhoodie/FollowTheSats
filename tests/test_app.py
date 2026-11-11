"""Smoke test: every page of the Streamlit app renders without an exception.

Skipped when no pipeline run has produced results yet (the app needs saved artifacts).
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Model comparison", "Transaction inspector", "Time travel"]


@pytest.fixture(scope="module")
def app():
    if not (ROOT / "results" / "metrics.csv").exists() or not (ROOT / "models").exists():
        pytest.skip("no saved results; run: python run_all.py --config configs/quick.yaml")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value  # the page drew its title


def test_inspector_buttons(app):
    app.run()
    app.sidebar.radio[0].set_value("Transaction inspector").run()
    for label in ("Random illicit", "Random flagged", "Random licit"):
        button = next(b for b in app.button if b.label == label)
        button.click().run()
        assert not app.exception, [e.value for e in app.exception]

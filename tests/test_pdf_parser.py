import pytest

from ingestion import pdf_parser
from ingestion.pdf_parser import (
    UnreliableExtractionError,
    extract_text_from_pdf,
    looks_letter_spaced,
    rejoin_spaced_letters,
    single_letter_ratio,
)

SPACED = "D a t a  A n a l y s t\nP y t h o n  a n d  S Q L  e x p e r t"
NORMAL = "Data Analyst with five years of Python and SQL experience building dashboards for finance teams."


def test_detects_letter_spaced_text():
    assert looks_letter_spaced(SPACED * 3)
    assert not looks_letter_spaced(NORMAL)


def test_short_text_is_not_flagged():
    assert single_letter_ratio("a b") == 0.0


def test_rejoin_restores_words_using_double_space_gaps():
    assert rejoin_spaced_letters("D a t a  A n a l y s t") == "Data Analyst"


def test_rejoin_leaves_normal_text_alone():
    assert rejoin_spaced_letters("Plan A or B\n" + NORMAL) == "Plan A or B\n" + NORMAL


class FakePage:
    def __init__(self, texts):
        self.texts = texts

    def extract_text(self, extraction_mode="plain"):
        return self.texts[extraction_mode]


def patch_reader(monkeypatch, plain, layout=None):
    pages = [FakePage({"plain": plain, "layout": layout if layout is not None else plain})]
    monkeypatch.setattr(pdf_parser, "PdfReader", lambda f: type("R", (), {"pages": pages})())


def test_spaced_text_is_repaired(monkeypatch):
    patch_reader(monkeypatch, (SPACED + "\n") * 4)
    text = extract_text_from_pdf("x.pdf")
    assert "Data Analyst" in text and not looks_letter_spaced(text)


def test_unrepairable_text_raises(monkeypatch):
    # Single spaces everywhere, no word gaps and mostly single letters that
    # aren't in runs of 3+ -> can't be re-joined.
    patch_reader(monkeypatch, "a b " * 40)
    with pytest.raises(UnreliableExtractionError):
        extract_text_from_pdf("x.pdf")


def test_normal_text_passes_through(monkeypatch):
    patch_reader(monkeypatch, NORMAL)
    assert extract_text_from_pdf("x.pdf") == NORMAL


def test_too_little_text_returns_none(monkeypatch):
    patch_reader(monkeypatch, "hello")
    assert extract_text_from_pdf("x.pdf") is None

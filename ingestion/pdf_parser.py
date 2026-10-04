import re

from pypdf import PdfReader

# If more than this share of whitespace-separated tokens are single letters,
# the extraction is treated as broken (typically a font that makes pypdf emit
# "D a t a  A n a l y s t"). Normal resumes sit around 0-4%.
SPACED_LETTER_THRESHOLD = 0.25
MIN_TOKENS_FOR_CHECK = 10
MIN_TEXT_LENGTH = 50

_WORD_GAP = re.compile(r"\s{2,}")


class UnreliableExtractionError(Exception):
    """Text was extracted but is garbled (e.g. letter-spaced) and can't be trusted."""


def single_letter_ratio(text):
    """Share of whitespace-separated tokens that are a single alphabetic character."""
    tokens = text.split()
    if len(tokens) < MIN_TOKENS_FOR_CHECK:
        return 0.0
    return sum(1 for t in tokens if len(t) == 1 and t.isalpha()) / len(tokens)


def looks_letter_spaced(text):
    return single_letter_ratio(text) > SPACED_LETTER_THRESHOLD


def rejoin_spaced_letters(text):
    """
    Re-join words that were extracted one character at a time.

    Within each line, segments separated by 2+ spaces are treated as words. A
    segment made only of single characters separated by single spaces
    ("D a t a") is collapsed ("Data"). If the extractor emitted no wider word
    gap, adjacent words in a run can't be told apart and will merge.
    """
    lines = []
    for line in text.split("\n"):
        segments = []
        for segment in _WORD_GAP.split(line):
            parts = segment.split(" ")
            if len(parts) >= 3 and all(len(p) == 1 and p.isalnum() for p in parts):
                segment = "".join(parts)
            segments.append(segment)
        lines.append(" ".join(segments))
    return "\n".join(lines)


def _collapse_whitespace(text):
    return " ".join(text.split())


def _extract_pages(reader, mode):
    kwargs = {} if mode == "plain" else {"extraction_mode": mode}
    return "\n".join(page.extract_text(**kwargs) or "" for page in reader.pages)


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded file object or a file path.

    Returns the text, or None if the PDF can't be read or has too little text
    (e.g. a scanned image). Raises UnreliableExtractionError if text was
    extracted but is still letter-spaced after the repair attempts, so callers
    can tell the user instead of scoring garbage.
    """
    try:
        reader = PdfReader(pdf_file)
        for mode in ("plain", "layout"):
            raw = _extract_pages(reader, mode)
            if looks_letter_spaced(raw):
                raw = rejoin_spaced_letters(raw)
            if not looks_letter_spaced(raw):
                text = _collapse_whitespace(raw)
                return text if len(text) >= MIN_TEXT_LENGTH else None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    raise UnreliableExtractionError(
        "This PDF's text could not be read reliably (characters are split apart "
        "by the embedded font). Try re-exporting it from the original document."
    )

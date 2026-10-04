import re

from pypdf import PdfReader

# If more than this share of whitespace-separated tokens are single letters,
# the extraction is treated as broken (typically a font that makes pypdf emit
# "D a t a  A n a l y s t"). Normal resumes sit around 0-4%.
SPACED_LETTER_THRESHOLD = 0.25
MIN_TOKENS_FOR_CHECK = 10
MIN_TEXT_LENGTH = 50
MAX_PLAUSIBLE_WORD_LENGTH = 30

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


def _merge_single_char_runs(tokens):
    """
    Greedily merge every run of 2+ consecutive single-character tokens.

    A run is closed by the first multi-character token, so partially broken
    text ("i n transforming") still repairs. Single punctuation characters in
    a run are glued on without a space ("c l e a r ," -> "clear,"). A run made
    only of punctuation is left alone.
    """
    out, run = [], []

    def flush():
        if len(run) >= 2 and any(ch.isalnum() for ch in run):
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for token in tokens:
        if len(token) == 1:
            run.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return out


def _repair_failed(text):
    """
    True if text is still letter-spaced, or the repair merged words together
    (no word gaps to split on gives absurd run-on "words" like "abababab...").
    """
    return looks_letter_spaced(text) or any(
        t.isalpha() and len(t) > MAX_PLAUSIBLE_WORD_LENGTH for t in text.split()
    )


def rejoin_spaced_letters(text):
    """
    Re-join words that were extracted one character at a time.

    Within each line, segments separated by 2+ spaces are treated as words
    (extractors that space out letters usually keep a wider gap between real
    words), and each segment is repaired by _merge_single_char_runs. If the
    extractor emitted no wider word gap, adjacent words in a run can't be told
    apart and will merge. Only call this on text already flagged by
    looks_letter_spaced: on healthy text it could merge legitimate single-letter
    words ("Plan A B").
    """
    lines = []
    for line in text.split("\n"):
        segments = [
            " ".join(_merge_single_char_runs(segment.split(" ")))
            for segment in _WORD_GAP.split(line)
        ]
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

    Returns the text, or None if the PDF can't be read or genuinely has too
    little text (e.g. a scanned image). Raises UnreliableExtractionError if any
    extraction mode produced text that was letter-spaced and could not be
    repaired, so callers can tell the user instead of scoring garbage or
    mistaking a readable-but-garbled resume for a blank one.
    """
    try:
        reader = PdfReader(pdf_file)
        saw_unrepairable = False
        for mode in ("plain", "layout"):
            raw = _extract_pages(reader, mode)
            if looks_letter_spaced(raw):
                raw = rejoin_spaced_letters(raw)
                if _repair_failed(raw):
                    saw_unrepairable = True
                    continue
            text = _collapse_whitespace(raw)
            if len(text) >= MIN_TEXT_LENGTH:
                return text
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    if saw_unrepairable:
        raise UnreliableExtractionError(
            "This PDF's text could not be read reliably (characters are split apart "
            "by the embedded font). Try re-exporting it from the original document."
        )
    return None

"""Word error rate scoring, vendored from
~/src/tests/transcript/test_vibevoice_diarization.py (lines ~198-221) rather
than imported across repos -- a test in one repo importing a file in another
breaks the first time either moves. Attribution kept here deliberately.
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]*>")  # ground truth annotation tags, e.g. <ST/>, <PName>
_PUNCT_RE = re.compile(r"[^\w']+")


def tokenize(text: str) -> list[str]:
    text = _TAG_RE.sub(" ", text.lower())
    text = _PUNCT_RE.sub(" ", text)
    return text.split()


def word_error_rate(reference_words: list[str], hypothesis_words: list[str]) -> float:
    """Levenshtein edit distance at the word level, normalized by reference length."""
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    prev_row = list(range(len(hypothesis_words) + 1))
    for i, r_word in enumerate(reference_words, start=1):
        curr_row = [i] + [0] * len(hypothesis_words)
        for j, h_word in enumerate(hypothesis_words, start=1):
            if r_word == h_word:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j - 1], prev_row[j], curr_row[j - 1])
        prev_row = curr_row
    return prev_row[len(hypothesis_words)] / len(reference_words)

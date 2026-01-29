"""Utilities used for data processing."""

import gruut


def normalize_text(text: str) -> str:
    """Normalizes text.

    Normalization includes converting currency, numbers to words, and standardizing punctuation.
    """

    sentences = [sentence.text_with_ws for sentence in gruut.sentences(text, phonemes=False)]
    return ' '.join(sentences)


def sanitize_categorized_options(cat_opts: dict[str, str]) -> dict[str, str]:
    """Sanitizes categorized options by removing unwanted characters."""

    replacement = {
        '–': '-',
        ' ': ' ',
        '…': '',
    }

    return {
        key: ''.join(replacement.get(char, char) for char in value).strip()
        for key, value in cat_opts.items()
    }

import re

from core.abbreviations import ABBREVIATIONS

LATIN_TO_CYRILLIC = str.maketrans("aceopxkmtbh", "асеорхкмтвн")
SIZE_SEPARATOR = re.compile(r"(?<=\d)\s*[х×*]\s*(?=\d)")
DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
WHITESPACE = re.compile(r"\s+")
ABBREVIATION_PATTERNS = [(re.compile(rf"(?<![а-я]){pattern}"), replacement) for pattern, replacement in ABBREVIATIONS]


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е").translate(LATIN_TO_CYRILLIC)
    text = SIZE_SEPARATOR.sub("х", text)
    text = DECIMAL_COMMA.sub(".", text)
    for pattern, replacement in ABBREVIATION_PATTERNS:
        text = pattern.sub(replacement, text)
    return WHITESPACE.sub(" ", text).strip()

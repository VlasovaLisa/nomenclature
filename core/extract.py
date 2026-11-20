import difflib
import re
from dataclasses import asdict, dataclass, fields

from core.normalize import normalize

SEAMLESS, WELDED, CASING, TUBING, TUBING_HIGH_TIGHT = "бесшовная", "электросварная", "обсадная", "НКТ", "НКМ"
OIL_COUNTRY_TYPES = {CASING, TUBING, TUBING_HIGH_TIGHT}

PIPE_TYPE_RULES = [
    (re.compile(r"обсадн"), CASING),
    (re.compile(r"(?<![а-я])нкм(?![а-я])"), TUBING_HIGH_TIGHT),
    (re.compile(r"(?<![а-я])нкт|насосно-компрессорн"), TUBING),
    (re.compile(r"бесшовн"), SEAMLESS),
    (re.compile(r"электросвар"), WELDED),
]
FUZZY_TYPE_WORDS = {"обсадная": CASING, "бесшовная": SEAMLESS, "электросварная": WELDED}
FUZZY_CUTOFF = 0.8

SIZE = re.compile(r"(?<![\d.])(\d{2,4}(?:\.\d+)?)х(\d{1,2}(?:\.\d+)?)(?![\d.]*х)")
ANY_SIZE = re.compile(r"\d+(?:\.\d+)?х\d+(?:\.\d+)?")
STEEL_GRADE = re.compile(r"(?<![а-я\d])(\d{2}[а-я]{1,3}\d?[а-я]{0,3}\d?[а-я]?(?:-у)?)(?![а-я\d])")
NUMERIC_STEEL_GRADE = re.compile(r"(?:сталь\s*|(?<![а-я])в-)(\d{2})(?![\d.а-я])")
STRENGTH_CLASS = re.compile(r"(?<![а-я])к-?\s?(\d{2})(?!\d)")
STRENGTH_GROUP_LETTERS = "декмлр"
API_STRENGTH_GROUP = re.compile(r"(?<![а-я\d])(n80|l80|r95|р110|j55|к55|т95|с90|q125)(?![\d])")
EXPLICIT_STRENGTH_GROUP = re.compile(rf"группа прочности\s+([{STRENGTH_GROUP_LETTERS}])(?![а-я])")
STRENGTH_GROUP_AFTER_SIZE = re.compile(rf"\d+х\d+(?:\.\d+)?\s*-?\s*([{STRENGTH_GROUP_LETTERS}])с?(?![а-я\d])")
STRENGTH_GROUP_BEFORE_SIZE = re.compile(rf"(?<![а-я])([{STRENGTH_GROUP_LETTERS}])с?\d?\s*-?\s*\d+(?:\.\d+)?х\d")
PIPE_GROUP = re.compile(r"группа\s*(\d)(?!\d)")
EXCLUSION_CLAUSE = re.compile(r"\(\s*исключая[^)]*\)")


@dataclass(frozen=True)
class PipeSpec:
    pipe_type: str | None = None
    diameter: float | None = None
    wall_thickness: float | None = None
    steel_grade: str | None = None

    def known_fields(self) -> dict[str, object]:
        return {name: value for name, value in asdict(self).items() if value is not None}

    def compare(self, other: "PipeSpec") -> dict[str, bool | None]:
        return {
            field.name: None if mine is None or theirs is None else bool(mine == theirs)
            for field in fields(self)
            for mine, theirs in [(getattr(self, field.name), getattr(other, field.name))]
        }

    def mismatches(self, other: "PipeSpec") -> list[str]:
        return [name for name, matched in self.compare(other).items() if matched is False]

    @classmethod
    def from_dict(cls, values: dict) -> "PipeSpec":
        return cls(
            pipe_type=values.get("pipe_type"),
            diameter=_to_float(values.get("diameter")),
            wall_thickness=_to_float(values.get("wall_thickness")),
            steel_grade=normalize_grade(values["steel_grade"]) if values.get("steel_grade") else None,
        )


def _to_float(value) -> float | None:
    return None if value is None else float(value)


def normalize_grade(grade: str) -> str:
    return normalize(grade).upper()


def extract_pipe_type(text: str) -> str | None:
    for pattern, pipe_type in PIPE_TYPE_RULES:
        if pattern.search(text):
            return pipe_type
    for word in re.findall(r"[а-я]{6,}", text):
        close = difflib.get_close_matches(word, FUZZY_TYPE_WORDS, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            return FUZZY_TYPE_WORDS[close[0]]
    return None


def extract_size(text: str) -> tuple[float | None, float | None]:
    match = SIZE.search(text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def extract_strength_group(text: str) -> str | None:
    for pattern in (EXPLICIT_STRENGTH_GROUP, API_STRENGTH_GROUP, STRENGTH_GROUP_AFTER_SIZE, STRENGTH_GROUP_BEFORE_SIZE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def extract_steel_grade(text: str, pipe_type: str | None) -> str | None:
    if pipe_type in OIL_COUNTRY_TYPES:
        candidates = (extract_strength_group(text),)
    else:
        text_without_sizes = ANY_SIZE.sub(" ", text)
        strength_class = STRENGTH_CLASS.search(text_without_sizes)
        grade = STEEL_GRADE.search(text_without_sizes) or NUMERIC_STEEL_GRADE.search(text_without_sizes)
        candidates = (strength_class and f"к{strength_class.group(1)}", grade and grade.group(1))
    found = next((candidate for candidate in candidates if candidate), None)
    return found.upper() if found else None


def extract_pipe_group(text: str) -> int | None:
    match = PIPE_GROUP.search(normalize(text))
    return int(match.group(1)) if match else None


def extract_spec(text: str) -> PipeSpec:
    normalized = EXCLUSION_CLAUSE.sub(" ", normalize(text))
    pipe_type = extract_pipe_type(normalized)
    diameter, wall_thickness = extract_size(normalized)
    return PipeSpec(
        pipe_type=pipe_type,
        diameter=diameter,
        wall_thickness=wall_thickness,
        steel_grade=extract_steel_grade(normalized, pipe_type),
    )

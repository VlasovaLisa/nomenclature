import re

from core.extract import SEAMLESS, WELDED, PipeSpec

TEMPLATE_PIPE_TYPES = {SEAMLESS, WELDED}
TEMPLATE_STANDARD = "ТТТ-01.02.04-01"
STRENGTH_CLASS = re.compile(r"К\d{2}")


def format_size(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def supports_template(spec: PipeSpec) -> bool:
    return (
        spec.pipe_type in TEMPLATE_PIPE_TYPES
        and spec.diameter is not None
        and spec.wall_thickness is not None
        and spec.steel_grade is not None
        and STRENGTH_CLASS.fullmatch(spec.steel_grade) is not None
    )


def build_name(spec: PipeSpec, group: int | None) -> str:
    if not supports_template(spec):
        raise ValueError(f"Шаблон не поддерживает параметры {spec}")
    group_part = f" группа {group}" if group is not None else ""
    size = f"{format_size(spec.diameter)}х{format_size(spec.wall_thickness)}"
    return f"Труба {spec.pipe_type}{group_part} {size} {spec.steel_grade} {TEMPLATE_STANDARD}"

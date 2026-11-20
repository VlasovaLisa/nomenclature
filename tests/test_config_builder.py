import pytest

from core.config_builder import build_name, format_size, supports_template
from core.extract import PipeSpec, extract_spec


@pytest.mark.parametrize(("value", "expected"), [(219.0, "219"), (7.4, "7,4"), (10.0, "10")])
def test_format_size(value, expected):
    assert format_size(value) == expected


def test_build_name_follows_catalog_template():
    spec = PipeSpec("бесшовная", 219, 10, "К48")

    assert build_name(spec, group=2) == "Труба бесшовная группа 2 219х10 К48 ТТТ-01.02.04-01"


def test_build_name_without_group():
    assert build_name(PipeSpec("электросварная", 720, 12, "К52"), group=None) == "Труба электросварная 720х12 К52 ТТТ-01.02.04-01"


def test_generated_name_preserves_all_parameters():
    spec = PipeSpec("электросварная", 530, 10, "К60")

    assert extract_spec(build_name(spec, group=4)) == spec


@pytest.mark.parametrize(
    "spec",
    [
        PipeSpec("обсадная", 114, 7.4, "Е"),
        PipeSpec("электросварная", 114, 5, "09Г2С"),
        PipeSpec("бесшовная", 89, None, "К52"),
        PipeSpec(None, 89, 8, "К52"),
    ],
)
def test_template_is_not_applied_outside_supported_format(spec):
    assert not supports_template(spec)
    with pytest.raises(ValueError):
        build_name(spec, group=2)

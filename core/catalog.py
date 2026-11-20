from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from core.extract import extract_pipe_group, extract_spec
from core.loader import ExcelSource, build_catalog, load_records

SPEC_COLUMNS = ["pipe_type", "diameter", "wall_thickness", "steel_grade"]


def add_specs(catalog: pd.DataFrame) -> pd.DataFrame:
    specs = pd.DataFrame([asdict(extract_spec(name)) for name in catalog["name"]], index=catalog.index)
    groups = catalog["name"].map(extract_pipe_group).astype("Int64").rename("pipe_group")
    return pd.concat([catalog, specs, groups], axis=1)


def prepare_catalog(sources: Iterable[ExcelSource]) -> pd.DataFrame:
    return add_specs(build_catalog(load_records(sources)))

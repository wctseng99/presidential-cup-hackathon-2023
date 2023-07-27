from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    return Path(".", "data")


@pytest.fixture
def result_dir(tmp_path) -> Path:
    result_dir = Path(tmp_path, "results")
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir

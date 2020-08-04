import os
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    "path_to_python_file",
    [
        "quickstart_classification.py",
        "quickstart_regression.py",
        "quickstart_segmentation.py",
    ],
)
def test_that_the_quickstart_notebooks_do_not_not_crash(
    path_to_python_file: str,
) -> None:
    os.chdir(THIS_DIR)
    return_value = os.system(f"poetry run python {path_to_python_file}")
    assert return_value == 0

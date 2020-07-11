import os
from pathlib import Path

THIS_DIR = Path(__file__).parent
QUICKSTART_PY = "quickstart_regression.py"


def test_that_the_quickstart_notebook_does_not_crash() -> None:
    os.chdir(THIS_DIR)
    return_value = os.system(f"poetry run python {QUICKSTART_PY}")
    assert return_value == 0

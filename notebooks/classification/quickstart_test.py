import os
from pathlib import Path

import jupytext

THIS_DIR = Path(__file__).parent
QUICKSTART_IPYNB = "quickstart.ipynb"
QUICKSTART_PY = "quickstart.py"


def test_that_the_quickstart_notebook_does_not_crash():
    os.chdir(THIS_DIR)
    return_value = os.system("poetry run python quickstart.py")
    assert return_value == 0


def test_that_ipynb_and_py_files_are_synchronized():
    py_from_ipynb = jupytext.writes(
        jupytext.read(THIS_DIR / QUICKSTART_IPYNB), fmt="py:percent"
    )

    py_from_py = jupytext.writes(
        jupytext.read(THIS_DIR / QUICKSTART_PY), fmt="py:percent"
    )
    assert py_from_ipynb == py_from_py

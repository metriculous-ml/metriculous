from pathlib import Path
from subprocess import check_output

THIS_DIR = Path(__file__).parent


def test_that_binder_requirements_txt_is_up_to_date() -> None:
    exported = check_output(["poetry", "export", "-f", "requirements.txt"]).decode()
    with open(THIS_DIR.parent / ".binder" / "requirements.txt") as requirements_txt:
        on_disk = requirements_txt.read()
    assert on_disk == exported

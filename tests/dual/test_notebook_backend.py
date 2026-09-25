"""Importing numerical error metrics must preserve an interactive caller's backend."""

import subprocess
import sys


def test_campaign_import_preserves_selected_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                'import matplotlib; matplotlib.use("svg"); '
                "import acoustic_freeform.dual.campaign; "
                'assert matplotlib.get_backend().lower() == "svg"'
            ),
        ],
        check=True,
    )

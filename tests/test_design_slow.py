"""End-to-end design of a published lens with the acoustic solver (minutes; run with -m slow)."""

import pytest

from acoustic_freeform.design import Singlet, design

pytestmark = pytest.mark.slow

# Silva-Lora and Torres, JOSA A 37, 1155 (2020), Fig. 9b, scaled 1/14 (study S09).
FIG9B = Singlet(object_m="-inf", image_m=0.09, thickness_m=0.01, intermediate_m=0.13795,
                scale=1 / 14, name="Fig. 9b")


def test_fig9b_needs_one_tone_and_reproduces_the_s09_cell():
    recipe = design(FIG9B, startup=False, progress=lambda *_: None)
    assert recipe.status == "command-only"
    assert recipe.attempts[0]["bands_hz"] == [1.5e6]
    assert recipe.prediction["frozen_pupil_error_m"] < 1e-9
    assert recipe.prediction["convex_bound_m"] < 1e-10
    # The pin heights of the saved S09 design (artifacts/.../fig9b/design/config.json).
    pins = recipe.settings["pin_heights_z_m"]
    assert pins == pytest.approx([0.0005250582175621069, 0.0007539410169507819], abs=1e-12)
    assert recipe.settings["drive"]["source_power_w"] <= 1.0 + 1e-6

"""Verify executed emitter notebook outputs and offline carrier-wave playback."""

import hashlib
import json
from pathlib import Path

import nbformat
from playwright.sync_api import sync_playwright

from acoustic_freeform.provenance import capture_execution


def main():
    root = Path(__file__).resolve().parents[1]
    stem = "09_emitter_driven_equilibrium"
    html = root / f"artifacts/notebooks/{stem}.html"
    notebook_path = root / f"artifacts/notebooks/{stem}.executed.ipynb"
    output = root / "artifacts/noa61-emitter-2026-09-23/notebook-check-inline-v2"
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    static_figures = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
    assert static_figures >= 3, "Setup, surface/spot and load figures must be embedded"
    last = "".join(o.get("text", "") for o in code[-1].outputs)
    numerical = json.loads(last)
    assert numerical["fresh_force_compliance_defect_m"] < 1e-12
    assert 0.5e-6 < numerical["max_geometric_spot_radius_m"] < 0.52e-6
    assert not numerical["all_rays_transmitted"]
    assert not numerical["formation_trajectory"]
    inputs = [
        root / f"notebooks/{stem}.ipynb",
        notebook_path,
        html,
        root / "artifacts/noa61-emitter-2026-09-23/verify-448-c4-clear-spot/design104/state.npz",
        Path(__file__),
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output / "config.json").write_text(json.dumps({"input_sha256": hashes}, indent=2) + "\n")
    records, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        slider = page.locator('input[id^="_anim_slider"]')
        panel = page.locator('img[id^="_anim_img"]')
        page.wait_for_function('document.querySelector("input[id^=_anim_slider]").max === "31"')
        for index in (0, 8, 16):
            slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}', index)
            page.wait_for_function('document.querySelector("img[id^=_anim_img]").complete')
            pixels = panel.screenshot(path=str(output / f"frame-{index}.png"))
            records.append(
                {
                    "frame": index,
                    "physical_time_s": index / (32 * 3600000),
                    "image_sha256": hashlib.sha256(pixels).hexdigest(),
                }
            )
        assert len({r["image_sha256"] for r in records}) == 3
        page.get_by_role("button", name="First frame", exact=True).click()
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_function(
            'Number(document.querySelector("input[id^=_anim_slider]").value) >= 2'
        )
        page.get_by_role("button", name="Pause", exact=True).click()
        paused = slider.input_value()
        page.wait_for_timeout(350)
        assert slider.input_value() == paused
        assert not errors, errors
        browser.close()
    result = {
        "numerical_checks": numerical,
        "frames": records,
        "embedded_static_figures": static_figures,
        "offline_play_pause_passed": True,
        "page_errors": errors,
        "scope": "Same-grid numerical reproduction and rendering checks; not formation or refinement validation.",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Emitter notebook verification\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

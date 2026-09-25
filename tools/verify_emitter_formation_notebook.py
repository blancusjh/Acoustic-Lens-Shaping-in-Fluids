"""Check initial surfaces/spot and real saved-state playback in notebook 10."""

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
from playwright.sync_api import sync_playwright

from acoustic_freeform.provenance import capture_execution


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/noa61-emitter-2026-09-23"
    html = root / "artifacts/notebooks/10_emitter_formation.html"
    notebook = root / "artifacts/notebooks/10_emitter_formation.executed.ipynb"
    trajectory = base / "viscous-formation/trajectory.npz"
    output = base / "formation-notebook-check"
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (html, notebook, trajectory, Path(__file__))
    }
    (output / "config.json").write_text(json.dumps({"input_sha256": hashes}, indent=2) + "\n")
    n = nbformat.read(notebook, as_version=4)
    code = [c for c in n.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert sum("image/png" in o.get("data", {}) for c in code for o in c.outputs) >= 1
    assert not any(
        "FigureCanvasAgg is non-interactive" in o.get("text", "") for c in code for o in c.outputs
    )
    saved = np.load(trajectory)
    times, states = saved["time_s"], saved["coefficients_m"]
    assert np.max(abs(states[0])) == 0
    records, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1700, "height": 1100})
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        slider = page.locator('input[id^="_anim_slider"]')
        image = page.locator('img[id^="_anim_img"]')
        page.wait_for_function(
            f'document.querySelector("input[id^=_anim_slider]").max === "{len(times) - 1}"'
        )
        for k in (0, len(times) // 2, len(times) - 1):
            slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}', k)
            page.wait_for_function('document.querySelector("img[id^=_anim_img]").complete')
            pixels = image.screenshot(path=str(output / f"frame-{k}.png"))
            records.append(
                {
                    "frame": k,
                    "physical_time_s": float(times[k]),
                    "state_sha256": hashlib.sha256(states[k].tobytes()).hexdigest(),
                    "image_sha256": hashlib.sha256(pixels).hexdigest(),
                }
            )
        assert len({r["state_sha256"] for r in records}) == 3
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
        "embedded_frames": len(times),
        "initial_flat_state_checked": True,
        "static_initial_surface_and_spot_embedded": True,
        "offline_play_pause_passed": True,
        "frames": records,
        "page_errors": errors,
        "scope": "Rendering and saved-state correspondence; not physical validity or convergence of the partial Stokes trajectory.",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Formation notebook browser check\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

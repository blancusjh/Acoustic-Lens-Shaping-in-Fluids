"""Check offline embedded animation playback and saved-state frame correspondence."""

import hashlib
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from acoustic_freeform.core.provenance import capture_execution


def main():
    root = Path(__file__).resolve().parents[1]
    html = root / "artifacts/studies/S03-stigmatic-target-ideal-load/notebooks/08_prescribed_load_formation.html"
    run = root / "artifacts/studies/S03-stigmatic-target-ideal-load/ideal-load-2026-09-23/viscous-fluid-refined"
    output = root / "artifacts/studies/S03-stigmatic-target-ideal-load/ideal-load-2026-09-23/notebook-browser-check"
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    (output / "config.json").write_text(
        json.dumps({"html": str(html), "run": str(run)}, indent=2) + "\n"
    )
    saved = np.load(run / "trajectory.npz")
    frames = np.unique(np.linspace(0, len(saved["time_s"]) - 1, 81).astype(int))
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        slider = page.locator('input[id^="_anim_slider"]')
        panel = page.locator('img[id^="_anim_img"]')
        page.wait_for_function('document.querySelector("input[id^=_anim_slider]").max === "80"')
        for index in (0, 40, 80):
            slider.evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}', index)
            page.wait_for_function('document.querySelector("img[id^=_anim_img]").complete')
            pixels = panel.screenshot(path=str(output / f"frame-{index}.png"))
            records.append(
                {
                    "frame": index,
                    "saved_time_s": float(saved["time_s"][frames[index]]),
                    "saved_coefficients_sha256": hashlib.sha256(
                        saved["coefficients_m"][frames[index]].tobytes()
                    ).hexdigest(),
                    "image_sha256": hashlib.sha256(pixels).hexdigest(),
                }
            )
        assert len({r["image_sha256"] for r in records}) == 3
        assert len({r["saved_coefficients_sha256"] for r in records}) == 3
        page.get_by_role("button", name="First frame", exact=True).click()
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_function(
            'Number(document.querySelector("input[id^=_anim_slider]").value) >= 2'
        )
        page.get_by_role("button", name="Pause", exact=True).click()
        paused = slider.input_value()
        page.wait_for_timeout(300)
        assert slider.input_value() == paused
        assert not errors, errors
        browser.close()
    result = {
        "frames": records,
        "offline_play_pause_passed": True,
        "page_errors": errors,
        "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Rendering/playback verification, not physical validation",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Embedded animation browser check\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

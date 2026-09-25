"""Browser verification of the saved dual notebook and harmonic-pressure animation."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/studies/S02-independent-two-face/notebooks/04_dual_cartesian"
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        for label, path in (
            ("standalone", out / "pressure-cycle.html"),
            ("notebook", out.parent / "04_dual_cartesian.html"),
        ):
            page = browser.new_page(viewport={"width": 1300, "height": 900})
            errors = []
            page.on("pageerror", lambda e, collected=errors: collected.append(str(e)))
            # Playback must work even without remote fonts, scripts or styles.
            page.route("https://**/*", lambda route: route.abort())
            page.route("http://**/*", lambda route: route.abort())
            page.goto(path.as_uri(), wait_until="load")
            slider = page.locator(".anim-slider")
            page.wait_for_function("document.querySelector('.anim-slider').max === '48'")
            plot = page.locator(".animation img")
            plot.scroll_into_view_if_needed()
            page.get_by_role("button", name="First frame", exact=True).click()
            before = plot.screenshot(path=str(out / f"browser-{label}-frame0.png"))
            slider.fill("12")
            slider.dispatch_event("input")
            page.wait_for_function("document.querySelector('.animation img').complete")
            after = plot.screenshot(path=str(out / f"browser-{label}-frame12.png"))
            a = np.asarray(Image.open(io.BytesIO(before)).convert("RGB"), dtype=int)
            b = np.asarray(Image.open(io.BytesIO(after)).convert("RGB"), dtype=int)
            # Separately verify the title/time band and the pressure-curve region.
            changed = np.max(abs(a - b), axis=2) > 15
            title_change = float(changed[:int(.22 * len(changed))].mean())
            curve_change = float(changed[int(.25 * len(changed)):int(.85 * len(changed))].mean())
            assert title_change > .0005 and curve_change > .001
            page.get_by_role("button", name="Play", exact=True).click()
            page.wait_for_function("Number(document.querySelector('.anim-slider').value) > 14")
            page.get_by_role("button", name="Pause", exact=True).click()
            stopped = slider.input_value()
            page.wait_for_timeout(250)
            assert slider.input_value() == stopped
            assert not errors, errors
            records.append({
                "view": label, "html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "offline_playback_passed": True, "pause_passed": True,
                "title_pixel_change_fraction": title_change,
                "pressure_curve_pixel_change_fraction": curve_change,
                "javascript_errors": errors, "browser": browser.version,
                "scope": "Harmonic pressure display only; not a formation validation.",
            })
            page.close()
        browser.close()
    (out / "browser-validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

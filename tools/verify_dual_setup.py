"""Check both-face visibility, saved-state selection and camera-only animation."""

import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/notebooks/05_formation_and_spots/two-face-setup"
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        for label, path in (("standalone", out / "setup.html"),
                            ("notebook", out.parent.parent / "05_formation_and_spots.html")):
            page = browser.new_page(viewport={"width": 1200, "height": 1000})
            errors = []
            page.on("pageerror", lambda e, found=errors: found.append(str(e)))
            page.route("https://**/*", lambda r: r.abort())
            page.route("http://**/*", lambda r: r.abort())
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function("window.dualSetupState !== undefined")
            canvas = page.locator("#dual-canvas")
            initial = page.evaluate("window.dualSetupState")
            assert initial["elements"] == 104 and initial["physical_time_s"] is None
            assert initial["bottom_vertex_mm"] < initial["top_vertex_mm"]
            before = canvas.screenshot(path=str(out / f"browser-{label}-104.png"))
            page.locator("#dual-state").select_option("0")
            coarse = page.evaluate("window.dualSetupState")
            assert coarse["elements"] == 52
            assert coarse["bottom_vertex_mm"] != initial["bottom_vertex_mm"]
            assert coarse["top_vertex_mm"] != initial["top_vertex_mm"]
            page.locator("#dual-state").select_option("2")
            page.locator("#dual-rotate").click()
            page.wait_for_function("Math.abs(window.dualSetupState.angle + .85) > .15")
            page.locator("#dual-rotate").click()
            rotated = page.evaluate("window.dualSetupState")
            assert rotated["bottom_vertex_mm"] == initial["bottom_vertex_mm"]
            assert rotated["top_vertex_mm"] == initial["top_vertex_mm"]
            assert rotated["physical_time_s"] is None
            after = canvas.screenshot(path=str(out / f"browser-{label}-rotated.png"))
            assert before != after
            page.wait_for_timeout(150)
            assert page.evaluate("window.dualSetupState.angle") == rotated["angle"]
            assert not errors, errors
            results.append({"view": label, "html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "both_interfaces_selected_from_saved_states": True,
                            "camera_changes_image_not_physical_geometry": True,
                            "camera_pause_passed": True, "physical_time_axis": False,
                            "javascript_errors": errors})
            page.close()
        browser.close()
    (out / "browser-validation.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

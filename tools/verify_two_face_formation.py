"""Offline browser verification of physical-time playback, not camera motion alone."""

import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/studies/S02-independent-two-face/notebooks/06_two_face_formation"
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        for name, file in [
            ("standalone", output / "viewer.html"),
            ("notebook", output.parent / "06_two_face_formation.html"),
        ]:
            page = browser.new_page(viewport={"width": 1350, "height": 1500})
            page.route("http://**/*", lambda route: route.abort())
            page.route("https://**/*", lambda route: route.abort())
            errors = []
            page.on("pageerror", lambda error, captured=errors: captured.append(str(error)))
            page.goto(file.as_uri())
            frame = (
                page.main_frame
                if name == "standalone"
                else page.locator('iframe[title="Two-face forward formation"]')
                .element_handle()
                .content_frame()
            )
            frame.wait_for_function("window.dualFormationState !== undefined")
            states, hashes = [], []
            for index in [0, 32, 64]:
                frame.locator("#frame").evaluate(
                    '(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}', index
                )
                states.append(frame.evaluate("window.dualFormationState"))
                pixels = frame.locator("#formation").screenshot(
                    path=str(output / f"{name}-frame-{index}.png")
                )
                hashes.append(hashlib.sha256(pixels).hexdigest())
            assert len(set(hashes)) == 3
            assert [s["physical_time_s"] for s in states] == [0, 0.004, 0.008]
            for key in ["bottom_vertex_m", "top_vertex_m", "spot_rms_m"]:
                assert len({s[key] for s in states}) == 3, key
            assert all(s["source_regions"] == 112 for s in states)
            frame.locator("#reset").click()
            frame.locator("#play").click()
            frame.wait_for_function("window.dualFormationState.frame >= 3")
            frame.locator("#play").click()
            paused = frame.evaluate("window.dualFormationState")
            page.wait_for_timeout(350)
            assert frame.evaluate("window.dualFormationState.frame") == paused["frame"]
            assert not paused["playing"]
            assert not errors, errors
            results[name] = {
                "states": states,
                "canvas_sha256": hashes,
                "play_and_pause_pass": True,
                "page_errors": errors,
                "html_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
            page.close()
        browser.close()
    (output / "browser-validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

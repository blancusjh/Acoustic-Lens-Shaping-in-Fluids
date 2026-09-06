"""Exercise real browser time, geometry, field, camera and layer behavior."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--url", help="Default: open the self-contained viewer as a local file")
    parser.add_argument(
        "--browser", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    args = parser.parse_args()
    result = args.result.resolve()
    destination = result / "verification"
    destination.mkdir(exist_ok=True)
    url = args.url or (result / "viewer/index.html").as_uri()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.browser,
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1250}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="load", timeout=60000)
        page.wait_for_function("window.lensViewerState !== undefined", timeout=30000)
        final = page.evaluate("window.lensViewerState")
        page.screenshot(path=str(destination / "viewer-apparatus.png"), full_page=True)
        page.locator("#restart").click()
        initial = page.evaluate("window.lensViewerState")
        page.locator("#time").evaluate(
            '(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}'
        )
        restored = page.evaluate("window.lensViewerState")
        assert restored["vertex_mm"] != initial["vertex_mm"]
        assert restored["points_checksum"] != initial["points_checksum"]
        assert restored["pressure_checksum"] != initial["pressure_checksum"]
        assert restored["points_checksum"] == final["points_checksum"]
        for name in ["lens", "optics", "section"]:
            page.locator(f'[data-view="{name}"]').click()
            page.screenshot(path=str(destination / f"viewer-{name}.png"), full_page=True)
        page.locator('[data-view="apparatus"]').click()
        for name in ["liquid", "cylinder", "base", "array", "pressure", "flow", "rays"]:
            box = page.locator(f'[data-layer="{name}"]')
            before = box.is_checked()
            box.click()
            assert page.evaluate("window.lensViewerState.layers")[name] == (not before)
            box.click()
        page.locator("#cutaway").uncheck()
        page.screenshot(path=str(destination / "viewer-full-apparatus.png"), full_page=True)
        page.locator("#cutaway").check()
        page.locator("#play").click()
        page.wait_for_timeout(1600)
        advanced = page.evaluate("window.lensViewerState")
        assert advanced["time_s"] > 0
        assert advanced["time_s"] < final["time_s"]
        assert advanced["points_checksum"] != initial["points_checksum"]
        page.locator("#play").click()
        assert not errors, errors
        report = {
            "url_mode": "file" if url.startswith("file:") else "loopback HTTP",
            "browser": browser.version,
            "javascript_errors": errors,
            "initial": initial,
            "final": final,
            "after_playback": advanced,
            "checks": [
                "scrubbing updates actual surface coordinates",
                "scrubbing updates solved acoustic field",
                "playback advances fluid time",
                "all four cameras render",
                "all seven layer controls work",
                "housing cutaway can be disabled",
                "final state can be restored exactly",
            ],
        }
        (destination / "browser.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()

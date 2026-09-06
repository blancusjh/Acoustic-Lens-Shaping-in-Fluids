"""Check that both exported PyVista scenes load offline and respond to rotation."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--browser", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    args = parser.parse_args()
    result = args.result.resolve()
    destination = result / "verification"
    destination.mkdir(exist_ok=True)
    records = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.browser,
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        for filename in ["stationary-viewer.html", "optics-viewer.html"]:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error, collected=errors: collected.append(str(error)))
            page.goto((result / filename).as_uri(), wait_until="load", timeout=60000)
            assert "STATIONARY CANDIDATE" in page.locator("header").inner_text()
            assert "Ray RMS:" in page.locator("header").inner_text()
            canvas = page.locator("canvas").first
            canvas.wait_for(state="visible", timeout=30000)
            page.wait_for_timeout(2000)
            before = page.screenshot(path=str(destination / f"browser-{filename[:-5]}.png"))
            bounds = canvas.bounding_box()
            x, y = bounds["x"] + 0.52 * bounds["width"], bounds["y"] + 0.53 * bounds["height"]
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 180, y + 70, steps=15)
            page.mouse.up()
            page.wait_for_timeout(250)
            after = page.screenshot()
            a = np.array(Image.open(io.BytesIO(before)).convert("RGB"), dtype=int)
            b = np.array(Image.open(io.BytesIO(after)).convert("RGB"), dtype=int)
            changed = float(np.mean(np.max(abs(a - b), axis=2) > 10))
            assert changed > 0.005, f"Camera interaction did not change the scene: {changed}"
            assert not errors, errors
            records.append(
                {
                    "file": filename,
                    "html_sha256": hashlib.sha256((result / filename).read_bytes()).hexdigest(),
                    "url_mode": "offline local file",
                    "browser": browser.version,
                    "javascript_errors": errors,
                    "rotation_changed_pixel_fraction": changed,
                    "scope": "Stationary scene loading and camera interaction; no time evolution claimed.",
                }
            )
            page.close()
        browser.close()
    (destination / "browser-stationary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

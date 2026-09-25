"""Verify offline synchronized formation/spot playback in both exported HTML views."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/studies/S01-single-interface/notebooks/05_formation_and_spots"
    config = json.loads((out / "config.json").read_text())
    frame_count = len(config["frame_indices"])
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        for label, path in (
            ("standalone", out / "formation-and-spots.html"),
            ("notebook", out.parent / "05_formation_and_spots.html"),
        ):
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            errors = []
            page.on("pageerror", lambda e, collected=errors: collected.append(str(e)))
            page.route("https://**/*", lambda route: route.abort())
            page.route("http://**/*", lambda route: route.abort())
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function(
                "n => Number(document.querySelector('.anim-slider').max) === n-1",
                arg=frame_count,
            )
            slider, plot = page.locator(".anim-slider"), page.locator(".animation img")
            plot.scroll_into_view_if_needed()
            page.get_by_role("button", name="First frame", exact=True).click()
            before = plot.screenshot(path=str(out / f"browser-{label}-initial.png"))
            page.get_by_role("button", name="Last frame", exact=True).click()
            page.wait_for_function("document.querySelector('.animation img').complete")
            after = plot.screenshot(path=str(out / f"browser-{label}-final.png"))
            assert slider.input_value() == str(frame_count-1)
            assert before != after
            # At endpoints a cursor can coincide with a plot spine. Test it
            # inside the axes rather than mistaking boundary occlusion for no motion.
            slider.fill(str(frame_count//2))
            slider.dispatch_event("input")
            page.wait_for_function("document.querySelector('.animation img').complete")
            middle = plot.screenshot(path=str(out / f"browser-{label}-middle.png"))
            a = np.asarray(Image.open(io.BytesIO(before)).convert("RGB"), dtype=int)
            b = np.asarray(Image.open(io.BytesIO(middle)).convert("RGB"), dtype=int)
            changed = np.max(abs(a-b), axis=2) > 15
            h, w = changed.shape
            regions = {
                "physical_time_and_metrics_title": (0, .12, 0, 1),
                "surface_geometry": (.13, .48, 0, .33),
                "height_error_profile": (.13, .48, .34, .66),
                "full_spot_diagram": (.13, .48, .67, 1),
                "height_time_cursor": (.54, .93, 0, .33),
                "spot_time_cursor": (.54, .93, .34, .66),
                "zoomed_spot_diagram": (.54, .93, .67, 1),
            }
            fractions = {
                name: float(changed[int(y0*h):int(y1*h), int(x0*w):int(x1*w)].mean())
                for name, (y0, y1, x0, x1) in regions.items()
            }
            assert all(value > .0002 for value in fractions.values()), fractions
            page.get_by_role("button", name="First frame", exact=True).click()
            page.get_by_role("button", name="Play", exact=True).click()
            page.wait_for_function("Number(document.querySelector('.anim-slider').value) > 3")
            page.get_by_role("button", name="Pause", exact=True).click()
            stopped = slider.input_value()
            page.wait_for_timeout(300)
            assert stopped == slider.input_value()
            page.wait_for_function("window.formationAudioReady === true")
            audio = page.locator("#formation-audio")
            page.wait_for_function("document.querySelector('audio').readyState >= 2")
            assert audio.evaluate("a => a.paused")  # No automatic sound.
            duration = audio.evaluate("a => a.duration")
            assert abs(duration-17.6) < 1/48000
            audio.evaluate("a => {a.currentTime = 4.4}")
            page.wait_for_function("document.querySelector('.anim-slider').value === '40'")
            assert "50.00 ms" in page.locator("#formation-audio-status").inner_text()
            # Muting only the automated browser avoids playing into the user's room.
            audio.evaluate("async a => {a.muted=true; a.currentTime=0; await a.play()}")
            page.wait_for_function("Number(document.querySelector('.anim-slider').value) >= 3")
            audio.evaluate("a => a.pause()")
            page.wait_for_timeout(100)
            audio_stopped = slider.input_value()
            page.wait_for_timeout(300)
            assert slider.input_value() == audio_stopped
            audio.evaluate("async a => {await a.play()}")
            slider.fill("100")
            slider.dispatch_event("input")
            page.wait_for_timeout(300)
            assert audio.evaluate("a => a.paused") and slider.input_value() == "100"
            assert not errors, errors
            records.append({
                "view": label, "html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "browser": browser.version, "offline_playback_and_pause_passed": True,
                "frame_count": frame_count, "region_pixel_changes": fractions,
                "region_comparison_frames": [0, frame_count//2],
                "first_last_frame_images_differ": True,
                "audio_duration_s": duration,
                "audio_seek_play_pause_and_manual_override_passed": True,
                "audio_autoplay": False,
                "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "javascript_errors": errors,
                "scope": "Display verification; not physics or convergence validation.",
            })
            page.close()
        browser.close()
    (out / "browser-validation.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

"""Audible diagnostic mapping of a pressure-amplitude history, not a recording."""

import base64
import io
import re

import numpy as np
from scipy.io import wavfile


def pressure_sonification(time_s, peak_pressure_pa, *, slowdown=88., pitch_hz=440.,
                          sample_rate_hz=48000, peak_gain=.12, fade_s=.025):
    t, p = np.asarray(time_s, float), np.asarray(peak_pressure_pa, float)
    if (t.ndim != 1 or p.shape != t.shape or len(t) < 2 or
            not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or
            np.any(np.diff(t) <= 0) or np.any(p < 0)):
        raise ValueError("Require increasing finite times and nonnegative pressure amplitudes.")
    if (not np.all(np.isfinite([slowdown, pitch_hz, sample_rate_hz, peak_gain, fade_s])) or
            slowdown <= 0 or not 0 < pitch_hz < sample_rate_hz/2 or
            int(sample_rate_hz) != sample_rate_hz or not 0 < peak_gain <= 1 or fade_s < 0):
        raise ValueError("Invalid audio mapping parameters.")
    duration = (t[-1]-t[0])*slowdown
    if 2*fade_s > duration:
        raise ValueError("Fades must not overlap.")
    n = round(duration*sample_rate_hz)
    if n < 2:
        raise ValueError("Audio duration is too short.")
    playback_t = np.arange(n)/sample_rate_hz
    physical_t = t[0] + playback_t/slowdown
    # Only a diagnostic scalar envelope is interpolated, never a surface state.
    amplitude = np.interp(physical_t, t, p)/max(float(np.max(p)), np.finfo(float).tiny)
    fade = np.ones(n)
    if fade_s:
        fade = np.minimum(np.clip(playback_t/fade_s, 0, 1),
                          np.clip((playback_t[-1]-playback_t)/fade_s, 0, 1))
    samples = peak_gain*amplitude*fade*np.sin(2*np.pi*pitch_hz*playback_t)
    metadata = {
        "scope": "Diagnostic sonification of peak cavity-pressure amplitude; not actual apparatus sound.",
        "pitch_hz": pitch_hz, "sample_rate_hz": int(sample_rate_hz),
        "slowdown": slowdown, "physical_start_s": float(t[0]),
        "physical_end_s": float(t[-1]), "playback_duration_s": n/sample_rate_hz,
        "pressure_normalization_pa": float(np.max(p)), "peak_gain_full_scale": peak_gain,
        "fade_s": fade_s, "phase": "Artificial zero phase; acoustic phase history is unavailable.",
        "envelope": "Linear interpolation of saved spatial peak pressure amplitudes.",
        "acoustic_recording": False, "calibrated_sound_pressure_level": False,
    }
    return samples, metadata


def wav_bytes(samples, sample_rate_hz):
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate_hz), np.round(np.asarray(samples)*32767).astype(np.int16))
    return buffer.getvalue()


def synchronized_audio_html(animation_html, audio_bytes, playback_step_s):
    """Audio is the master clock; the original animation controls remain silent."""
    match = re.search(r'(anim[0-9a-f]+)\.set_frame', animation_html)
    if match is None or not np.isfinite(playback_step_s) or playback_step_s <= 0:
        raise ValueError("Missing animation binding or invalid playback step.")
    name = match[1]
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f'''
<section id="formation-sonification">
<h2>Audible diagnostic sonification — not a recording</h2>
<p>The modeled 1 MHz carrier is not audible. This artificial 440 Hz tone follows
the saved peak cavity-pressure amplitude, with time slowed 88× and normalized
playback volume. It is not the sound radiated into air, a microphone signal,
or a calibrated sound-pressure level. No acoustic phase history was saved.</p>
<p><b>Use this audio player's Play button for sound-synchronized formation.</b>
The animation's own controls stop the audio and operate silently. Start with
low playback volume. The diagram remains a time-step-sensitive numerical trajectory.</p>
<audio id="formation-audio" controls preload="metadata" style="width:90%">
<source src="data:audio/wav;base64,{encoded}" type="audio/wav"></audio>
<p id="formation-audio-status">Loading synchronization…</p>
</section>
<script>
(function bindAudio(attempt) {{
  const audio = document.getElementById('formation-audio');
  const status = document.getElementById('formation-audio-status');
  const anim = window['{name}'];
  if (!anim) {{
    if (attempt < 100) setTimeout(() => bindAudio(attempt+1), 100);
    else status.textContent = 'Animation unavailable; audio remains a diagnostic only.';
    return;
  }}
  let raf = null;
  let manualControl = false;
  function update() {{
    const k = Math.min(anim.frames.length-1, Math.floor(audio.currentTime/{playback_step_s!r}+1e-7));
    anim.set_frame(k);
    status.textContent = 'Audio synchronized • displayed physical t = ' +
      (k*{playback_step_s!r}/88*1000).toFixed(2) + ' ms • artificial 440 Hz tone';
  }}
  function tick() {{
    if (audio.paused || audio.ended) return;
    update();
    if (!audio.paused && !audio.ended) raf = requestAnimationFrame(tick);
  }}
  audio.addEventListener('play', () => {{ manualControl = false; anim.pause_animation(); tick(); }});
  audio.addEventListener('pause', () => {{ cancelAnimationFrame(raf); if (!manualControl) update(); }});
  audio.addEventListener('seeking', () => {{ manualControl = false; anim.pause_animation(); update(); }});
  audio.addEventListener('ended', () => {{ anim.last_frame(); update(); }});
  function stopForManualControl() {{
    manualControl = true; cancelAnimationFrame(raf); audio.pause();
  }}
  const slider = document.getElementById(anim.slider_id);
  slider.addEventListener('input', stopForManualControl, true);
  document.getElementById(anim.img_id).parentElement.querySelectorAll('button')
    .forEach(b => b.addEventListener('click', stopForManualControl, true));
  status.textContent = 'Ready: audio player drives the animation. No automatic sound.';
  window.formationAudioReady = true;
}})(0);
</script>'''

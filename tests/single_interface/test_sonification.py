import io

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_freeform.single_interface.sonification import pressure_sonification, wav_bytes


def test_constant_envelope_has_declared_pitch_and_duration():
    samples, meta = pressure_sonification([0, .2], [1e6, 1e6], slowdown=10, fade_s=0)
    frequency = np.fft.rfftfreq(len(samples), 1/48000)
    assert frequency[np.argmax(abs(np.fft.rfft(samples)))] == 440
    assert len(samples) == 96000
    assert np.max(abs(samples)) <= .12+1e-14
    rate, decoded = wavfile.read(io.BytesIO(wav_bytes(samples, 48000)))
    assert rate == 48000 and decoded.dtype == np.int16
    assert meta["acoustic_recording"] is False


def test_envelope_mapping_and_fades():
    samples, _ = pressure_sonification([0, .1, .2], [0, 1, 0], slowdown=10)
    rms = [np.sqrt(np.mean(x*x)) for x in np.array_split(samples, 4)]
    assert rms[1] > rms[0] and rms[2] > rms[3]
    assert samples[0] == samples[-1] == 0
    zero, _ = pressure_sonification([0, .2], [0, 0])
    assert np.all(zero == 0)
    with pytest.raises(ValueError):
        pressure_sonification([0, 0], [1, 2])
    with pytest.raises(ValueError):
        pressure_sonification([0, .2], [1, -2])

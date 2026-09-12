import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.corruptions.drift import add_baseline_drift


class BaselineDriftTest(unittest.TestCase):
    def _power_at(self, x, freq, fs=100.0):
        spectrum = torch.fft.rfft(x, dim=-1).abs().square()
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
        idx = torch.argmin((freqs - freq).abs())
        return spectrum[..., idx].mean().item()

    def test_post_bandpass_removes_sub_band_drift(self):
        x = torch.zeros(2, 1, 3000)
        f = torch.full((2,), 0.05)
        amp = torch.ones(2)
        raw = add_baseline_drift(x, f=f, amplitude=amp, post_bandpass=False)
        filtered = add_baseline_drift(x, f=f, amplitude=amp, post_bandpass=True)

        self.assertLess(self._power_at(filtered, 0.05), self._power_at(raw, 0.05) * 1e-4)

    def test_post_bandpass_keeps_in_band_drift(self):
        x = torch.zeros(2, 1, 3000)
        f = torch.full((2,), 0.5)
        amp = torch.ones(2)
        raw = add_baseline_drift(x, f=f, amplitude=amp, post_bandpass=False)
        filtered = add_baseline_drift(x, f=f, amplitude=amp, post_bandpass=True)

        self.assertGreater(self._power_at(filtered, 0.5), self._power_at(raw, 0.5) * 0.5)


if __name__ == "__main__":
    unittest.main()

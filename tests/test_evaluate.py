"""The metrics are only worth reporting if they're right."""

from __future__ import annotations

import unittest

import numpy as np

from afterimage.evaluate import delta_e2000, masked_median, psnr, srgb_to_lab, temporal_median


class ColourTest(unittest.TestCase):
    def test_ciede2000_reference_pairs(self) -> None:
        # From Sharma, Wu & Dalal (2005), Table 1.
        pairs = [
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
            ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
            ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
            ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
            ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.3350), 1.0000),
        ]
        for lab1, lab2, expected in pairs:
            self.assertAlmostEqual(float(delta_e2000(np.array(lab1), np.array(lab2))), expected, places=3)

    def test_lab_of_white_and_black(self) -> None:
        np.testing.assert_allclose(srgb_to_lab(np.array([255, 255, 255])), [100, 0, 0], atol=0.05)
        np.testing.assert_allclose(srgb_to_lab(np.array([0, 0, 0])), [0, 0, 0], atol=0.05)

    def test_psnr(self) -> None:
        image = np.full((4, 4, 3), 100, np.uint8)
        self.assertEqual(psnr(image, image), float("inf"))
        self.assertAlmostEqual(psnr(image, image + 1), 48.13, places=2)


class BaselineTest(unittest.TestCase):
    def test_temporal_median_ignores_a_brief_visitor(self) -> None:
        frames = [np.full((4, 4, 3), 200, np.uint8) for _ in range(5)]
        frames[2] = np.zeros((4, 4, 3), np.uint8)
        outputs = list(temporal_median(frames, 5))
        self.assertTrue((outputs[-1] == 200).all())

    def test_masked_median_uses_only_uncovered_samples(self) -> None:
        frames = [np.full((4, 4, 3), 200, np.uint8) for _ in range(4)]
        frames[0][:2] = 10
        masks = [np.zeros((4, 4), bool) for _ in range(4)]
        masks[0][:2] = True
        outputs = list(masked_median(frames, masks, segment=4))
        self.assertTrue((outputs[0] == 200).all())


if __name__ == "__main__":
    unittest.main()

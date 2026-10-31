"""The clean plate should end up showing all the paint and none of the painter."""

from __future__ import annotations

import unittest

import numpy as np

from afterimage.cleanplate import BLIND_FRAMES, CleanPlate, CleanPlateConfig, first_visible_canvas, visible_mask
from tests.synthetic import PALETTE, SKIN, shadow_session, synthetic_session


def _error(canvas: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(canvas.astype(np.int16) - truth.astype(np.int16)).max(axis=2)


class CleanPlateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = synthetic_session()
        self.config = CleanPlateConfig(dilate_px=6)

    def run_plate(self) -> list[np.ndarray]:
        start = first_visible_canvas(
            (frame for frame, _, _ in self.session),
            (mask for _, mask, _ in self.session),
            dilate_px=self.config.dilate_px,
        )
        plate = CleanPlate(start, self.config)
        return [plate.update(frame, mask) for frame, mask, _ in self.session]

    def start_canvas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The start canvas, the true canvas at frame 0, and what frame 0 shows uncovered."""
        start = first_visible_canvas(
            (frame for frame, _, _ in self.session),
            (mask for _, mask, _ in self.session),
            dilate_px=6, steady_frames=4,
        )
        first_mask = self.session[0][1]
        return start, self.session[0][2], visible_mask(first_mask, first_mask.shape, 6)

    def test_start_skips_the_shadow_in_the_first_frame(self) -> None:
        start, truth, seen = self.start_canvas()
        shadowed = seen & (_error(self.session[0][0], truth) > 25)
        self.assertGreater(shadowed.sum(), 1000)
        self.assertLess((_error(start, truth)[shadowed] > 25).mean(), 0.02)
        # Matched to frame 0's exposure, though the frames after it jitter by up to 7%.
        self.assertLess(_error(start, truth)[seen].mean(), 0.5)

    def test_paint_from_the_first_moments_still_arrives_as_new_paint(self) -> None:
        start, truth, seen = self.start_canvas()
        early = seen & (_error(self.session[6][2], truth) > 25)
        self.assertGreater(early.sum(), 500)
        self.assertLess((_error(start, truth)[early] > 25).mean(), 0.02)

    def test_final_canvas_matches_the_painting(self) -> None:
        final = self.run_plate()[-1]
        error = _error(final, self.session[-1][2])
        self.assertLess(error.mean(), 3.0)
        self.assertLess((error > 25).mean(), 0.01)

    def test_painter_never_shows_up(self) -> None:
        skin = np.array(SKIN, dtype=np.int16)
        for index, canvas in enumerate(self.run_plate()[5:], start=5):
            close_to_skin = np.abs(canvas.astype(np.int16) - skin).max(axis=2) < 12
            self.assertLess(close_to_skin.mean(), 0.002, f"skin visible in frame {index}")

    def test_beats_the_raw_footage(self) -> None:
        outputs = self.run_plate()
        raw_errors, clean_errors = [], []
        for (frame, _, truth), canvas in zip(self.session[10:-8], outputs[10:-8]):
            raw_errors.append((_error(frame, truth) > 25).mean())
            clean_errors.append((_error(canvas, truth) > 25).mean())
        self.assertLess(np.mean(clean_errors), np.mean(raw_errors) / 3)


class ShadowedExposureTest(unittest.TestCase):
    """A shadow over most of the canvas must not read as a darker exposure.

    If it did, the uncovered part would be brightened to compensate, committed as brighter
    paint, and the plate would never find its way back: measured against itself, the brighter
    plate is the new normal. On real footage this ran away by 80 levels over an hour.
    """

    def test_a_shadow_over_most_of_the_canvas_does_not_brighten_the_rest(self) -> None:
        session = shadow_session()
        plate = CleanPlate(session[0][2], CleanPlateConfig(dilate_px=6))
        brightened = []
        for frame, mask, truth, shadowed in session:
            canvas = plate.update(frame, mask)
            lit = ~shadowed & visible_mask(mask, mask.shape, 6)
            brightened.append(float((canvas.astype(np.int16) - truth.astype(np.int16))[lit].mean()))
        self.assertLess(max(brightened), 4.0, "the uncovered canvas was brightened to make up for the shadow")
        self.assertLess(_error(plate.canvas, session[-1][2]).mean(), 3.0)
        self.assertAlmostEqual(float(plate.gain.mean()), 1.0, delta=0.12)

    def test_exposure_is_not_measured_on_a_sliver_of_canvas(self) -> None:
        session = synthetic_session()
        plate = CleanPlate(session[0][2], CleanPlateConfig(dilate_px=6))
        plate.update(session[0][0], session[0][1])
        before = plate.gain.copy()
        shadowed = np.clip(session[1][2].astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
        covered = np.zeros(shadowed.shape[:2], dtype=bool)
        covered[:, :int(shadowed.shape[1] * 0.8)] = True
        plate.update(shadowed, covered)
        np.testing.assert_allclose(plate.gain, before)


class BlindFramesTest(unittest.TestCase):
    """A frame that is no view of the canvas changes nothing; a scene that really changed gets through."""

    def test_someone_walking_in_front_of_the_lens_is_not_committed(self) -> None:
        session = synthetic_session()
        plate = CleanPlate(session[0][2], CleanPlateConfig(dilate_px=6))
        shirt = np.full_like(session[0][0], (180, 60, 40))
        nothing_masked = np.zeros(session[0][1].shape, dtype=bool)
        for index, (frame, mask, _) in enumerate(session):
            if 30 <= index < 42:  # far longer than the persistence, and SAM saw nothing
                plate.update(shirt, nothing_masked)
                self.assertGreater(plate.disagreement, 0.5)
            else:
                plate.update(frame, mask)
        error = _error(plate.canvas, session[-1][2])
        self.assertLess((error > 25).mean(), 0.01)
        self.assertAlmostEqual(float(plate.gain.mean()), 1.0, delta=0.1)

    def test_a_scene_that_really_changed_is_accepted_after_the_blind_frames(self) -> None:
        session = synthetic_session(frames=20)
        plate = CleanPlate(session[0][2], CleanPlateConfig(dilate_px=6))
        changed = session[-1][2].copy()
        changed[:, :int(changed.shape[1] * 0.7)] = PALETTE[0]
        for _ in range(BLIND_FRAMES + plate.config.persist_frames + 1):
            plate.update(changed, None)
        self.assertLess(_error(plate.canvas, changed).mean(), 3.0)


class RefinedRecordTest(unittest.TestCase):
    """A stroke committed from a noisy first impression ends up on record with the colour the
    camera keeps showing afterwards, as one change, not the first frame's value for good."""

    def test_the_record_follows_the_settled_colour(self) -> None:
        from afterimage.cleanplate import CleanPlate, CleanPlateConfig

        size = (40, 30)
        white = np.full((size[1], size[0], 3), 235, dtype=np.uint8)
        true = np.array([40, 60, 200], dtype=np.uint8)
        noisy = np.array([55, 75, 214], dtype=np.uint8)  # 15 levels off, within the change threshold
        plate = CleanPlate(white, CleanPlateConfig(persist_frames=3, record=True, normalize_exposure=False, dilate_px=0))
        rng = np.random.default_rng(0)
        for update in range(70):
            frame = white.copy()
            if update >= 5:
                colour = noisy if update < 8 else true
                frame[10:20, 10:25] = colour
                frame = np.clip(frame.astype(np.int16) + rng.integers(-2, 3, size=frame.shape), 0, 255).astype(np.uint8)
            plate.update(frame, None)
        timeline = plate.timeline()
        stroke = timeline.pixel == 15 * size[0] + 17
        self.assertEqual(int(stroke.sum()), 1, "one change on record for the stroke")
        self.assertEqual(int(timeline.frame[stroke][0]), 5, "dated to when it was first seen")
        self.assertLessEqual(int(np.abs(timeline.after[stroke][0].astype(int) - true.astype(int)).max()), 4,
                             f"the recorded colour follows the camera: {timeline.after[stroke][0]}")
        self.assertLessEqual(int(np.abs(plate.canvas[15, 17].astype(int) - true.astype(int)).max()), 4)



class SurroundExposureTest(unittest.TestCase):
    """Light is measured on the room around the canvas: paint over the whole canvas stays paint,
    and a change of light after the canvas is covered is still a change of light."""

    @staticmethod
    def scene() -> tuple[list[np.ndarray], np.ndarray, dict[int, np.ndarray]]:
        height, width = 120, 160
        canvas = np.zeros((height, width), dtype=bool)
        canvas[20:100, 40:120] = True
        person = np.zeros((height, width), dtype=bool)
        person[60:120, 128:160] = True
        frames, masks = [], {}
        for index in range(45):
            frame = np.empty((height, width, 3), dtype=np.float32)
            frame[:] = (180, 170, 150)  # the wall
            frame[canvas] = (230, 230, 230)
            painted = min(max(index - 9, 0), 10) * 8  # a pink wash, eight columns a frame from frame 10
            frame[20:100, 40:40 + painted] = (220, 160, 170)
            if 5 <= index < 15:
                frame[person] = (40, 50, 60)
                masks[index] = person
            if index >= 30:
                frame *= 1.2  # then the phone brightens everything
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        return frames, canvas, masks

    def test_gains_follow_the_light_not_the_paint(self) -> None:
        from afterimage.cleanplate import surround_gains

        frames, canvas, masks = self.scene()
        gains = np.array(surround_gains(frames, canvas, masks.get, smooth=1))
        np.testing.assert_allclose(gains[:30], 1.0, atol=0.02, err_msg="the wash is not a change of light")
        np.testing.assert_allclose(gains[31:], 1.2, atol=0.03, err_msg="the brighter exposure is")

    def test_paint_stays_paint_and_light_stays_light(self) -> None:
        from afterimage.cleanplate import CleanPlate, CleanPlateConfig, surround_gains

        frames, canvas, masks = self.scene()
        views = [frame[20:100, 40:120] for frame in frames]
        gains = surround_gains(frames, canvas, masks.get, smooth=1)
        config = CleanPlateConfig(dilate_px=0, persist_frames=3, record=True)
        plate = CleanPlate(views[0], config)
        for index, view in enumerate(views):
            plate.update(view, None, gains[index])
        pink = np.array((220, 160, 170))
        self.assertLess(int(np.abs(plate.canvas[40, 40].astype(int) - pink).max()), 6,
                        "the wash is kept as paint and the brighter light is undone")
        self.assertEqual(int(plate.gain[1] * 100 + 0.5), 120, "the plate used the measured gain")


if __name__ == "__main__":
    unittest.main()

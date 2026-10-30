"""A resting hand's shadow holds still long enough to be committed; the timeline must veto it."""

from __future__ import annotations

import unittest

import numpy as np

from afterimage.cleanplate import CleanPlate, CleanPlateConfig
from afterimage.timeline import Timeline
from tests.synthetic import resting_hand_session, synthetic_session

REST = (20, 50)


def shadow_ghost_frames(canvases: list[np.ndarray], session) -> int:
    """Frames where a noticeable patch of the reconstruction is darker than the real canvas."""
    count = 0
    for canvas, (_, _, truth, _) in zip(canvases, session):
        darker = (truth.astype(np.int16) - canvas.astype(np.int16)).min(axis=2) > 25
        if darker.mean() > 0.002:
            count += 1
    return count


class TimelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = resting_hand_session(rest=REST)
        self.plate = CleanPlate(self.session[0][2], CleanPlateConfig(dilate_px=4, persist_frames=3, record=True))
        self.forward = [self.plate.update(frame, mask).copy() for frame, mask, _, _ in self.session]

    def test_forward_pass_bakes_in_the_resting_shadow(self) -> None:
        # Documents the failure the timeline exists to fix.
        self.assertGreater(shadow_ghost_frames(self.forward, self.session), 10)

    def test_timeline_vetoes_the_shadow(self) -> None:
        timeline = self.plate.timeline()
        dropped = timeline.transients(threshold=20, window=60)
        self.assertGreater(dropped.sum(), 0)
        replayed = [canvas.copy() for canvas, _ in timeline.without(dropped).replay(self.session[0][2], len(self.session))]
        self.assertEqual(shadow_ghost_frames(replayed, self.session), 0)

    def test_replay_matches_forward_when_nothing_is_dropped(self) -> None:
        timeline = self.plate.timeline()
        replayed = [canvas.copy() for canvas, _ in timeline.replay(self.session[0][2], len(self.session))]
        difference = np.abs(replayed[-1].astype(np.int16) - self.forward[-1].astype(np.int16)).max(axis=2)
        self.assertLess(np.percentile(difference, 99), 12)

    def test_real_paint_is_never_vetoed(self) -> None:
        session = synthetic_session(frames=90)
        plate = CleanPlate(session[0][0], CleanPlateConfig(dilate_px=6, record=True))
        for frame, mask, _ in session:
            plate.update(frame, mask)
        timeline = plate.timeline()
        kept = timeline.without(timeline.transients(threshold=20, window=60))
        final = [canvas.copy() for canvas, _ in kept.replay(session[0][0], len(session))][-1]
        error = np.abs(final.astype(np.int16) - session[-1][2].astype(np.int16)).max(axis=2)
        self.assertLess((error > 25).mean(), 0.01)

    def test_a_shadow_over_revealed_canvas_keeps_the_reveal(self) -> None:
        # Pixel starts shadowed (D), is revealed (W), a shadow passes (W->D->W), then paint (P).
        dark, white, paint = (150, 150, 150), (228, 228, 228), (30, 90, 160)
        timeline = Timeline(
            (1, 1),
            np.zeros(4, np.int32),
            np.array([2, 10, 14, 30], np.int32),
            np.array([dark, white, dark, white], np.uint8),
            np.array([white, dark, white, paint], np.uint8),
        )
        dropped = timeline.transients(threshold=20, window=60)
        np.testing.assert_array_equal(dropped, [False, True, True, False])

    def test_paint_is_dated_to_when_it_was_first_seen(self) -> None:
        white, blue = (228, 228, 228), (30, 90, 160)
        frames = [np.full((40, 40, 3), white, np.uint8) for _ in range(12)]
        for frame in frames[3:]:
            frame[10:20, 10:20] = blue
        hand = np.zeros((40, 40), dtype=bool)
        hand[5:25, 5:25] = True
        plate = CleanPlate(frames[0], CleanPlateConfig(dilate_px=2, persist_frames=3, record=True))
        forward = [plate.update(frame, hand if 4 <= i <= 6 else None).copy() for i, frame in enumerate(frames)]
        replayed = [canvas.copy() for canvas, _ in plate.timeline().replay(frames[0], len(frames))]

        def first_blue(canvases: list[np.ndarray]) -> int:
            return next(i for i, canvas in enumerate(canvases) if (canvas[15, 15] == blue).all())

        self.assertEqual(first_blue(forward), 8)  # confirmed once seen 3 times: frames 3, 7 and 8
        self.assertEqual(first_blue(replayed), 3)
        self.assertEqual(plate.painted_at[15, 15], 3)

    def test_peel_shows_what_was_underneath(self) -> None:
        # Pixel 0 went bare -> A -> B -> C; pixel 3 went bare -> D; pixels 1 and 2 never changed.
        bare, a, b, c, d = (228, 228, 228), (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40)
        timeline = Timeline(
            (2, 2),
            np.array([0, 3, 0, 0], np.int32),
            np.array([1, 2, 5, 9], np.int32),
            np.array([bare, bare, a, b], np.uint8),
            np.array([a, d, b, c], np.uint8),
        )
        initial = np.full((2, 2, 3), bare, np.uint8)
        final = np.array([[c, bare], [bare, d]], np.uint8)
        np.testing.assert_array_equal(timeline.depth(), [[3, 0], [0, 1]])
        layers = [canvas.copy() for canvas in timeline.peel(initial, final, 4)]
        np.testing.assert_array_equal(layers[0], [[b, bare], [bare, bare]])
        np.testing.assert_array_equal(layers[1], [[a, bare], [bare, bare]])
        np.testing.assert_array_equal(layers[2], initial)
        np.testing.assert_array_equal(layers[3], initial)

    def test_peel_follows_the_kept_changes_when_one_was_dropped(self) -> None:
        bare, a, b, c = (228, 228, 228), (10, 10, 10), (20, 20, 20), (30, 30, 30)
        timeline = Timeline(
            (1, 1), np.zeros(3, np.int32), np.array([1, 5, 9], np.int32),
            np.array([bare, a, b], np.uint8), np.array([a, b, c], np.uint8),
        )
        kept = timeline.without(np.array([False, True, False]))
        initial, final = np.full((1, 1, 3), bare, np.uint8), np.full((1, 1, 3), c, np.uint8)
        layers = [canvas.copy() for canvas in kept.peel(initial, final, 2)]
        # The dropped change (to B) never shows: peeling C reveals A, not B.
        np.testing.assert_array_equal(layers[0][0, 0], a)
        np.testing.assert_array_equal(layers[1][0, 0], bare)

    def test_painted_at(self) -> None:
        timeline = Timeline(
            (2, 2), np.array([0, 3, 0], np.int32), np.array([1, 2, 5], np.int32),
            np.zeros((3, 3), np.uint8), np.full((3, 3), 9, np.uint8),
        )
        np.testing.assert_array_equal(timeline.painted_at(), [[5, -1], [-1, 2]])


if __name__ == "__main__":
    unittest.main()

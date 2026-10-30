"""The API path's decoded masks are kept beside their event stream and reused."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from afterimage import pipeline, sam
from afterimage.masks import Blob


class DecodedCacheTest(unittest.TestCase):
    def test_decoding_happens_once_per_stream(self) -> None:
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, 2:7] = True
        decoded = mock.Mock(return_value={3: {"hand:1": Blob(10, 20, mask)}})
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sam, "frame_objects", decoded):
            events = Path(folder) / "hand.events.ndjson"
            events.write_text("{}\n", encoding="utf-8")
            first = pipeline._decoded(events, object(), prefix="hand:")
            second = pipeline._decoded(events, object(), prefix="hand:")
            self.assertEqual(decoded.call_count, 1)
            self.assertTrue((Path(folder) / "hand.events.ndjson.masks.npz").exists())
            self.assertEqual(list(second), [3])
            np.testing.assert_array_equal(second[3]["hand:1"].mask, first[3]["hand:1"].mask)
            self.assertEqual((second[3]["hand:1"].left, second[3]["hand:1"].top), (10, 20))
            # A changed stream, or another concept's prefix, is decoded afresh.
            events.write_text("{}\n{}\n", encoding="utf-8")
            pipeline._decoded(events, object(), prefix="hand:")
            self.assertEqual(decoded.call_count, 2)
            pipeline._decoded(events, object(), prefix="person:")
            self.assertEqual(decoded.call_count, 3)


if __name__ == "__main__":
    unittest.main()

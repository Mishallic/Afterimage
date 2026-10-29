"""Masks go to disk and come back identical, in a byte-stable file."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from nohands import maskfile
from nohands.masks import Blob
from tests.synthetic import painted_hand_session


class MaskFileTest(unittest.TestCase):
    def test_round_trip_and_stable_bytes(self) -> None:
        session, _ = painted_hand_session(frames=12)
        objects = {index: dict(blobs) for index, (_, blobs, _) in enumerate(session) if blobs}
        data = maskfile.pack(objects, {"model": "facebook/sam3.1", "prompts": ["hand"]})
        again = maskfile.pack(dict(reversed(list(objects.items()))), {"prompts": ["hand"], "model": "facebook/sam3.1"})
        self.assertEqual(hashlib.sha256(data).hexdigest(), hashlib.sha256(again).hexdigest())

        loaded, meta = maskfile.unpack(data)
        self.assertEqual(meta, {"model": "facebook/sam3.1", "prompts": ["hand"]})
        self.assertEqual(sorted(loaded), sorted(objects))
        for index, blobs in objects.items():
            self.assertEqual(sorted(loaded[index]), sorted(blobs))
            for key, blob in blobs.items():
                back = loaded[index][key]
                self.assertEqual((back.left, back.top), (blob.left, blob.top))
                np.testing.assert_array_equal(back.mask, blob.mask)

    def test_from_full_frame_arrays_drops_empty_masks(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:9, 10:14] = True
        objects = maskfile.from_arrays({3: {"hand:1": mask, "hand:2": np.zeros((20, 30), bool)}})
        self.assertEqual(list(objects[3]), ["hand:1"])
        blob, expected = objects[3]["hand:1"], Blob.of(mask)
        self.assertEqual((blob.left, blob.top), (expected.left, expected.top))
        np.testing.assert_array_equal(blob.mask, expected.mask)

    def test_empty(self) -> None:
        loaded, meta = maskfile.unpack(maskfile.pack({}))
        self.assertEqual((loaded, meta), ({}, {}))


if __name__ == "__main__":
    unittest.main()

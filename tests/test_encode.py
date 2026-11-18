"""Check manifest identity and patient pooling at the study storage precision."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cryopair_encode import patient_features, read_manifest


class EncodeTests(unittest.TestCase):
    def test_multi_slide_pooling(self):
        arrays = {"a": [[1.0004, 2], [4, 3]], "b": [[12, 13]]}
        core = SimpleNamespace(DIM=2, MIN_TILES=1, sha256_file=lambda p: p.name,
                               plan_coords=lambda p, key: (list(range(len(arrays[key]))), 0.5, "measured", 3),
                               read_tiles=lambda p, coords, mpp: arrays[p.name],
                               encode_tiles=lambda encoder, tiles, wait: np.asarray(tiles, np.float32))
        pooled = patient_features([Path("a"), Path("b")], None, core)
        np.testing.assert_array_equal(pooled, np.array([17, 18], dtype=np.float32) / 3)

    def test_manifest_and_duplicate_slide(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / "slides.csv"
            for name in ("a", "b"):
                (folder / name).touch()
            path.write_text("patient,slide\nNA,b\nNA,a\n", encoding="utf-8")
            self.assertEqual(read_manifest(path), {"NA": [folder / "b", folder / "a"]})
            with path.open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["other", "b"])
            with self.assertRaisesRegex(ValueError, "repeated slide"):
                read_manifest(path)

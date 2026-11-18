"""Fit a reader and a map from synthetic patient features using the installed command."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


class FittingTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.command = shutil.which("cryopair-fit")
        self.assertIsNotNone(self.command, "Install the package before running the tests.")
        rng = np.random.default_rng(137)
        frozen = rng.normal(size=(132, 6)).astype(np.float32)
        fixed = frozen @ rng.normal(size=(6, 6)).astype(np.float32) + 1
        expression = fixed @ rng.normal(size=(6, 3)).astype(np.float32) + 4
        self.test_frozen, self.test_fixed, self.test_expression = frozen[120:], fixed[120:], expression[120:]
        patients = np.array([f"example-{i:03d}" for i in range(120)])
        self.genes = np.array(["example_A", "example_B", "example_C"])
        np.savez(self.folder / "frozen.npz", patient=patients, features=frozen[:120])
        np.savez(self.folder / "fixed.npz", patient=patients[::-1], features=fixed[:120][::-1])
        np.savez(self.folder / "rna.npz", patient=patients, gene=self.genes, values=expression[:120])

    def fit(self, *args, check=True):
        return subprocess.run([self.command, *args], cwd=self.folder, text=True, capture_output=True, check=check)

    def test_reader(self):
        self.fit("reader", "--features", "fixed.npz", "--targets", "rna.npz", "--space", "synthetic",
                 "--units", "example units", "--out", "reader.npz")
        with np.load(self.folder / "reader.npz", allow_pickle=False) as reader:
            self.assertEqual(json.loads(str(reader["meta"]))["target_units"], "example units")
            np.testing.assert_array_equal(reader["gene"], self.genes)
            prediction = self.test_fixed @ reader["W"] + reader["b"]
        np.testing.assert_allclose(prediction, self.test_expression, atol=0.05, rtol=0)

    def test_pairmap(self):
        self.fit("pairmap", "--frozen", "frozen.npz", "--ffpe", "fixed.npz",
                 "--backbone", "synthetic", "--out", "map.npz")
        with np.load(self.folder / "map.npz", allow_pickle=False) as mapping:
            self.assertEqual(json.loads(str(mapping["meta"]))["kind"], "PAIRMAP")
            prediction = self.test_frozen @ mapping["W"] + mapping["b"]
        np.testing.assert_allclose(prediction, self.test_fixed, atol=0.03, rtol=0)

    def test_unmatched_patients(self):
        with np.load(self.folder / "fixed.npz", allow_pickle=False) as fixed:
            np.savez(self.folder / "subset.npz", patient=fixed["patient"][1:], features=fixed["features"][1:])
        result = self.fit("pairmap", "--frozen", "frozen.npz", "--ffpe", "subset.npz",
                          "--backbone", "synthetic", "--out", "map.npz", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("different patients", result.stderr)
        self.assertFalse((self.folder / "map.npz").exists())


if __name__ == "__main__":
    unittest.main()

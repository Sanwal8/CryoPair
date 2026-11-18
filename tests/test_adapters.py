"""Check which adapter files the prediction command accepts, using adapters of the right shape filled with random numbers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import cryopair_predict as predict
    import cryopair_train as train
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and pandas are needed for the adapter tests")
class AdapterFileTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        rng = np.random.default_rng(137)
        width = predict.DIM // 2
        self.state = {}
        for layer in range(64):
            out = 3 * width if layer % 2 == 0 else width
            self.state[f"block{layer}.A"] = torch.from_numpy(rng.normal(size=(predict.LORA_RANK, width)).astype(np.float32))
            self.state[f"block{layer}.B"] = torch.from_numpy(rng.normal(size=(out, predict.LORA_RANK)).astype(np.float32))

    def save(self, name, **facts):
        path = self.folder / name
        identity = train.save_adapters(path, self.state, **facts)
        return path, identity

    def test_finished_paired_run_is_read(self):
        path, identity = self.save("paired.npz", pairing="paired", complete=True, steps=3672, total_steps=3672)
        state, layers, read_identity = predict.load_adapters(path)
        self.assertEqual((layers, read_identity), (64, identity))
        self.assertTrue(torch.equal(state["block0.A"], self.state["block0.A"]))

    def test_file_without_run_facts_is_read(self):
        path, identity = self.save("earlier.npz")
        self.assertEqual(predict.load_adapters(path)[2], identity)

    def test_shuffled_control_is_refused(self):
        path, _ = self.save("shuffled.npz", pairing="shuffled", complete=True, steps=3672, total_steps=3672)
        with self.assertRaisesRegex(predict.Problem, "shuffled"):
            predict.load_adapters(path)

    def test_unfinished_run_is_refused(self):
        path, _ = self.save("trial.npz", pairing="paired", complete=False, steps=20, total_steps=3672)
        with self.assertRaisesRegex(predict.Problem, "20 of 3672 steps"):
            predict.load_adapters(path)


if __name__ == "__main__":
    unittest.main()

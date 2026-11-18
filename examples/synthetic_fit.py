"""Fit a reader and a PAIRMAP on made-up patient features and apply them to held-out patients; runs on a CPU in a few seconds."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

rng = np.random.default_rng(7)
n_train, n_test, n_features, n_genes = 200, 40, 16, 5

# Made-up data: FFPE features are a linear function of frozen features, and expression is a linear function of FFPE features.
frozen = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
ffpe = frozen @ rng.normal(size=(n_features, n_features)).astype(np.float32) + 1
expression = ffpe @ rng.normal(size=(n_features, n_genes)).astype(np.float32) + 4
patients = np.array([f"patient-{i:03d}" for i in range(n_train)])
genes = np.array([f"gene_{i}" for i in range(n_genes)])

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    np.savez(folder / "frozen.npz", patient=patients, features=frozen[:n_train])
    np.savez(folder / "ffpe.npz", patient=patients, features=ffpe[:n_train])
    np.savez(folder / "rna.npz", patient=patients, gene=genes, values=expression[:n_train])

    fit = [sys.executable, "-m", "cryopair_fit"]
    subprocess.run(fit + ["reader", "--features", str(folder / "ffpe.npz"), "--targets", str(folder / "rna.npz"),
                          "--space", "synthetic", "--units", "made-up units", "--out", str(folder / "reader.npz")], check=True)
    subprocess.run(fit + ["pairmap", "--frozen", str(folder / "frozen.npz"), "--ffpe", str(folder / "ffpe.npz"),
                          "--backbone", "synthetic", "--out", str(folder / "map.npz")], check=True)

    with np.load(folder / "reader.npz") as reader, np.load(folder / "map.npz") as pairmap:
        held_out = frozen[n_train:]
        direct = held_out @ reader["W"] + reader["b"]
        mapped = (held_out @ pairmap["W"] + pairmap["b"]) @ reader["W"] + reader["b"]

truth = expression[n_train:]
print(f"held-out patients: {n_test}, genes: {n_genes}")
print(f"mean absolute error, FFPE reader applied to frozen features directly: {np.abs(direct - truth).mean():.3f}")
print(f"mean absolute error, frozen features passed through the PAIRMAP first: {np.abs(mapped - truth).mean():.3f}")

"""Extract original CRISP patient features from a cohort of frozen or FFPE slides."""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_manifest(path):
    patients = {}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["patient", "slide"]:
            raise ValueError("the manifest must have the header patient,slide")
        for row in rows:
            patient = (row["patient"] or "").strip()
            name = (row["slide"] or "").strip()
            if not patient or not name or None in row:
                raise ValueError(f"invalid manifest row {rows.line_num}")
            slide = (path.parent / name).resolve()
            if not slide.is_file() or slide in seen:
                raise ValueError(f"missing or repeated slide at row {rows.line_num}: {name}")
            patients.setdefault(patient, []).append(slide)
            seen.add(slide)
    if not patients:
        raise ValueError("empty slide manifest")
    return patients


def patient_features(paths, encoder, core):
    blocks = []
    for path in paths:
        coords, mpp, source, _ = core.plan_coords(path, core.sha256_file(path))
        if coords is None or len(coords) < core.MIN_TILES:
            print(f"Skipping {path.name}: missing pixel size or too few tissue tiles", file=sys.stderr)
            continue
        print(f"{path.name}: {len(coords)} tiles, {mpp:g} micrometers per pixel ({source})", flush=True)
        tiles = core.read_tiles(path, coords, mpp)
        block = core.encode_tiles(encoder, tiles, 60).astype(np.float16)
        if block.shape != (len(coords), core.DIM) or not np.isfinite(block).all():
            raise ValueError(f"{path.name}: invalid tile features")
        blocks.append(block)
    if not blocks:
        raise ValueError("no usable slides for this patient")
    return np.concatenate(blocks).astype(np.float32).mean(axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with patient,slide columns")
    parser.add_argument("--virchow2", required=True, type=Path, help="Virchow2 weights file or folder")
    parser.add_argument("--crisp", required=True, type=Path, help="official CRISP checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="new patient feature file (.npz)")
    args = parser.parse_args(argv)
    try:
        if args.out.suffix != ".npz" or args.out.exists():
            raise ValueError("--out must name a new .npz file")
        patients = read_manifest(args.manifest)
        try:
            import cryopair_predict as core
            import openslide
            import timm
        except (ImportError, OSError) as exc:
            raise ValueError(f"install the slides extra and native OpenSlide library: {exc}") from exc
        try:
            encoder = core.load_virchow2(args.virchow2)
            state, merged, total = core.crisp_state(args.crisp, False)
            core.to_crisp(encoder, state, merged, total)
            del state
            features = []
            for patient, slides in patients.items():
                try:
                    features.append(patient_features(slides, encoder, core))
                except (ValueError, openslide.OpenSlideError) as exc:
                    raise ValueError(f"{patient}: {exc}") from exc
        except core.Problem as exc:
            raise ValueError(str(exc)) from exc
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
            archive = Path(tmp) / "features.npz"
            np.savez(archive, patient=np.array(list(patients)), features=np.stack(features))
            os.link(archive, args.out)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

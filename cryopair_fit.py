"""Fit an RNA reader or a PAIRMAP from patient-level features, for use with the CryoPair prediction command."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
ALPHAS = (10.0, 100.0, 1000.0, 1e4, 1e5)
GENE_CHUNK = 2000
MIN_FIT, MIN_VALIDATION = 30, 10


class Problem(Exception):
    """An input problem that the user can fix; reported without a traceback."""


def log(*parts):
    print(time.strftime("%H:%M:%S") + " | " + " ".join(str(p) for p in parts), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def ridge_multi(X, Y, groups, chunk=GENE_CHUNK):
    """Ridge for thousands of targets: one inner three-fold patient split, a per-target penalty, each decomposition computed once."""
    inner = np.array([stable_int(20261010, "inner", g) % 3 for g in groups])
    p = X.shape[1]

    def decompose(Xa):
        s = Xa.std(0) + 1e-6
        Xs = Xa / s
        mx = Xs.mean(0)
        Xc = Xs - mx
        if Xc.shape[0] > Xc.shape[1]:
            lam, V = np.linalg.eigh(Xc.T @ Xc)
            return dict(eig=True, s=s, mx=mx, Xc=Xc, V=V, lam=np.clip(lam, 0, None))
        U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        return dict(eig=False, s=s, mx=mx, U=U, sv=sv, Vt=Vt)

    def solve(dec, Ya, a):
        my = Ya.mean(0)
        if dec["eig"]:
            W = dec["V"] @ ((dec["V"].T @ (dec["Xc"].T @ (Ya - my))) / (dec["lam"] + a * p / 1000)[:, None])
        else:
            W = dec["Vt"].T @ ((dec["sv"] / (dec["sv"] ** 2 + a * p / 1000))[:, None] * (dec["U"].T @ (Ya - my)))
        W = W / dec["s"][:, None]
        return W, my - (dec["mx"] * dec["s"]) @ W

    m = Y.shape[1]
    sc = np.zeros((len(ALPHAS), m))
    for k in range(3):
        tr, va = inner != k, inner == k
        if tr.sum() < 30 or va.sum() < 10:
            continue
        dec = decompose(X[tr])
        for j0 in range(0, m, chunk):
            Yt, Yv = Y[tr, j0:j0 + chunk], Y[va, j0:j0 + chunk]
            den = np.maximum(((Yv - Yv.mean(0)) ** 2).sum(0), 1e-12)
            for i, a in enumerate(ALPHAS):
                W, b = solve(dec, Yt, a)
                sc[i, j0:j0 + chunk] += 1 - ((Yv - (X[va] @ W + b)) ** 2).sum(0) / den
    best = np.argmax(sc, 0)
    dec = decompose(X)
    W_all = np.zeros((p, m), np.float32)
    b_all = np.zeros(m, np.float32)
    for j0 in range(0, m, chunk):
        for i, a in enumerate(ALPHAS):
            j = np.flatnonzero(best[j0:j0 + chunk] == i) + j0
            if len(j):
                W, b = solve(dec, Y[:, j], a)
                W_all[:, j], b_all[j] = W, b
    return W_all, b_all


def ridge_cv(X, Y, groups):
    """Ridge with a per-target penalty from an inner three-fold patient split, all targets fitted at once; used for the PAIRMAP."""
    inner = np.array([stable_int(20261010, "inner", g) % 3 for g in groups])
    m = Y.shape[1]

    def fit(Xa, Ya, alphas):
        s = Xa.std(0) + 1e-6
        Xs = Xa / s
        mx, my = Xs.mean(0), Ya.mean(0)
        Xc = Xs - mx
        if Xc.shape[0] > Xc.shape[1]:
            lam, V = np.linalg.eigh(Xc.T @ Xc)
            lam = np.clip(lam, 0, None)
            XtY = Xc.T @ (Ya - my)
            Ws = [V @ ((V.T @ XtY) / (lam + a * Xs.shape[1] / 1000)[:, None]) for a in alphas]
        else:
            U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
            UtY = U.T @ (Ya - my)
            Ws = [Vt.T @ ((sv / (sv ** 2 + a * Xs.shape[1] / 1000))[:, None] * UtY) for a in alphas]
        return [(W / s[:, None], my - (mx * s) @ (W / s[:, None])) for W in Ws]

    def r2(y, p):
        return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))

    sc = np.zeros((len(ALPHAS), m))
    for k in range(3):
        tr, va = inner != k, inner == k
        if tr.sum() < 30 or va.sum() < 10:
            continue
        for i, (W, b) in enumerate(fit(X[tr], Y[tr], ALPHAS)):
            P = X[va] @ W + b
            sc[i] += [r2(Y[va, j], P[:, j]) for j in range(m)]
    best = np.argmax(sc, 0)
    fits = fit(X, Y, ALPHAS)
    W = np.zeros((X.shape[1], m))
    b = np.zeros(m)
    for i in np.unique(best):
        j = best == i
        W[:, j] = fits[i][0][:, j]
        b[j] = fits[i][1][j]
    return W.astype(np.float32), b.astype(np.float32)


def load_table(path, key, what):
    """A table of patients by columns from an .npz file: patient names under 'patient' and the values under the given key. Read without pickle."""
    try:
        with np.load(path, allow_pickle=False) as z:
            patients, values = [str(p) for p in z["patient"].tolist()], z[key]
            names = [str(g) for g in z["gene"].tolist()] if "gene" in z.files else None
    except KeyError as e:
        raise Problem(f"{Path(path).name}: {what} need the arrays 'patient' and '{key}' ({e} is missing)")
    except Exception as e:
        raise Problem(f"{Path(path).name} could not be read ({type(e).__name__}: {e})")
    if values.ndim != 2 or values.shape[0] != len(patients) or values.shape[1] == 0 or not np.issubdtype(values.dtype, np.floating):
        raise Problem(f"{Path(path).name}: '{key}' must be a table of numbers with one row per patient")
    if len(set(patients)) != len(patients):
        raise Problem(f"{Path(path).name}: a patient is listed twice")
    if not np.isfinite(values).all():
        raise Problem(f"{Path(path).name}: '{key}' holds missing or non-finite values; remove those patients first")
    return patients, values, names


def align(patients, other_patients, other_values, what):
    """The rows of the second table in the order of the first; both must list the same patients."""
    if set(patients) != set(other_patients):
        only = sorted(set(patients) ^ set(other_patients))
        raise Problem(f"{what} list different patients ({len(only)} are in one file only, for example {only[0]})")
    index = {p: i for i, p in enumerate(other_patients)}
    return other_values[[index[p] for p in patients]]


def check_folds(patients):
    """Every inner fold must be large enough to choose the penalty; otherwise the choice would silently fall back to the weakest penalty."""
    inner = np.array([stable_int(20261010, "inner", g) % 3 for g in patients])
    for k in range(3):
        if (inner != k).sum() < MIN_FIT or (inner == k).sum() < MIN_VALIDATION:
            raise Problem(f"{len(patients)} patients are too few to choose the penalty (a validation fold has {(inner == k).sum()} patients; at least {MIN_VALIDATION} are needed)")


def new_file(path):
    out = Path(path)
    if out.exists():
        raise Problem(f"{out} already exists; choose another name")
    if out.suffix != ".npz":
        raise Problem("the output file name must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def fit_reader(args):
    patients, X, _ = load_table(args.features, "features", "features")
    target_patients, Y, genes = load_table(args.targets, "values", "targets")
    if genes is None or len(genes) != Y.shape[1] or len(set(genes)) != len(genes):
        raise Problem(f"{Path(args.targets).name}: 'gene' must name every target column once")
    if not args.units.strip():
        raise Problem("--units must state the scale of the targets, for example 'log2(TPM + 1)'")
    Y = align(patients, target_patients, Y, "features and targets")
    check_folds(patients)
    out = new_file(args.out)
    log(f"fitting a reader: {len(patients)} patients, {X.shape[1]} features, {len(genes)} targets")
    W, b = ridge_multi(X, Y, np.array(patients))
    W, b = np.asarray(W, np.float32), np.asarray(b, np.float32)
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise Problem("the fit gave non-finite coefficients; check the features for constant or extreme columns")
    meta = dict(kind="rna_reader", space=args.space, target_units=args.units, feature_dimension=int(X.shape[1]), genes=len(genes), patients=len(patients),
                fit="ridge with a per-target penalty from an inner three-fold patient split", penalties=list(ALPHAS), tool_version=VERSION,
                features_sha256=sha256_file(args.features), targets_sha256=sha256_file(args.targets))
    np.savez(out, W=W, b=b, gene=np.array(genes), meta=np.array(json.dumps(meta)))
    log(f"reader written to {out}")


def fit_pairmap(args):
    patients, frozen, _ = load_table(args.frozen, "features", "frozen features")
    ffpe_patients, ffpe, _ = load_table(args.ffpe, "features", "FFPE features")
    ffpe = align(patients, ffpe_patients, ffpe, "frozen and FFPE features")
    if frozen.shape[1] != ffpe.shape[1]:
        raise Problem(f"frozen features have {frozen.shape[1]} columns and FFPE features {ffpe.shape[1]}; both must come from the same encoder")
    check_folds(patients)
    out = new_file(args.out)
    log(f"fitting a PAIRMAP: {len(patients)} patients, {frozen.shape[1]} features")
    W, b = ridge_cv(frozen, ffpe, np.array(patients))
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise Problem("the fit gave non-finite coefficients; check the features for constant or extreme columns")
    meta = dict(kind="PAIRMAP", backbone=args.backbone, input="patient mean of frozen tile features", output="patient mean of FFPE tile features, same encoder",
                fit="ridge with a per-output penalty from an inner three-fold patient split", training_patients=len(patients), penalties=list(ALPHAS), tool_version=VERSION,
                frozen_sha256=sha256_file(args.frozen), ffpe_sha256=sha256_file(args.ffpe))
    np.savez(out, W=W, b=b, meta=np.array(json.dumps(meta)))
    log(f"PAIRMAP written to {out}")


def parse(argv):
    p = argparse.ArgumentParser(description="Fit the files the CryoPair prediction command reads, from patient-level features (one row per patient, the mean over the patient's tiles).")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("reader", help="an RNA reader from FFPE features and measured expression of the same patients")
    r.add_argument("--features", required=True, help=".npz with 'patient' and 'features' (FFPE patient features)")
    r.add_argument("--targets", required=True, help=".npz with 'patient', 'gene' and 'values' (expression, patients by genes)")
    r.add_argument("--space", required=True, help="name of the encoder the features come from, for example CRISP")
    r.add_argument("--units", required=True, help="scale of the targets, for example 'log2(TPM + 1)'")
    r.add_argument("--out", required=True, help="new .npz file for the reader")
    m = sub.add_parser("pairmap", help="a PAIRMAP from frozen and FFPE features of the same patients")
    m.add_argument("--frozen", required=True, help=".npz with 'patient' and 'features' (frozen patient features)")
    m.add_argument("--ffpe", required=True, help=".npz with 'patient' and 'features' (FFPE patient features of the same encoder)")
    m.add_argument("--backbone", required=True, help="name of the encoder, for example CRISP")
    m.add_argument("--out", required=True, help="new .npz file for the map")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    try:
        (fit_reader if args.command == "reader" else fit_pairmap)(args)
    except Problem as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

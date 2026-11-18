"""Predict RNA expression from frozen H&E slides with CRISP, CryoPair-CRISP or CRISP with PAIRMAP."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "0.2.1"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DIM = 2560
LORA_RANK, LORA_ALPHA = 8, 16
CRISP_LORA_SCALE = 2.0
TILE_PX, TARGET_MPP, MAX_TILES, MIN_TILES, MIN_TISSUE, THUMB_MAX = 224, 0.50, 256, 16, 0.50, 2048
EXTRACT_SEED = 20260927
ENCODE_BATCH = 32
GPU_POLL_S = 60
ROUTES = {"crisp": ("crisp", False), "cryopair-crisp": ("cryopair-crisp", False), "crisp-pairmap": ("crisp", True)}   # route: (encoder state, PAIRMAP applied)


class Problem(Exception):
    """An input or environment problem that the user can fix; reported without a traceback."""


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


def slide_mpp(s):
    """Micrometers per pixel at full resolution: measured from the file if recorded, otherwise inferred from the objective magnification."""
    p = s.properties
    for k in ("openslide.mpp-x", "aperio.MPP"):
        try:
            v = float(p.get(k))
            if 0.05 < v < 5:
                return v, "measured"
        except (TypeError, ValueError):
            pass
    try:
        unit, res = str(p.get("tiff.ResolutionUnit", "")).lower(), float(p.get("tiff.XResolution"))
        if res > 0 and unit in ("centimeter", "cm") and 0.05 < 10000.0 / res < 5:
            return 10000.0 / res, "measured"
        if res > 0 and unit == "inch" and 0.05 < 25400.0 / res < 5:
            return 25400.0 / res, "measured"
    except (TypeError, ValueError):
        pass
    for k in ("openslide.objective-power", "aperio.AppMag"):
        try:
            mag = float(p.get(k))
            if mag > 0:
                return 10.0 / mag, "inferred"
        except (TypeError, ValueError):
            pass
    return None, "missing"


def otsu(x):
    h, e = np.histogram(x[np.isfinite(x)], bins=256, range=(0, 1))
    w = np.cumsum(h)
    w2 = w[-1] - w
    m = np.cumsum(h * (e[:-1] + e[1:]) / 2)
    mu1, mu2 = m / np.maximum(w, 1), (m[-1] - m) / np.maximum(w2, 1)
    return float(e[np.argmax(w * w2 * (mu1 - mu2) ** 2)])


def tissue_mask(thumb):
    from scipy import ndimage
    a = np.asarray(thumb.convert("RGB"), dtype=np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    m = (sat > max(0.05, otsu(sat))) & (mx < 0.95) & (mx > 0.08)
    return ndimage.binary_closing(ndimage.binary_opening(m, iterations=1), iterations=2)


def plan_coords(path, seed_text):
    """Tissue tiles of 224 pixels at 0.5 micrometers per pixel; if there are more than MAX_TILES, that many are drawn with a seed from seed_text."""
    import openslide
    s = openslide.OpenSlide(str(path))
    try:
        mpp0, source = slide_mpp(s)
        if mpp0 is None:
            return None, None, source, 0
        W, H = s.dimensions
        tile0 = TILE_PX * TARGET_MPP / mpp0
        mask = tissue_mask(s.get_thumbnail((THUMB_MAX, THUMB_MAX)))
    finally:
        s.close()
    sx, sy = mask.shape[1] / W, mask.shape[0] / H
    integ = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    coords = []
    for j in range(int(H // tile0)):
        for i in range(int(W // tile0)):
            x0, y0 = i * tile0, j * tile0
            a0, a1 = int(x0 * sx), max(int((x0 + tile0) * sx), int(x0 * sx) + 1)
            b0, b1 = int(y0 * sy), max(int((y0 + tile0) * sy), int(y0 * sy) + 1)
            a1, b1 = min(a1, mask.shape[1]), min(b1, mask.shape[0])
            if (integ[b1, a1] - integ[b0, a1] - integ[b1, a0] + integ[b0, a0]) / max((a1 - a0) * (b1 - b0), 1) >= MIN_TISSUE:
                coords.append((int(round(x0)), int(round(y0))))
    found = len(coords)
    if len(coords) > MAX_TILES:
        idx = np.sort(np.random.default_rng(stable_int(EXTRACT_SEED, seed_text)).choice(len(coords), MAX_TILES, replace=False))
        coords = [coords[k] for k in idx]
    return np.array(coords, np.int64).reshape(-1, 2), mpp0, source, found


def tile_reader(path, mpp0):
    """224-pixel tiles at 0.5 micrometers per pixel, read from the nearest finer level."""
    import openslide
    sl = openslide.OpenSlide(str(path))
    tile0 = TILE_PX * TARGET_MPP / mpp0
    ds = np.asarray(sl.level_downsamples, float)
    level = int(np.max(np.flatnonzero(ds <= tile0 / TILE_PX * 1.01))) if np.any(ds <= tile0 / TILE_PX * 1.01) else 0
    return sl, level, int(round(tile0 / ds[level]))


def read_tiles(path, coords, mpp0):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    sl, level, read_px = tile_reader(path, mpp0)
    out = np.empty((len(coords), TILE_PX, TILE_PX, 3), np.uint8)
    try:
        for i, (x, y) in enumerate(coords):
            region = sl.read_region((int(x), int(y)), level, (read_px, read_px))
            bg = Image.new("RGB", region.size, (255, 255, 255))
            bg.paste(region, mask=region.split()[-1] if region.mode == "RGBA" else None)
            if read_px != TILE_PX:
                bg = bg.resize((TILE_PX, TILE_PX), Image.BILINEAR)
            out[i] = np.asarray(bg, dtype=np.uint8)
    finally:
        sl.close()
    return out


def load_virchow2(weights):
    """Virchow2 from its released weight file (or the folder that holds it)."""
    import timm
    weights = Path(weights)
    wf = weights if weights.is_file() else next((weights / n for n in ("model.safetensors", "pytorch_model.bin") if (weights / n).is_file()), None)
    if wf is None:
        raise Problem(f"no Virchow2 weight file (model.safetensors or pytorch_model.bin) at {weights}")
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, reg_tokens=4, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=nn.SiLU, mlp_ratio=6832 / 1280, init_values=1e-5, global_pool="", num_classes=0)
    if wf.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(wf))
    else:
        sd = torch.load(str(wf), map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(sd, strict=False)
    if len(missing) > 5:
        raise Problem(f"Virchow2: {len(missing)} weights missing from {wf.name}; this is not the Virchow2 weight file")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device).to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    log(f"Virchow2: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, on {device}")
    return dict(name="Virchow2", model=model, device=device, mean=IMAGENET[0], std=IMAGENET[1], pool="cls_mean", n_prefix=n_prefix, weight_dtype=torch.float32, dim=DIM,
                weights_file=wf)


def crisp_state(path, trust):
    """CRISP's released checkpoint as plain Virchow2 weights: its LoRA branches merged into the layers they adapt."""
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=not trust)
    except Exception as e:
        if trust:
            raise
        raise Problem(f"the CRISP checkpoint could not be read as plain weights ({type(e).__name__}). If the file comes from its official source, "
                      "rerun with --trust-checkpoint, which lets the file run code while it loads")
    if isinstance(state, dict) and "state_dict" in state and not any("lora_" in k for k in state):
        state = state["state_dict"]
    base, lora = {}, {}
    for k, v in state.items():
        k2 = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        if ".lora_A." in k2 or ".lora_B." in k2:
            lora.setdefault(k2.split(".lora_")[0], {})["A" if ".lora_A." in k2 else "B"] = v.float()
        elif ".lora_" not in k2:
            base[k2.replace(".base_layer.", ".")] = v
    sd, merged = dict(base), 0
    for mod, ab in lora.items():
        if "A" in ab and "B" in ab and f"{mod}.weight" in base:
            w = base[f"{mod}.weight"]
            sd[f"{mod}.weight"] = (w.float() + CRISP_LORA_SCALE * (ab["B"] @ ab["A"])).to(w.dtype)
            merged += 1
    return sd, merged, len(lora)


def to_crisp(backbone, sd, merged, n_lora):
    """Replace the Virchow2 weights in the loaded model by CRISP's, in place."""
    if merged == 0 or merged != n_lora:
        raise Problem(f"CRISP: {merged} of {n_lora} LoRA branches could be merged; the checkpoint layout is not the expected one")
    before = {k: v.detach().clone() for k, v in backbone["model"].state_dict().items() if k.endswith("attn.qkv.weight")}
    missing, unexpected = backbone["model"].load_state_dict({k: v.to(torch.float32) if torch.is_floating_point(v) else v for k, v in sd.items()}, strict=False)
    layer_keys = [k for k in list(missing) + list(unexpected) if k.startswith(("blocks.", "patch_embed.", "norm."))]
    if layer_keys or len(missing) > 5:
        raise Problem(f"CRISP: {len(missing)} weights missing and {len(unexpected)} unused, layer weights among them: {layer_keys[:5]}")
    after = backbone["model"].state_dict()
    changed = sum(int(not torch.equal(before[k], after[k])) for k in before)
    if changed == 0:
        raise Problem("CRISP: no attention weight differs from Virchow2 after loading; the adaptation was not applied")
    backbone["name"] = "CRISP"
    log(f"CRISP: {merged} LoRA branches merged, {changed} of {len(before)} attention blocks differ from Virchow2")


def forward_features(backbone, tiles_uint8):
    model, device = backbone["model"], backbone["device"]
    mean = torch.tensor(backbone["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbone["std"], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(tiles_uint8)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = ((x - mean) / std).to(backbone["weight_dtype"])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = getattr(model, "patch_drop", torch.nn.Identity())(x)
        x = getattr(model, "norm_pre", torch.nn.Identity())(x)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        if backbone["pool"] == "cls_mean":
            f = torch.cat([x[:, 0], x[:, backbone["n_prefix"]:].mean(1)], -1)
        else:
            f = x[:, 0]
    return f.float()


def out_of_memory(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def encode_tiles(backbone, tiles, wait_minutes):
    """Features of every tile, in batches of fixed size. If the GPU runs out of memory the batch is repeated after a pause, so the batch size never changes."""
    out, k, started = [], 0, None
    with torch.inference_mode():
        while k < len(tiles):
            try:
                out.append(forward_features(backbone, np.asarray(tiles[k:k + ENCODE_BATCH])).cpu().numpy())
                k += ENCODE_BATCH
                started = None
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not out_of_memory(e):
                    raise
                started = started or time.time()
                if time.time() - started > wait_minutes * 60:
                    raise Problem(f"the GPU had too little free memory for {wait_minutes:.0f} minutes; rerun when it is free")
                log("out of GPU memory; waiting and repeating the batch")
                torch.cuda.empty_cache()
                time.sleep(GPU_POLL_S)
    return np.concatenate(out) if out else np.zeros((0, backbone["dim"]), np.float32)


class Adapter(nn.Module):
    """A LoRA branch on a frozen Linear layer; with enabled False the layer is the original one."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = LORA_ALPHA / LORA_RANK
        self.enabled = True
        self.A = nn.Parameter(torch.zeros(LORA_RANK, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, LORA_RANK))

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + ((x @ self.A.t()) @ self.B.t()) * self.scale


def add_adapters(backbone, state):
    """The CryoPair adapters on every attention qkv and proj layer of CRISP."""
    adapters = []
    for blk in backbone["model"].blocks:
        for attr in ("qkv", "proj"):
            ad = Adapter(getattr(blk.attn, attr)).to(backbone["device"])
            setattr(blk.attn, attr, ad)
            adapters.append(ad)
    with torch.no_grad():
        for i, ad in enumerate(adapters):
            ad.A.copy_(state[f"block{i}.A"].to(ad.A.device))
            ad.B.copy_(state[f"block{i}.B"].to(ad.B.device))
    for ad in adapters:
        ad.A.requires_grad_(False)
        ad.B.requires_grad_(False)
    backbone["adapters"] = adapters


def set_adapters(backbone, enabled):
    for ad in backbone["adapters"]:
        ad.enabled = enabled


def load_npz(path, what):
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        raise Problem(f"{what} {Path(path).name} could not be read ({type(e).__name__}: {e})")


def load_reader(path):
    """An RNA reader: coefficients (features x genes), intercepts and gene names."""
    z = load_npz(path, "reader")
    try:
        W, b, genes, meta = z["W"], z["b"], [str(g) for g in z["gene"].tolist()], json.loads(str(z["meta"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise Problem(f"{Path(path).name} is not an RNA reader file")
    if not isinstance(meta, dict) or meta.get("kind") != "rna_reader" or W.ndim != 2 or b.shape != (W.shape[1],) or len(genes) != W.shape[1] or len(set(genes)) != len(genes):
        raise Problem(f"{Path(path).name} is not a consistent RNA reader file")
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise Problem(f"{Path(path).name} holds non-finite coefficients")
    if meta.get("space") != "CRISP" or W.shape[0] != DIM:
        raise Problem(f"{Path(path).name} is the reader of {meta.get('space')!r} with {W.shape[0]} features; the CRISP reader ({DIM} features) is needed")
    units = meta.get("target_units")
    if not isinstance(units, str) or not units.strip():
        raise Problem(f"{Path(path).name}: target units are missing; they cannot be inferred from the coefficients")
    return dict(W=W, b=b, genes=genes, units=units)


def load_map(path):
    """A PAIRMAP: a square matrix and an offset that move frozen patient features toward the FFPE features of the same encoder."""
    z = load_npz(path, "PAIRMAP")
    try:
        W, b, meta = z["W"], z["b"], json.loads(str(z["meta"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise Problem(f"{Path(path).name} is not a PAIRMAP file")
    if not isinstance(meta, dict) or meta.get("kind") != "PAIRMAP" or W.ndim != 2 or W.shape[0] != W.shape[1] or b.shape != (W.shape[0],) or not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise Problem(f"{Path(path).name} is not a consistent PAIRMAP file")
    if meta.get("backbone") != "CRISP" or W.shape[0] != DIM:
        raise Problem(f"{Path(path).name} is the PAIRMAP of {meta.get('backbone')!r}; the CRISP map is needed")
    return dict(W=W, b=b)


def load_adapters(path):
    """Validate identity, backbone, LoRA settings and tensor layout before attaching adapters."""
    z = load_npz(path, "adapter file")
    try:
        meta = json.loads(str(z["meta"]))
    except (KeyError, ValueError, TypeError):
        raise Problem(f"{Path(path).name}: adapter metadata is missing or invalid")
    if not isinstance(meta, dict) or meta.get("kind") != "cryopair_adapters" or meta.get("backbone") != "CRISP":
        raise Problem(f"{Path(path).name}: CryoPair-CRISP adapters are required")
    if meta.get("rank") != LORA_RANK or meta.get("alpha") != LORA_ALPHA or meta.get("layers") != "attention qkv and proj of every block":
        raise Problem(f"{Path(path).name}: adapter rank, scale or layer layout does not match CRISP")
    if meta.get("pairing", "paired") != "paired":
        raise Problem(f"{Path(path).name}: these adapters were trained with {meta.get('pairing')!r} pairing (a control), so they are not CryoPair-CRISP")
    if not meta.get("complete", True):
        raise Problem(f"{Path(path).name}: these adapters come from an unfinished training run ({meta.get('steps')} of {meta.get('total_steps')} steps); finish the run first")
    keys = sorted(k for k in z if k != "meta")
    n = len(keys) // 2
    if not n or n % 2 or keys != sorted([f"block{i}.A" for i in range(n)] + [f"block{i}.B" for i in range(n)]):
        raise Problem(f"{Path(path).name}: incomplete adapter layers")
    width = DIM // 2
    state, h = {}, hashlib.sha256()
    for k in keys:
        a = np.asarray(z[k])
        layer = int(k.split('.')[0][5:])
        shape = (LORA_RANK, width) if k.endswith('.A') else ((3 * width if layer % 2 == 0 else width), LORA_RANK)
        if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
            raise Problem(f"{Path(path).name}: {k} must be a finite float32 array of shape {shape}")
        h.update(k.encode())
        h.update(np.ascontiguousarray(a).tobytes())
        state[k] = torch.from_numpy(a.copy())
    identity = h.hexdigest()[:16]
    if identity != meta.get("identity"):
        raise Problem(f"{Path(path).name}: adapter arrays do not match their recorded identity")
    return state, n, identity


def load_tile_plan(path, slide_sha256=None):
    z = load_npz(path, "tile plan")
    try:
        coords, mpp0 = np.asarray(z["coords"]), float(z["mpp0"])
    except (KeyError, TypeError, ValueError):
        raise Problem(f"{Path(path).name} is not a tile plan (coords and mpp0)")
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.issubdtype(coords.dtype, np.integer) or not np.isfinite(mpp0) or not 0.05 < mpp0 < 5 or len(coords) > MAX_TILES or (coords < 0).any() or len({tuple(c) for c in coords.tolist()}) != len(coords):
        raise Problem(f"{Path(path).name} holds an invalid tile plan")
    if slide_sha256 is not None and str(z.get("slide_sha256", "")) != slide_sha256:
        raise Problem(f"{Path(path).name}: the tile plan is not bound to this slide file")
    source = str(z["mpp_source"]) if "mpp_source" in z else "from tile plan"
    return coords.astype(np.int64), mpp0, source


def parse(argv):
    p = argparse.ArgumentParser(description="Predict RNA expression, log2(TPM + 1), for one patient from frozen H&E whole-slide images.")
    p.add_argument("--slides", nargs="+", required=True, help="slide files of ONE patient; their tiles are pooled together")
    p.add_argument("--models", nargs="+", required=True, choices=sorted(ROUTES), help="one or more of: crisp, cryopair-crisp, crisp-pairmap")
    p.add_argument("--virchow2", required=True, help="Virchow2 weight file, or the folder that holds it")
    p.add_argument("--crisp", required=True, help="CRISP checkpoint (CRISP.pth)")
    p.add_argument("--reader", required=True, help="RNA reader of CRISP (.npz)")
    p.add_argument("--adapters", help="CryoPair-CRISP adapter file (.npz); needed for cryopair-crisp")
    p.add_argument("--pairmap", help="PAIRMAP of CRISP (.npz); needed for crisp-pairmap")
    p.add_argument("--tiles", nargs="+", help="tile plans saved by an earlier run, one per slide in the same order; without them tiles are planned from the slide")
    p.add_argument("--seed", type=int, help="draw another tile sample: a whole number mixed into the sampling seed; without it the sample depends on the slide file alone")
    p.add_argument("--out", required=True, help="new folder for the results")
    p.add_argument("--trust-checkpoint", action="store_true", help="allow the CRISP checkpoint to be loaded with full unpickling")
    p.add_argument("--gpu-wait-minutes", type=float, default=60.0, help="how long to wait for GPU memory before stopping")
    return p.parse_args(argv)


def run(args):
    started = time.time()
    routes = list(dict.fromkeys(args.models))
    states = list(dict.fromkeys(ROUTES[r][0] for r in routes))
    slides = [Path(s) for s in args.slides]
    for f in slides:
        if not f.is_file():
            raise Problem(f"slide {f} not found")
    if len({str(f.resolve()) for f in slides}) != len(slides):
        raise Problem("the same slide is listed twice")
    if args.seed is not None and args.tiles:
        raise Problem("--seed has no effect with saved tile plans; give one or the other")
    if args.seed is not None and args.seed < 0:
        raise Problem("--seed must be zero or a positive whole number")
    if args.tiles and len(args.tiles) != len(slides):
        raise Problem(f"{len(args.tiles)} tile plans for {len(slides)} slides; give one per slide, in the same order")
    for name, f in (("--crisp", args.crisp), ("--reader", args.reader), ("--adapters", args.adapters), ("--pairmap", args.pairmap)):
        if f and not Path(f).is_file():
            raise Problem(f"{name}: {f} not found")
    if not Path(args.virchow2).exists():
        raise Problem(f"--virchow2: {args.virchow2} not found")
    if "cryopair-crisp" in states and not args.adapters:
        raise Problem("cryopair-crisp needs --adapters")
    if any(ROUTES[r][1] for r in routes) and not args.pairmap:
        raise Problem("crisp-pairmap needs --pairmap")
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Problem(f"{out} exists and is not an empty folder")

    reader = load_reader(args.reader)
    pairmap = load_map(args.pairmap) if any(ROUTES[r][1] for r in routes) else None
    adapters = load_adapters(args.adapters) if "cryopair-crisp" in states else None
    try:
        import openslide  # noqa: F401
        import timm  # noqa: F401
    except ImportError as e:
        raise Problem(f"a required package is missing: {e}")

    plans, notes = [], []
    for i, f in enumerate(slides):
        identity = sha256_file(f)
        if args.tiles:
            coords, mpp0, source = load_tile_plan(args.tiles[i], identity)
            found = None
        else:
            try:
                coords, mpp0, source, found = plan_coords(f, identity if args.seed is None else f"{identity}|{args.seed}")
            except Problem:
                raise
            except Exception as e:
                raise Problem(f"{f.name} could not be opened as a whole-slide image ({type(e).__name__}: {e})")
        plan = dict(slide=f.name, path=f, sha256=identity, mpp=mpp0, mpp_source=source, tissue_tiles_found=found, tiles=0 if coords is None else int(len(coords)), used=False, coords=coords)
        if coords is None:
            notes.append(f"{f.name}: no pixel size in the file; slide not used")
        elif len(coords) < MIN_TILES:
            notes.append(f"{f.name}: {len(coords)} tissue tiles, fewer than {MIN_TILES}; slide not used")
        else:
            plan["used"] = True
            if source == "inferred":
                notes.append(f"{f.name}: pixel size inferred from the objective magnification, not measured")
        plans.append(plan)
        log(f"{f.name}: {plan['tiles']} tiles, {mpp0} micrometers per pixel ({source}){'' if plan['used'] else ', NOT USED'}")
    if not any(p["used"] for p in plans):
        raise Problem("no slide has a pixel size and at least " + str(MIN_TILES) + " tissue tiles; nothing to predict")

    sd, merged, n_lora = crisp_state(args.crisp, args.trust_checkpoint)
    backbone = load_virchow2(args.virchow2)
    if backbone["device"].type != "cuda":
        notes.append("no GPU: encoded on the CPU without mixed precision; results were validated on a GPU only")
    to_crisp(backbone, sd, merged, n_lora)
    del sd
    adapter_id = None
    if adapters is not None:
        state, n, adapter_id = adapters
        if n != 2 * len(backbone["model"].blocks):
            raise Problem(f"the adapter file holds {n} layers, the encoder has {2 * len(backbone['model'].blocks)}")
        add_adapters(backbone, state)

    out.mkdir(parents=True, exist_ok=True)
    feats = {s: [] for s in states}
    for k, plan in enumerate(plans):
        if not plan["used"]:
            continue
        tiles = read_tiles(plan["path"], plan["coords"], plan["mpp"])
        for s in states:
            if "adapters" in backbone:
                set_adapters(backbone, s == "cryopair-crisp")
            f = encode_tiles(backbone, tiles, args.gpu_wait_minutes)
            if f.shape != (len(tiles), DIM) or not np.isfinite(f).all():
                raise Problem(f"{plan['slide']}: the encoder returned features of shape {f.shape} or non-finite values")
            with np.errstate(over="ignore"):
                f16 = f.astype(np.float16)                      # tile features are kept at half precision, as in the study
            if not np.isfinite(f16).all():
                raise Problem(f"{plan['slide']}: tile features do not fit half precision")
            feats[s].append(f16)
            np.savez(out / f"tile_features_{s}_slide{k + 1}.npz", feats=f16, coords=plan["coords"], mpp0=plan["mpp"])
        np.savez(out / f"tiles_slide{k + 1}.npz", coords=plan["coords"], mpp0=plan["mpp"], mpp_source=plan["mpp_source"], slide_sha256=plan["sha256"])
        log(f"{plan['slide']}: {len(tiles)} tiles encoded")

    pooled = {s: np.concatenate(v).astype(np.float32).mean(0) for s, v in feats.items()}    # one mean over the tiles of all slides
    predictions = {}
    for r in routes:
        x = pooled[ROUTES[r][0]][None, :]
        if ROUTES[r][1]:
            x = x @ pairmap["W"] + pairmap["b"]
        p = (x @ reader["W"] + reader["b"]).astype(np.float32)[0]
        if not np.isfinite(p).all():
            raise Problem(f"{r}: the prediction is not finite")
        predictions[r] = p
    np.savez(out / "patient_features.npz", **pooled)
    with open(out / "predicted_rna.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gene"] + routes)
        for g, row in zip(reader["genes"], np.stack([predictions[r] for r in routes], 1)):
            w.writerow([g] + [format(float(v), ".9g") for v in row])

    import timm
    record = dict(tool="cryopair_predict", version=VERSION, models=routes, units=reader["units"], genes=len(reader["genes"]),
                  slides=[{k: v for k, v in p.items() if k not in ("coords", "path")} for p in plans], tiles_pooled=int(sum(p["tiles"] for p in plans if p["used"])),
                  tiles_from="saved tile plans" if args.tiles else "planned from the slides, seed from each slide file",
                  sampling_seed=args.seed,
                  inputs=dict(virchow2_sha256=sha256_file(backbone["weights_file"]), crisp_sha256=sha256_file(args.crisp), reader_sha256=sha256_file(args.reader),
                              pairmap_sha256=sha256_file(args.pairmap) if pairmap is not None else None,
                              adapters_sha256=sha256_file(args.adapters) if adapters is not None else None, adapters_identity=adapter_id),
                  settings=dict(tile_pixels=TILE_PX, micrometers_per_pixel=TARGET_MPP, max_tiles_per_slide=MAX_TILES, min_tiles_per_slide=MIN_TILES, batch=ENCODE_BATCH,
                                feature_storage="float16", pooling="mean over all tiles of all slides", trust_checkpoint=bool(args.trust_checkpoint)),
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, timm=timm.__version__, device=str(backbone["device"]),
                                   gpu=torch.cuda.get_device_name(0) if backbone["device"].type == "cuda" else None),
                  notes=notes + ["predictions are research outputs; absolute expression levels are not calibrated against measured RNA for new laboratories"],
                  seconds=round(time.time() - started, 1))
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"done: {len(reader['genes'])} genes for {', '.join(routes)} in {out}")


def main(argv=None):
    args = parse(argv)
    try:
        run(args)
    except Problem as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

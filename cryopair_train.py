"""Train CryoPair adapters for CRISP from paired frozen and FFPE tiles of the same patients."""
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

VERSION = "0.2.1"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DIM = 2560
LORA_RANK, LORA_ALPHA = 8, 16
CRISP_LORA_SCALE = 2.0
EPOCHS = 4
BATCH_PATIENTS = 8
TILES_FROZEN, TILES_FFPE = 32, 16          # per patient and step
PACK_FROZEN, PACK_FFPE = 256, 128          # tiles kept per training patient; a step draws from these
LR, WEIGHT_DECAY, WARMUP_STEPS, GRAD_CLIP = 1e-4, 0.01, 100, 1.0
RETENTION_WEIGHT = 1.0
MICRO_NOW = {"patients": 2}                # patients per forward pass; 1 when GPU memory is short (same result, slower)
CHECKPOINT_STEPS = 200
GPU_POLL_S = 60
SEED = 20261017


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


def stable_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Adapter(nn.Module):
    """A LoRA branch on a frozen Linear layer: y = base(x) + (x A^T) B^T * alpha / rank, with B starting at zero."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = LORA_ALPHA / LORA_RANK
        self.A = nn.Parameter(torch.zeros(LORA_RANK, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, LORA_RANK))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x @ self.A.t()) @ self.B.t()) * self.scale


def make_lora(backbone):
    """Wrap every attention qkv and proj Linear with a LoRA branch; only the adapters train."""
    model = backbone["model"]
    adapters = []
    for blk in model.blocks:
        for attr in ("qkv", "proj"):
            base = getattr(blk.attn, attr)
            if isinstance(base, Adapter):
                base = base.base
            ad = Adapter(base).to(backbone["device"])
            setattr(blk.attn, attr, ad)
            adapters.append(ad)
    backbone["adapters"] = adapters
    return adapters


def lora_parameters(backbone):
    return [p for ad in backbone["adapters"] for p in (ad.A, ad.B)]


def reset_lora(backbone):
    """The same starting adapters for every arm: the initialization is seeded, so arms differ only in their targets."""
    torch.manual_seed(stable_int(SEED, "lora") % (2 ** 31))
    with torch.no_grad():
        for ad in backbone["adapters"]:
            nn.init.kaiming_uniform_(ad.A, a=math.sqrt(5))
            ad.B.zero_()


def lora_state(backbone):
    return {f"block{i}.{n}": t.detach().float().cpu() for i, ad in enumerate(backbone["adapters"]) for n, t in (("A", ad.A), ("B", ad.B))}


def load_lora_state(backbone, state):
    with torch.no_grad():
        for i, ad in enumerate(backbone["adapters"]):
            ad.A.copy_(state[f"block{i}.A"].to(ad.A.device))
            ad.B.copy_(state[f"block{i}.B"].to(ad.B.device))


def forward_features(backbone, tiles_uint8, checkpoint=False):
    """uint8 tiles (n, 224, 224, 3) to feature vectors, the same pixel path as the cached features."""
    from torch.utils.checkpoint import checkpoint as ckpt
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
            x = ckpt(blk, x, use_reentrant=False) if checkpoint else blk(x)
        x = model.norm(x)
        if backbone["pool"] == "cls_mean":
            f = torch.cat([x[:, 0], x[:, backbone["n_prefix"]:].mean(1)], -1)
        else:
            f = x[:, 0]
    return f.float()


def standardization(pack, train_rows):
    """Feature means and scales from the training patients only; nothing from the heldout set enters the loss."""
    S = pack.sample_features[train_rows].reshape(-1, DIM).astype(np.float32)
    return S.mean(0).astype(np.float32), (S.std(0) + 1e-6).astype(np.float32)


class Batches:
    """Draws a step's tiles from the pack; the draw depends only on the seed, so both arms and a resumed run see the same tiles.
    The next batch is prepared in a background thread."""

    def __init__(self, pack, train_rows, mu, sd):
        self.pack, self.rows, self.mu, self.sd = pack, train_rows, mu, sd
        self.pool = ThreadPoolExecutor(2)

    def make(self, rows, partner, seed):
        rng = np.random.default_rng(seed)
        frozen, ffpe, ffpe_target, slide_target = [], [], [], []
        for i in rows:
            j = self.pack.pack_index[int(i)]
            fi = np.sort(rng.choice(PACK_FROZEN, TILES_FROZEN, replace=False))
            di = np.sort(rng.choice(PACK_FFPE, TILES_FFPE, replace=False))
            frozen.append(np.asarray(self.pack.frozen_tiles[j][fi]))
            ffpe.append(np.asarray(self.pack.ffpe_tiles[j][di]))
            ffpe_target.append((self.pack.ffpe_tile_features[j][di].astype(np.float32) - self.mu) / self.sd)
            slide_target.append((self.pack.ffpe_mean[int(partner[int(i)])] - self.mu) / self.sd)
        return dict(frozen=np.concatenate(frozen), ffpe=np.concatenate(ffpe), ffpe_target=np.concatenate(ffpe_target).astype(np.float32),
                    slide_target=np.stack(slide_target).astype(np.float32), n_frozen=[len(f) for f in frozen], n_ffpe=[len(f) for f in ffpe], mu=self.mu, sd=self.sd)

    def submit(self, rows, partner, seed):
        return self.pool.submit(self.make, rows, partner, seed)


def partners(pack, train_rows, arm, rng):
    """paired: every patient is its own target. shuffled: a different patient of the same cancer, when the cancer has two or more."""
    out = {int(i): int(i) for i in train_rows}
    if arm == "paired":
        return out
    cancers = pack.patients.cancer.to_numpy()
    for c in np.unique(cancers[train_rows]):
        members = train_rows[cancers[train_rows] == c]
        if len(members) < 2:
            continue
        perm = rng.permutation(members)
        for _ in range(20):
            if not np.any(perm == members):
                break
            perm = rng.permutation(members)
        for a, b in zip(members, perm):
            out[int(a)] = int(b)
    return out


def train_step(backbone, optimizer, batch, scheduler=None):
    """One optimizer step over the batch as micro-batches with accumulated gradients; the loss is that of the whole batch."""
    device = backbone["device"]
    mu, sd = (torch.as_tensor(batch[k], device=device) for k in ("mu", "sd"))
    slide_target = torch.as_tensor(batch["slide_target"], device=device)
    ffpe_target = torch.as_tensor(batch["ffpe_target"], device=device)
    n_frozen, n_ffpe = list(batch["n_frozen"]), list(batch["n_ffpe"])
    P, T = len(n_frozen), int(sum(n_ffpe))
    f_edge, d_edge = np.cumsum([0] + n_frozen), np.cumsum([0] + n_ffpe)
    optimizer.zero_grad(set_to_none=True)
    pair_sum = ret_sum = forward_s = backward_s = 0.0
    for p0 in range(0, P, MICRO_NOW["patients"]):
        p1 = min(P, p0 + MICRO_NOW["patients"])
        frozen = batch["frozen"][f_edge[p0]:f_edge[p1]]
        ffpe = batch["ffpe"][d_edge[p0]:d_edge[p1]]
        t0 = time.perf_counter()
        z = (forward_features(backbone, np.concatenate([frozen, ffpe]), checkpoint=True) - mu) / sd
        zf, zd = z[:len(frozen)], z[len(frozen):]
        means = torch.stack([zf[s - f_edge[p0]:e - f_edge[p0]].mean(0) for s, e in zip(f_edge[p0:p1], f_edge[p0 + 1:p1 + 1])])
        pair = ((means - slide_target[p0:p1]) ** 2).mean(1).sum() / P
        ret = ((zd - ffpe_target[d_edge[p0]:d_edge[p1]]) ** 2).mean(1).sum() / T
        loss = pair + RETENTION_WEIGHT * ret
        sync(device)
        t1 = time.perf_counter()
        loss.backward()
        sync(device)
        t2 = time.perf_counter()
        pair_sum += float(pair)
        ret_sum += float(ret)
        forward_s += t1 - t0
        backward_s += t2 - t1
    t2 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(lora_parameters(backbone), GRAD_CLIP)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    sync(device)
    t3 = time.perf_counter()
    return dict(loss=pair_sum + RETENTION_WEIGHT * ret_sum, pair=pair_sum, retention=ret_sum, forward_s=forward_s, backward_s=backward_s, optimizer_s=t3 - t2)


def make_optimizer(backbone, total_steps):
    opt = torch.optim.AdamW(lora_parameters(backbone), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def out_of_memory(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


class TrainingSet:
    """The prepared training arrays, opened read-only: tiles of every training patient and the original encoder's FFPE features."""

    def __init__(self, folder, tiles_folder):
        folder, tiles_folder = Path(folder), Path(tiles_folder)
        table = folder / "patients.csv"
        if not table.is_file():
            raise Problem(f"{table} not found; the training folder needs patients.csv with the columns patient and role")
        self.patients = pd.read_csv(table, dtype={"patient": str})
        if not {"patient", "role"} <= set(self.patients.columns) or self.patients.patient.duplicated().any():
            raise Problem("patients.csv needs the columns patient and role, with every patient listed once")
        if folder.resolve() != tiles_folder.resolve() and (tiles_folder / "frozen_tiles.npy").is_file():
            tile_manifest = tiles_folder / "patients.csv"
            if not tile_manifest.is_file():
                raise Problem("separate tile folder requires its patients.csv for row alignment")
            tile_table = pd.read_csv(tile_manifest, dtype={"patient": str})
            if not {"role", "patient"} <= set(tile_table.columns):
                raise Problem("tile manifest requires patient and role columns")
            if list(tile_table.loc[tile_table.role == "train", "patient"]) != list(self.patients.loc[self.patients.role == "train", "patient"]):
                raise Problem("tile and feature training patient orders differ")
        rows = np.flatnonzero(self.patients.role.to_numpy() == "train")
        self.pack_index = {int(r): i for i, r in enumerate(rows)}
        n_all, n_train = len(self.patients), len(rows)
        shapes = dict(ffpe_mean=(n_all, DIM), sample_features=(n_all, None, DIM), ffpe_tile_features=(n_train, PACK_FFPE, DIM), done=(n_all,), valid=(n_all,),
                      frozen_tiles=(n_train, PACK_FROZEN, 224, 224, 3), ffpe_tiles=(n_train, PACK_FFPE, 224, 224, 3))
        for name, shape in shapes.items():
            path = (tiles_folder if name.endswith("_tiles") else folder) / f"{name}.npy"
            if not path.is_file():
                raise Problem(f"{path} not found")
            a = np.load(path, mmap_mode="r")
            if len(a.shape) != len(shape) or any(want is not None and got != want for got, want in zip(a.shape, shape)):
                raise Problem(f"{path.name} has shape {a.shape}; expected {shape} for {n_all} patients, {n_train} of them for training")
            setattr(self, name, a)
        self.train_rows = np.flatnonzero((self.patients.role.to_numpy() == "train") & np.asarray(self.done) & np.asarray(self.valid))
        if len(self.train_rows) < BATCH_PATIENTS:
            raise Problem(f"only {len(self.train_rows)} complete training patients; at least {BATCH_PATIENTS} are needed")


def batch_digest(batch, patient_ids):
    digest = hashlib.sha256()
    digest.update(json.dumps([str(p) for p in patient_ids]).encode())
    for key in ("frozen", "ffpe", "slide_target", "ffpe_target", "n_frozen", "n_ffpe", "mu", "sd"):
        value = np.ascontiguousarray(batch[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.data)
    return digest.hexdigest()


def adapter_identity(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_adapters(path, state, **facts):
    """Adapter weights as plain arrays, in the format the prediction command reads."""
    arrays = {k: v.detach().float().cpu().numpy() for k, v in state.items()}
    meta = dict(kind="cryopair_adapters", backbone="CRISP", identity=adapter_identity(state), rank=LORA_RANK, alpha=LORA_ALPHA,
                layers="attention qkv and proj of every block", tool_version=VERSION, **facts)
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, meta=np.array(json.dumps(meta)), **arrays)
    temp.replace(path)
    return meta["identity"]


def step_with_retry(backbone, optimizer, batch, scheduler, wait_minutes):
    """A training step that is repeated on the same batch when the GPU runs out of memory; no parameter has changed when that happens."""
    started = None
    update_started = [False]
    def mark_update(*_):
        update_started[0] = True
    while True:
        hook = optimizer.register_step_pre_hook(mark_update)
        try:
            return train_step(backbone, optimizer, batch, scheduler)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not out_of_memory(e):
                raise
            if update_started[0]:
                raise Problem("GPU memory ran out during or after the optimizer update; restart from the last checkpoint")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if MICRO_NOW["patients"] > 1:
                MICRO_NOW["patients"] = 1
                log("out of GPU memory; repeating the step with one patient per forward pass")
                continue
            started = started or time.time()
            if time.time() - started > wait_minutes * 60:
                raise Problem(f"the GPU had too little free memory for {wait_minutes:.0f} minutes; rerun the same command to resume from the last checkpoint")
            log("out of GPU memory; waiting and repeating the step")
            time.sleep(GPU_POLL_S)
        finally:
            hook.remove()


def parse(argv):
    p = argparse.ArgumentParser(description="Train CryoPair adapters for CRISP. Rerunning the same command resumes an unfinished run from its last checkpoint.")
    p.add_argument("--pack", required=True, help="training folder: patients.csv, ffpe_mean.npy, sample_features.npy, ffpe_tile_features.npy, done.npy, valid.npy")
    p.add_argument("--tiles", help="folder with frozen_tiles.npy and ffpe_tiles.npy, if they are not in the training folder")
    p.add_argument("--virchow2", required=True, help="Virchow2 weight file, or the folder that holds it")
    p.add_argument("--crisp", required=True, help="CRISP checkpoint (CRISP.pth)")
    p.add_argument("--out", required=True, help="folder for the adapters, the training log and checkpoints")
    p.add_argument("--pairing", choices=("paired", "shuffled"), default="paired", help="shuffled: the control that pairs each patient with another patient of the same cancer type")
    p.add_argument("--steps", type=int, help="stop after this many steps in total (for a short trial); the adapters are then marked incomplete")
    p.add_argument("--trust-checkpoint", action="store_true", help="allow the CRISP checkpoint to be loaded with full unpickling")
    p.add_argument("--gpu-wait-minutes", type=float, default=60.0, help="how long to wait for GPU memory before stopping")
    return p.parse_args(argv)


def run(args):
    started = time.time()
    if args.steps is not None and args.steps < 1:
        raise Problem("--steps must be a positive whole number")
    for name, f in (("--pack", args.pack), ("--tiles", args.tiles), ("--virchow2", args.virchow2), ("--crisp", args.crisp)):
        if f and not Path(f).exists():
            raise Problem(f"{name}: {f} not found")
    pack = TrainingSet(args.pack, args.tiles or args.pack)
    if args.pairing == "shuffled" and "cancer" not in pack.patients.columns:
        raise Problem("the shuffled control needs a cancer column in patients.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "adapters.npz"
    if final.exists() and not (out / "checkpoint.pt").is_file():
        raise Problem(f"{final} already exists; choose another output folder")
    train_rows = pack.train_rows
    mu, sd = standardization(pack, train_rows)
    sd_state, merged, n_lora = crisp_state(args.crisp, args.trust_checkpoint)
    backbone = load_virchow2(args.virchow2)
    to_crisp(backbone, sd_state, merged, n_lora)
    del sd_state
    make_lora(backbone)
    reset_lora(backbone)
    initial = adapter_identity(lora_state(backbone))
    steps_per_epoch = math.ceil(len(train_rows) / BATCH_PATIENTS)
    total = EPOCHS * steps_per_epoch
    limit = total if args.steps is None else min(total, args.steps)
    opt, sched = make_optimizer(backbone, total)
    order_rng = np.random.default_rng(stable_int(SEED, "order"))
    partner_rng = np.random.default_rng(stable_int(SEED, "partners"))
    batches = Batches(pack, train_rows, mu, sd)
    history, step, start_epoch, start_chunk, saved_epoch, first_batch = [], 0, 0, 0, None, None
    settings = dict(pairing=args.pairing, patients=int(len(train_rows)), seed=SEED, epochs=EPOCHS, batch_patients=BATCH_PATIENTS)
    checkpoint = out / "checkpoint.pt"
    settings["crisp_sha256"] = sha256_file(args.crisp)
    settings["virchow2_sha256"] = sha256_file(backbone["weights_file"])
    settings["patients_sha256"] = sha256_file(Path(args.pack) / "patients.csv")
    settings["array_files"] = {}
    for name in ("ffpe_mean", "sample_features", "ffpe_tile_features", "done", "valid", "frozen_tiles", "ffpe_tiles"):
        path = Path(getattr(pack, name).filename).resolve()
        stat = path.stat()
        settings["array_files"][name] = [hashlib.sha256(str(path).encode()).hexdigest(), stat.st_size, stat.st_mtime_ns]
    if checkpoint.is_file():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["settings"] != settings or ck["initial"] != initial:
            raise Problem(f"{checkpoint} belongs to a run with other settings or data; choose another output folder")
        load_lora_state(backbone, ck["state"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        order_rng.bit_generator.state, partner_rng.bit_generator.state = json.loads(ck["order_rng"]), json.loads(ck["partner_rng"])
        start_epoch, start_chunk, step, first_batch = ck["epoch"], ck["chunk"], ck["step"], ck["first_batch"]
        if final.exists() and step >= limit:
            raise Problem("the checkpoint has already reached the requested step count")
        if start_chunk > 0:
            saved_epoch = (ck["order"].numpy(), dict(zip(ck["partner"][0].tolist(), ck["partner"][1].tolist())))
        history = pd.read_csv(out / "training.csv").to_dict("records")[:step]
        log(f"resuming at epoch {start_epoch + 1}, step {step}")

    def save(epoch, chunk, order, partner):
        temporary_log = out / "training.tmp.csv"
        pd.DataFrame(history).to_csv(temporary_log, index=False)
        temporary_log.replace(out / "training.csv")
        torch.save(dict(state=lora_state(backbone), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch, chunk=chunk, step=step, settings=settings,
                        initial=initial, first_batch=first_batch, order_rng=json.dumps(order_rng.bit_generator.state), partner_rng=json.dumps(partner_rng.bit_generator.state),
                        order=None if order is None else torch.as_tensor(np.asarray(order)),
                        partner=None if partner is None else torch.as_tensor(np.array([list(partner.keys()), list(partner.values())]))), checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)

    log(f"{len(train_rows)} training patients, {steps_per_epoch} steps per epoch, {limit} of {total} steps to run, starting adapters {initial}")
    for epoch in range(start_epoch, EPOCHS):
        if step >= limit:
            break
        if epoch == start_epoch and saved_epoch is not None:
            order, partner = saved_epoch
        else:
            partner = partners(pack, train_rows, args.pairing, partner_rng)
            order = order_rng.permutation(train_rows)
        chunks = [order[k:k + BATCH_PATIENTS] for k in range(0, len(order), BATCH_PATIENTS)]
        first = start_chunk if epoch == start_epoch else 0
        for ci in range(first, len(chunks)):
            batch = batches.make(chunks[ci], partner, stable_int(SEED, "tiles", epoch, ci))
            if epoch == 0 and ci == 0:
                first_batch = batch_digest(batch, pack.patients.patient.iloc[chunks[ci]])
            r = step_with_retry(backbone, opt, batch, sched, args.gpu_wait_minutes)
            step += 1
            history.append(dict(epoch=epoch + 1, step=step, lr=sched.get_last_lr()[0], **r))
            last = ci + 1 == len(chunks)
            if step % CHECKPOINT_STEPS == 0 or step >= limit or last:
                log(f"step {step}/{limit}: loss {np.mean([h['loss'] for h in history[-CHECKPOINT_STEPS:]]):.4f}, {(time.time() - started) / 60:.0f} min")
                save(epoch + 1 if last else epoch, 0 if last else ci + 1, None if last else order, None if last else partner)
            if step >= limit:
                break
    complete = step >= total
    identity = save_adapters(final, lora_state(backbone), complete=bool(complete), steps=int(step), total_steps=int(total), **settings)
    record = dict(tool="cryopair_train", version=VERSION, adapters=identity, starting_adapters=initial, first_batch=first_batch, complete=bool(complete), steps=int(step),
                  total_steps=int(total), settings=dict(settings, tiles_frozen=TILES_FROZEN, tiles_ffpe=TILES_FFPE, learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
                                                        gradient_clip=GRAD_CLIP, retention_weight=RETENTION_WEIGHT, rank=LORA_RANK, alpha=LORA_ALPHA,
                                                        patients_per_forward_pass=MICRO_NOW["patients"]),
                  inputs=dict(virchow2_sha256=sha256_file(backbone["weights_file"]), crisp_sha256=sha256_file(args.crisp)),
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, device=str(backbone["device"]),
                                   gpu=torch.cuda.get_device_name(0) if backbone["device"].type == "cuda" else None),
                  notes=["GPU training results can vary across machines and software versions"],
                  seconds=round(time.time() - started, 1))
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"adapters {identity} written to {final}" + ("" if complete else f" (incomplete: {step} of {total} steps)"))


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

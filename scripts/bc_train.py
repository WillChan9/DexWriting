"""Behaviour-clone the teacher dataset into the PPO Agent network.

    python scripts/bc_train.py data.npz out.pt --init checkpoints/grasp.pt --epochs 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexcraft.env import INK_MAP_H, INK_MAP_W
from dexcraft.train_ppo import Agent, ACT_DIM, TRAJ_ARCHS, VEC_DIM, STATE_KEEP, adapt_checkpoint, tip_px_from_vec

MAP_PX = INK_MAP_H * INK_MAP_W


def episodes(ds):
    """(shard, global step offset, shard-local step offset, length, episode index).

    Two offsets because `vec`/`act` are concatenated across shards while `ink_n` and
    `task_map` are indexed within their own shard.
    """
    out, off = [], 0
    for d in ds:
        local = 0
        for k, L in enumerate(d["ep_len"]):
            out.append((d, off + local, local, int(L), k))
            local += int(L)
        off += int(d["ep_len"].sum())
    return out


def episode_frames(ep, V, A, vmu, vsd, hist=0, chunk=1, maps_off=False):
    """Replay one episode from the compact shard: yield (x, maps, tip, hist, act) per training frame.

    The maps are not stored per step - the task map is constant within an episode and the ink
    map arrives as sparse deltas - so the ink map accumulates as the episode is walked. With
    chunk > 1 one frame is emitted every `chunk` steps and its label is the next `chunk` actions.
    """
    d, g0, l0, L, k = ep
    KH = max(hist, 1)
    if maps_off:
        # `traj` archs: the network has no image branch, so the ink replay - the expensive part of
        # this loop - is skipped and a 1x1 placeholder goes in its place.
        hq = np.zeros((KH, V.shape[1]), np.float32)
        blank = np.zeros((2, 1, 1), np.uint8)
        tip0 = np.zeros(2, np.float32)
        for t in range(0, L, chunk):
            x = np.clip((V[g0 + t] - vmu) / vsd, -10, 10).astype(np.float32)
            j = np.minimum(np.arange(t, t + chunk), L - 1)
            yield x, blank, tip0, hq.copy(), A[g0 + j].reshape(-1)
            if hist:
                hq = np.concatenate([hq[1:], x[None]], 0)
        return
    tmap = d["task_map"][k].astype(np.uint8)
    ink = np.zeros((INK_MAP_H, INK_MAP_W), np.uint8)
    n_all, delta = d["ink_n"], d["ink_delta"]
    cur = int(n_all[:l0].sum())
    hq = np.zeros((KH, V.shape[1]), np.float32)
    for t in range(0, L, chunk):
        for u in range(t, min(t + chunk, L)):
            n = int(n_all[l0 + u])
            if n:
                blk = delta[cur:cur + n]
                ink[blk[:, 0], blk[:, 1]] = blk[:, 2].astype(np.uint8)
            cur += n
            if u == t:
                ink_at_t = ink.copy()
        x = np.clip((V[g0 + t] - vmu) / vsd, -10, 10).astype(np.float32)
        j = np.minimum(np.arange(t, t + chunk), L - 1)
        yield x, np.stack([tmap, ink_at_t]), tip_px_from_vec(V[g0 + t]).astype(np.float32), hq.copy(), A[g0 + j].reshape(-1)
        if hist:
            hq = np.concatenate([hq[1:], x[None]], 0)


def batches(eps, V, A, vmu, vsd, batch, rng, device="cpu", buf_steps=2048, hist=0, chunk=1, maps_off=False):
    """Single-process replay: episodes in a random order through a shuffle buffer."""
    order = rng.permutation(len(eps))
    bx, bm, bt, bh, ba = [], [], [], [], []
    for e in order:
        for x, m, tp, h, a in episode_frames(eps[e], V, A, vmu, vsd, hist, chunk, maps_off):
            bx.append(x); bm.append(m); bt.append(tp); bh.append(h); ba.append(a)
            if len(bx) >= buf_steps:
                yield from _emit(bx, bm, bt, bh, ba, batch, rng, device)
                bx, bm, bt, bh, ba = [], [], [], [], []
    if bx:
        yield from _emit(bx, bm, bt, bh, ba, batch, rng, device)


class _Frames(torch.utils.data.IterableDataset):
    """The same replay split across DataLoader workers. Each worker takes every
    num_workers-th episode of a per-epoch permutation and shuffles through its own buffer;
    the loader interleaves workers, so a batch mixes episodes. The replay is pure Python,
    so a single process leaves the GPU mostly idle."""

    def __init__(self, eps, V, A, vmu, vsd, hist, chunk, seed, buf_steps=2048, maps_off=False):
        self.eps, self.V, self.A, self.vmu, self.vsd = eps, V, A, vmu, vsd
        self.hist, self.chunk, self.seed, self.buf_steps, self.epoch = hist, chunk, seed, buf_steps, 0
        self.maps_off = maps_off

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng(self.seed * 100003 + self.epoch * 1009 + wid)
        order = np.random.default_rng(self.seed * 7 + self.epoch).permutation(len(self.eps))[wid::nw]
        buf = []
        for e in order:
            for fr in episode_frames(self.eps[e], self.V, self.A, self.vmu, self.vsd, self.hist, self.chunk, self.maps_off):
                buf.append(fr)
                if len(buf) >= self.buf_steps:
                    for i in rng.permutation(len(buf)):
                        yield buf[i]
                    buf = []
        for i in rng.permutation(len(buf)):
            yield buf[i]


def batches_parallel(ds, batch, workers, device="cpu"):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, drop_last=False,
                                         pin_memory=(device != "cpu"), prefetch_factor=4, persistent_workers=False)
    for x, m, tp, h, a in loader:
        yield x.to(device), m.to(device), tp.to(device), h.to(device), a.to(device)


def _emit(bx, bm, bt, bh, ba, batch, rng, device="cpu"):
    X = torch.as_tensor(np.asarray(bx, np.float32))
    M = torch.as_tensor(np.asarray(bm, np.uint8))
    Tp = torch.as_tensor(np.asarray(bt, np.float32))
    H = torch.as_tensor(np.asarray(bh, np.float32))
    Y = torch.as_tensor(np.asarray(ba, np.float32))
    perm = torch.as_tensor(rng.permutation(len(X)))
    for i in range(0, len(X), batch):
        idx = perm[i:i + batch]
        yield X[idx].to(device), M[idx].to(device), Tp[idx].to(device), H[idx].to(device), Y[idx].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("out")
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256, help="image-sized observations: 1024 does not fit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Shards collected under deliberately different settings (e.g. DAgger data under another
    # constraint) must name the keys allowed to differ; the check below rejects any other mix.
    ap.add_argument("--seed", type=int, default=0, help="shuffle order and torch init; clones vary a lot run to run")
    ap.add_argument("--allow-mixed", default="", help="comma list of collection env keys allowed to differ across shards, e.g. constraint")
    ap.add_argument("--arch", default="conv", choices=["conv", "xf", "mlp", "traj", "traj2"], help="xf: + transformer over a tip-centred map crop, the state and its history")
    ap.add_argument("--hist", type=int, default=0, help="xf: past observation vectors in the history tokens")
    ap.add_argument("--crop", type=int, default=96, help="xf: side of the tip-centred map crop [px = mm]")
    ap.add_argument("--rows", default="", help="train only these output groups (comma list of arm,wrist,hand,done); "
                                              "the rest get no loss - the executor clones the arm and wrist, the grip stays frozen")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for the episode replay (0 = in-process)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="predict K consecutive actions per forward (action chunking); evaluate applies them in order")
    ap.add_argument("--weight-by-std", type=int, default=0,
                    help="1: weight each output dim by (median label std / its std)^2, clipped to [1, 100] - without it the arm's "
                         "steps (std 0.06) and the done flag vanish under the fingers (std 0.5): v12 clones fit the arm at R2 0.47")
    ap.add_argument("--norm-from-init", type=int, default=0,
                    help="normalise with --init's statistics instead of the data's (the transplanted grasp must see the statistics it was trained with)")
    ap.add_argument("--distill", default=None,
                    help="frozen grip checkpoint; the finger rows are regressed onto its outputs on the same frames, cursor blinded")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    args = ap.parse_args()
    # npz members are decompressed on EVERY access, and `batches` touches task_map / ink_delta
    # once per episode, which otherwise dominates an epoch, single-threaded.
    # Materialise what the replay loop reads (vec / act are concatenated
    # below anyway, so they stay lazy here).
    ds = []
    for _p in args.data:
        z = np.load(_p)
        d = {k: z[k] for k in ("ep_len", "task_map", "ink_n", "ink_delta") if k in z.files}
        if "meta" in z.files:
            d["meta"] = z["meta"]
        d["vec"], d["act"] = z["vec"], z["act"]
        if "traj" in z.files:
            d["traj"] = z["traj"]
        ds.append(d)
    # The env the data was collected in, straight from the shards. Inheriting
    # --init's env_kwargs instead would silently propagate whatever the
    # ancestor was trained under.
    metas = [json.loads(str(d["meta"])) for d in ds if "meta" in d]
    mixed_ok = {k for k in args.allow_mixed.split(",") if k}
    strict = [{k: v for k, v in m.items() if k not in mixed_ok} for m in metas]
    if strict and any(m != strict[0] for m in strict[1:]):
        differing = {k for m in strict[1:] for k in m if m[k] != strict[0].get(k)}
        raise SystemExit(f"datasets were collected under different envs, differing in {sorted(differing)}")
    mixed = {k: sorted({str(m.get(k)) for m in metas}) for k in mixed_ok if len({str(m.get(k)) for m in metas}) > 1}
    V = np.concatenate([d["vec"] for d in ds])
    if V.shape[1] < VEC_DIM:        # shards from before the external group grew (+7 brush entries): pad, as adapt_checkpoint does
        V = np.concatenate([V, np.zeros((len(V), VEC_DIM - V.shape[1]), V.dtype)], 1)
    maps_off = args.arch in TRAJ_ARCHS
    if maps_off:
        # what the network is handed - the kept state entries, then the trajectory window
        TJ = np.concatenate([d["traj"].reshape(len(d["traj"]), -1) for d in ds]).astype(np.float32)
        V = np.concatenate([V[:, STATE_KEEP], TJ], 1)
        print(f"traj packing: {V.shape[1]} entries ({len(STATE_KEEP)} state + {TJ.shape[1]} trajectory)", flush=True)
    A = np.clip(np.concatenate([d["act"] for d in ds]), -1, 1)
    eps = episodes(ds)
    print(f"{len(V):,} steps in {len(eps)} episodes", flush=True)
    # Statistics for the VECTOR group only. The maps are uint8 images the model
    # converts itself; per-pixel mean/var would whiten away the spatial structure the
    # conv exists to read.
    mu, var = V.mean(0), V.var(0) + 1e-8
    act_dim = A.shape[1]                       # 34, or 35 when the expert declared the end
    if args.init:
        ck = adapt_checkpoint(torch.load(args.init, map_location="cpu", weights_only=False))
        agent = Agent(len(mu), act_dim, hidden=ck.get("hidden", 256), min_logstd=ck.get("min_logstd", -3.0),
                      arch=args.arch, hist=args.hist, crop=args.crop, chunk=args.chunk)
        sd = ck["model"]
        n_old = sd["actor_mean.2.bias"].shape[0]
        if n_old < act_dim:                    # pad a 34-output checkpoint's head for the done entry
            pad = act_dim - n_old
            sd["actor_mean.2.weight"] = torch.cat([sd["actor_mean.2.weight"], torch.zeros(pad, sd["actor_mean.2.weight"].shape[1])])
            sd["actor_mean.2.bias"] = torch.cat([sd["actor_mean.2.bias"], torch.full((pad,), -1.0)])
            sd["actor_logstd"] = torch.cat([sd["actor_logstd"], torch.full((1, pad), -1.6)], 1)
        if args.chunk > 1 and sd["actor_mean.2.bias"].shape[0] == act_dim:
            for k in ("actor_mean.2.weight", "actor_mean.2.bias", "xf_head.weight", "xf_head.bias"):
                if k in sd:
                    sd[k] = sd[k].repeat(args.chunk, *([1] * (sd[k].dim() - 1)))
            sd["actor_logstd"] = sd["actor_logstd"].repeat(1, args.chunk)
            print(f"head tiled {args.chunk}x: every chunk step starts as the single-step head", flush=True)
        fresh = args.arch == "xf" and not any(k.startswith("xf") for k in sd)
        agent.load_state_dict(sd, strict=not fresh)          # an xf clone of a conv init: the branch starts fresh
        if fresh:
            print("transformer branch starts fresh (the init has none)", flush=True)
        scale = np.asarray(ck["action_scale"], np.float32)
        if len(scale) < act_dim:
            scale = np.concatenate([scale, np.ones(act_dim - len(scale), np.float32)])
        base = {k: ck[k] for k in ("tasks", "hidden", "min_logstd", "action_scale", "obs_mode",
                                   "control_mode", "control_hz", "env_kwargs") if k in ck}
    else:
        agent = Agent(len(mu), act_dim, arch=args.arch, hist=args.hist, crop=args.crop, chunk=args.chunk)
        scale = np.ones(act_dim, np.float32); base = {}
    agent = agent.to(args.device)
    A = A / np.maximum(scale, 1e-6)
    A = np.clip(A, -1, 1).astype(np.float32)
    vmu, vsd = mu, np.sqrt(var)
    if args.norm_from_init:
        mu, var = np.asarray(ck["obs_mean"], np.float64), np.asarray(ck["obs_var"], np.float64)
        vmu, vsd = mu, np.sqrt(var + 1e-8)                 # PPOPolicy's and train_ppo's convention
        print("normalising with the init's statistics", flush=True)
    teacher = None
    if args.distill:
        # The frozen grip sees the same frame with the TASK signal zeroed - the cursor slots in the
        # 167-entry layout, the trajectory window in the traj layout - normalised with its own statistics.
        # Those equal the student's everywhere else, so the teacher's input is the student's x with
        # that block replaced by the grip's normalised zero.
        blind = slice(150, 160) if not maps_off else slice(len(STATE_KEEP), V.shape[1])
        tk = adapt_checkpoint(torch.load(args.distill, map_location="cpu", weights_only=False))
        n_t = tk["model"]["actor_mean.2.bias"].shape[0]
        teacher = Agent(len(mu), n_t, hidden=tk.get("hidden", 256), arch=args.arch).to(args.device)
        teacher.load_state_dict(tk["model"]); teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        tmu, tvar = np.asarray(tk["obs_mean"], np.float64), np.asarray(tk["obs_var"], np.float64)
        other = np.r_[0:blind.start, blind.stop:len(mu)]
        assert np.allclose(tmu[other], mu[other]) and np.allclose(tvar[other], var[other], rtol=1e-4, atol=1e-8), \
            "the grip's statistics must equal the student's outside the task block (use --norm-from-init)"
        t_cur = torch.as_tensor(np.clip(-tmu[blind] / np.sqrt(tvar[blind] + 1e-8), -10, 10),
                                dtype=torch.float32, device=args.device)
        print(f"finger rows distilled onto {args.distill} (weight {args.distill_weight}), "
              f"its task block {blind.start}:{blind.stop} blinded", flush=True)
    w_dim = None
    if args.weight_by_std:
        sd = A.std(0)
        w_dim = torch.as_tensor(np.clip((np.median(sd) / np.maximum(sd, 1e-3)) ** 2, 1.0, 100.0), dtype=torch.float32, device=args.device)
        print("per-dim loss weights:", np.round(w_dim.cpu().numpy(), 1).tolist(), flush=True)
    row_mask = None
    if args.rows:
        groups = {"arm": range(0, 7), "wrist": range(7, 9), "hand": range(9, 34), "done": range(34, act_dim)}
        keep = np.zeros(act_dim, np.float32)
        for g in args.rows.split(","):
            keep[list(groups[g])] = 1.0
        row_mask = torch.as_tensor(np.tile(keep, args.chunk), dtype=torch.float32, device=args.device)
        print(f"loss on rows {args.rows} only ({int(keep.sum())} of {act_dim})", flush=True)
    opt = torch.optim.Adam([p for n, p in agent.named_parameters() if "critic" not in n], lr=args.lr)
    N = len(V)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    frames = _Frames(eps, V, A, vmu, vsd, args.hist, args.chunk, args.seed, maps_off=maps_off) if args.workers > 0 else None
    for ep in range(args.epochs):
        tot, seen, dtot = 0.0, 0, 0.0
        if frames is not None:
            frames.epoch = ep
            it = batches_parallel(frames, args.batch, args.workers, args.device)
        else:
            it = batches(eps, V, A, vmu, vsd, args.batch, rng, args.device, hist=args.hist, chunk=args.chunk, maps_off=maps_off)
        for xb, mb_, tb, hb, ab in it:
            pred = agent.mean(xb, mb_, tb, hb)
            w = torch.ones_like(pred).reshape(len(pred), args.chunk, -1)
            w[:, :, :7] = 4.0                                  # arm steps integrate; weight them
            if w.shape[2] > 34:
                w[:, :, 34:] = 4.0                             # ... and so does stopping at the right moment
            w = w.reshape(len(pred), -1)
            if w_dim is not None:
                w = w_dim.repeat(args.chunk).expand_as(pred)   # by label spread instead of the fixed 4x
            if row_mask is not None:
                w = w * row_mask
            loss = (w * (pred - ab) ** 2).mean()
            if teacher is not None:
                xt = xb.clone(); xt[:, blind] = t_cur
                with torch.no_grad():
                    ft = teacher.mean(xt, mb_, tb, hb)[:, 9:34]
                dl = ((pred[:, 9:34] - ft) ** 2).mean()
                loss = loss + args.distill_weight * dl
                dtot += float(dl.detach()) * len(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb); seen += len(xb)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  mse {tot / max(seen, 1):.4f}" + (f"  finger distill {dtot / max(seen, 1):.5f}" if teacher is not None else ""), flush=True)
    agent = agent.to("cpu")
    out = dict(base, model=agent.state_dict(), obs_mean=mu, obs_var=var, obs_dim=len(mu),
               arch=args.arch, hist=args.hist, crop=args.crop, chunk=args.chunk,
               iteration="BC(" + ",".join(os.path.basename(p) for p in args.data) + ")")
    if metas:
        m0 = metas[0]
        out["env_kwargs"] = m0
        out["env_kwargs_source"] = "collection"
        if mixed:
            out["env_kwargs_mixed"] = mixed          # shards differed here (first shard's value stamped above)
        # The control settings come from the data as well, so a checkpoint trained from
        # scratch has them too; without them PPOPolicy falls back to delta_pos at 100 Hz
        # and drops the pen.
        for k in ("obs_mode", "control_mode", "control_hz"):
            if k in m0:
                out[k] = m0[k]
    out.setdefault("action_scale", scale)      # from --init if it had one, else ones
    torch.save(out, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

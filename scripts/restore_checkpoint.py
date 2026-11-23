"""Convert full training checkpoints to/from a pickle-free format (safetensors + JSON).

A training checkpoint (.pt, written by scripts/train.py) holds the model weights, both
optimizer states (Muon momentum, AdamW moments), both LR schedulers, the step, RNG state and
the config. Raw .pt files are Python pickles, which can run code when loaded. This script
stores the same content as:

    model.safetensors       model weights
    optimizer.safetensors   every tensor inside the Muon and AdamW optimizer states
    rng.safetensors         CPU/CUDA RNG state
    trainer_state.json      step, scheduler states, optimizer scalars/param groups, config

and rebuilds a .pt that scripts/train.py --resume accepts, identical to the original.

    # export (with a bit-exact round-trip check)
    python scripts/restore_checkpoint.py export --pt step_22888.pt --out_dir phase2-step22888 --verify
    # restore, then resume training
    python scripts/restore_checkpoint.py restore --dir phase2-step22888 --pt step_22888.pt
    python scripts/train.py configs/phase2.yaml --resume step_22888.pt
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OPT_KEYS = ("muon_opt", "adamw_opt")
SCHED_KEYS = ("muon_sched", "adamw_sched")
TENSOR_TAG = "__tensor__"


def _split(obj, prefix, tensors):
    """Replace tensors in a nested structure with references, collecting them by name."""
    if isinstance(obj, torch.Tensor):
        tensors[prefix] = obj.detach().contiguous().clone()
        return {TENSOR_TAG: prefix}
    if isinstance(obj, dict):
        return {"__dict__": [[_split(k, f"{prefix}.k{i}", tensors), _split(v, f"{prefix}.{_key(k, i)}", tensors)]
                             for i, (k, v) in enumerate(obj.items())]}
    if isinstance(obj, (list, tuple)):
        return {"__list__" if isinstance(obj, list) else "__tuple__": [_split(v, f"{prefix}.{i}", tensors) for i, v in enumerate(obj)]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"unsupported type at {prefix}: {type(obj)}")


def _key(k, i):
    return str(k) if isinstance(k, (int, str)) else f"i{i}"


def _join(obj, tensors):
    if isinstance(obj, dict):
        if TENSOR_TAG in obj:
            return tensors[obj[TENSOR_TAG]]
        if "__dict__" in obj:
            return {_join(k, tensors): _join(v, tensors) for k, v in obj["__dict__"]}
        if "__list__" in obj:
            return [_join(v, tensors) for v in obj["__list__"]]
        if "__tuple__" in obj:
            return tuple(_join(v, tensors) for v in obj["__tuple__"])
    return obj


def export(pt_path, out_dir):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck["model"].items()}, str(out / "model.safetensors"), metadata={"step": str(ck["step"])})
    opt_t, rng_t, state = {}, {}, {"step": ck["step"], "config": ck.get("config"), "keys": sorted(ck.keys())}
    for k in OPT_KEYS:
        state[k] = _split(ck[k], k, opt_t)
    for k in SCHED_KEYS:
        state[k] = _split(ck[k], k, opt_t)  # schedulers are tensor-free in practice
    state["rng"] = _split({"rng_state": ck.get("rng_state"), "cuda_rng_state": ck.get("cuda_rng_state")}, "rng", rng_t)
    save_file(opt_t, str(out / "optimizer.safetensors"))
    save_file(rng_t, str(out / "rng.safetensors"))
    (out / "trainer_state.json").write_text(json.dumps(state, indent=1))
    return ck


def restore(dir_path):
    d = Path(dir_path)
    state = json.loads((d / "trainer_state.json").read_text())
    opt_t = load_file(str(d / "optimizer.safetensors"))
    rng_t = load_file(str(d / "rng.safetensors"))
    ck = {"model": load_file(str(d / "model.safetensors")), "step": state["step"], "config": state["config"]}
    for k in OPT_KEYS + SCHED_KEYS:
        ck[k] = _join(state[k], opt_t)
    ck.update(_join(state["rng"], rng_t))
    return ck


def _same(a, b, path="ck"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--pt", required=True); e.add_argument("--out_dir", required=True); e.add_argument("--verify", action="store_true")
    r = sub.add_parser("restore"); r.add_argument("--dir", required=True); r.add_argument("--pt", required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        orig = export(args.pt, args.out_dir)
        if args.verify:
            _same(orig, restore(args.out_dir))
            print(f"[verify] round-trip identical: every tensor and value in {args.pt} matches the export")
        print(f"[export] {args.pt} -> {args.out_dir}")
    else:
        torch.save(restore(args.dir), args.pt)
        print(f"[restore] {args.dir} -> {args.pt} (use with scripts/train.py --resume)")


if __name__ == "__main__":
    main()

import argparse
import yaml

import jax
import numpy as np

from hafqmc.utils import load_pickle


def _extract_params(obj):
    if isinstance(obj, tuple):
        if len(obj) == 2 and isinstance(obj[1], tuple) and len(obj[1]) > 1:
            return obj[1][1]
        if len(obj) > 1:
            return obj[1]
    return obj


def _summarize_tree(tree, prefix=""):
    leaves = jax.tree_util.tree_leaves(tree)
    print(f"{prefix}num_leaves: {len(leaves)}")
    for i, leaf in enumerate(leaves[:10]):
        if hasattr(leaf, "shape"):
            print(f"{prefix}leaf[{i}] shape={leaf.shape} dtype={leaf.dtype}")
    if len(leaves) > 10:
        print(f"{prefix}... ({len(leaves)-10} more leaves)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="Path to params .pkl")
    parser.add_argument("--hparams", default=None, help="Path to hparams.yml (optional)")
    args = parser.parse_args()

    params_raw = load_pickle(args.params)
    params = _extract_params(params_raw)
    print(f"# params_raw type: {type(params_raw)}")
    print(f"# params type: {type(params)}")

    if isinstance(params, dict):
        keys = list(params.keys())
        print(f"# params keys: {keys}")
        params_inner = params.get("params", params)
    else:
        params_inner = params

    if isinstance(params_inner, dict):
        ansatz_params = params_inner.get("ansatz", params_inner)
        wfn_a = ansatz_params.get("wfn_a", None) if isinstance(ansatz_params, dict) else None
        wfn_b = ansatz_params.get("wfn_b", None) if isinstance(ansatz_params, dict) else None
        wfn = ansatz_params.get("wfn", None) if isinstance(ansatz_params, dict) else None
        print(f"# has wfn_a: {wfn_a is not None}, wfn_b: {wfn_b is not None}, wfn: {wfn is not None}")
        if wfn_a is not None:
            print(f"# wfn_a shape: {np.array(wfn_a).shape}")
        if wfn_b is not None:
            print(f"# wfn_b shape: {np.array(wfn_b).shape}")
        if wfn is not None:
            print(f"# wfn shape: {np.array(wfn).shape}")
    else:
        print("# params_inner is not a dict; cannot locate wfn_a/wfn_b")

    print("# tree summary:")
    _summarize_tree(params_inner, prefix="  ")

    if args.hparams:
        with open(args.hparams, "r") as fh:
            hparams = yaml.safe_load(fh)
        ansatz = hparams.get("ansatz", {})
        print("# hparams ansatz:")
        for k in ("wfn_param", "wfn_spinmix", "wfn_complex", "propagators"):
            print(f"  {k}: {ansatz.get(k)}")


if __name__ == "__main__":
    main()

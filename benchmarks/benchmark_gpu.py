#!/usr/bin/env python
"""CPU -> GPU speedup benchmark for pycdft.

Compares the numpy backend (1 thread / N threads) with the torch backend on
CPU and on the best available device (cuda/rocm/mps).  Synthetic grid case:
(T_cal, N) calibration and (T_tar, N) target, per-cell CDF-t.

Run:
    python benchmarks/benchmark_gpu.py [--t-cal 10000] [--t-tar 5000] [--n-cells 50000]
"""
import argparse
import os
import sys
import time

import numpy as np

# runnable from anywhere: the package root is the parent of benchmarks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycdft import CDFt
from pycdft.gpu import CDFt as CDFtGPU
from pycdft.gpu.devices import available_devices


def make_data(t_cal, t_tar, n, seed=0):
    r = np.random.default_rng(seed)
    obs = r.normal(280.0, 5.0, (t_cal, n)).astype(np.float32)
    cm_hist = obs + r.normal(0.0, 0.6, (t_cal, n)).astype(np.float32)
    cm_future = r.normal(281.0, 5.5, (t_tar, n)).astype(np.float32)
    return obs, cm_hist, cm_future


def timeit(fn, reps=2):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-cal", type=int, default=10000)
    ap.add_argument("--t-tar", type=int, default=5000)
    ap.add_argument("--n-cells", type=int, default=50000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-quantiles", type=int, default=1001)
    args = ap.parse_args()

    obs, cm_hist, cm_future = make_data(args.t_cal, args.t_tar, args.n_cells)
    print(f"data: T_cal={args.t_cal}, T_tar={args.t_tar}, N={args.n_cells} "
          f"cells, Q={args.n_quantiles}")

    rows = []

    def run_numpy(workers):
        m = CDFt(n_quantiles=args.n_quantiles, tail="linear",
                 delta_shift="additive", workers=workers)
        m.fit(obs, cm_hist)
        t = timeit(lambda: m.transform(cm_future), args.reps)
        return t

    for w in (1, 16):
        t = run_numpy(w)
        rows.append((f"numpy (workers={w})", t))
        print(f"{rows[-1][0]:28s} {t:8.3f} s")

    dev = available_devices()
    print("devices available to torch:", dev)
    for d in (["cpu"] if "cuda" not in dev and "rocm" not in dev else
              ["cpu", "cuda"]):
        try:
            m = CDFtGPU(n_quantiles=args.n_quantiles, tail="linear",
                        delta_shift="additive", device=d)
            m.fit(obs, cm_hist)
            t = timeit(lambda: m.transform(cm_future), args.reps)
            rows.append((f"torch ({d})", t))
            print(f"{rows[-1][0]:28s} {t:8.3f} s")
        except Exception as e:                       # pragma: no cover
            print(f"torch ({d}): skipped ({e})")

    base = rows[0][1]
    print("\n--- speedups vs numpy (workers=1) ---")
    for name, t in rows:
        print(f"{name:28s} {t:8.3f} s   {base/t:6.2f}x")


if __name__ == "__main__":
    main()

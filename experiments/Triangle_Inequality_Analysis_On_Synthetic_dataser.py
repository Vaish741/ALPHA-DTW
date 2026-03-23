# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:33:35 2026

@author: vaish
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from itertools import combinations, product
from typing import Callable

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, desc=""):
        return iterable


"""
Triangle inequality analysis using Cython implementations of ACDTW,
DTW (alpha = 0), and Alpha-DTW.

- Evaluates violation rates on synthetic datasets.
- Supports equilateral, isosceles, and scalene triplets.
- Uses compiled Cython modules when available.
- Falls back to Python version if needed.

All computations preserve the original algorithmic logic.
"""


# ------------------------------------------------
# Path setup
# ------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BUILD_DIRS = sorted((ROOT / "build").glob("lib.*"))
ACDTW_DIRS = [ROOT / "ACDTW", ROOT / "src" / "ACDTW"]


def _prepend(path: Path):
    if path.exists():
        p = str(path)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _setup_paths():
    for p in BUILD_DIRS:
        _prepend(p)

    _prepend(ROOT)

    for p in ACDTW_DIRS:
        _prepend(p)


# ------------------------------------------------
# Synthetic dataset generation
# ------------------------------------------------
def generate_white_noise(n=50, length=50):
    return np.random.normal(0, 1, (n, length))


def generate_random_walk(n=50, length=50):
    X = np.zeros((n, length))
    for i in range(1, length):
        X[:, i] = X[:, i - 1] + np.random.normal(0, 1, n)
    return X


def generate_sinusoidal(n=50, length=50):
    t = np.arange(length)
    omega = (4 * np.pi * t) / (length - 1)

    return np.array([
        np.sin(omega + np.random.uniform(0, 2 * np.pi))
        + np.random.normal(0, np.sqrt(0.2), length)
        for _ in range(n)
    ])


# Dataset 2
def generate_type_a(m, eps, L, l, n=50):
    curves = []
    even_idx = [2 * i for i in range(1, 2 * m)]
    length = 4 * m + 1

    for _ in range(n):
        c = np.zeros(length)
        peaks = np.random.choice(even_idx, l, replace=False)

        for i in even_idx:
            c[i] = L if i in peaks else np.random.uniform(0, eps)

        curves.append(c)

    return np.array(curves)


def generate_type_b(m, eps, L, n=50):
    length = 4 * m + 1
    curves = []

    for _ in range(n):
        c = np.zeros(length)
        for i in range(1, length - 1):
            c[i] = L + np.random.uniform(0, eps) if i % 2 == 0 else L
        curves.append(c)

    return np.array(curves)


def generate_type_c(m, eps, n=50):
    length = 4 * m + 1
    curves = []

    for _ in range(n):
        c = np.zeros(length)
        for i in range(2, length - 1, 2):
            c[i] = np.random.uniform(0, eps)
        curves.append(c)

    return np.array(curves)


# ------------------------------------------------
# Triplets
# ------------------------------------------------
def get_equilateral(y):
    return list(set(sum([list(combinations(np.where(y == c)[0], 3)) for c in np.unique(y)], [])))


def get_isosceles(y):
    triplets = []
    classes = np.unique(y)

    for c1 in classes:
        idx1 = np.where(y == c1)[0]
        for c2 in classes:
            if c1 == c2:
                continue
            idx2 = np.where(y == c2)[0]

            for p in combinations(idx1, 2):
                for k in idx2:
                    triplets.append(tuple(sorted([p[0], p[1], k])))

    return list(set(triplets))


def get_scalene(y):
    classes = np.unique(y)
    if len(classes) < 3:
        return []

    triplets = []
    for cset in combinations(classes, 3):
        idxs = [np.where(y == c)[0] for c in cset]
        for t in product(*idxs):
            triplets.append(tuple(sorted(t)))

    return list(set(triplets))


# ------------------------------------------------
# Triangle looseness
# ------------------------------------------------
def looseness(d_ij, d_jk, d_ik):
    vals = [d_ij + d_jk - d_ik,
            d_ik + d_jk - d_ij,
            d_ik + d_ij - d_jk]

    neg = [v for v in vals if v < 0]
    return min(neg) if neg else 0


# ------------------------------------------------
# Distance loading (Cython)
# ------------------------------------------------
def load_acdtw():
    _setup_paths()
    mod = importlib.import_module("acdtw")
    return mod.acdtw_equal_len_paper


def load_alpha(alpha):
    _setup_paths()
    mod = importlib.import_module("alpha_dtw")

    return lambda a, b: float(
        mod.alpha_dtw_distance(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            float(alpha),
        )
    )


# ------------------------------------------------
# Triangle evaluation
# ------------------------------------------------
def evaluate(X, triplets, dist, name):
    v = 0
    total = len(triplets)

    print(f"{name}: {total} triplets")

    for i, j, k in tqdm(triplets, desc=name):
        d_ij = dist(X[i], X[j])
        d_jk = dist(X[j], X[k])
        d_ik = dist(X[i], X[k])

        if looseness(d_ij, d_jk, d_ik) < 0:
            v += 1

    rate = v / total
    print(f"{name}: {v}/{total} ({100 * rate:.4f}%)")
    return rate


# ------------------------------------------------
# Experiments
# ------------------------------------------------
SEEDS = [1, 15, 20, 38, 40, 45, 53, 68, 75, 86]


def run_dataset_1(dist_name, dist):
    print("\nDataset 1:", dist_name)
    stats = {"equilateral": [], "isosceles": [], "scalene": []}

    for s in SEEDS:
        np.random.seed(s)

        X = np.vstack([
            generate_white_noise(),
            generate_random_walk(),
            generate_sinusoidal()
        ])

        y = np.repeat([0, 1, 2], 50)

        triplets = {
            "equilateral": get_equilateral(y),
            "isosceles": get_isosceles(y),
            "scalene": get_scalene(y)
        }

        for t in triplets:
            stats[t].append(evaluate(X, triplets[t], dist, t))

    print("\nMean:")
    for k in stats:
        print(k, np.mean(stats[k]) * 100)


def run_dataset_2(dist_name, dist):
    print("\nDataset 2:", dist_name)
    stats = {"equilateral": [], "isosceles": [], "scalene": []}

    for s in SEEDS:
        np.random.seed(s)

        X = np.vstack([
            generate_type_a(10, 0.5, 10, 10),
            generate_type_b(10, 0.5, 10),
            generate_type_c(10, 0.5)
        ])

        y = np.repeat([0, 1, 2], 50)

        triplets = {
            "equilateral": get_equilateral(y),
            "isosceles": get_isosceles(y),
            "scalene": get_scalene(y)
        }

        for t in triplets:
            stats[t].append(evaluate(X, triplets[t], dist, t))

    print("\nMean:")
    for k in stats:
        print(k, np.mean(stats[k]) * 100)


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    print("1: ACDTW | 2: DTW | 3: Alpha-DTW")
    c = input("Choice: ").strip()

    if c == "1":
        dist = load_acdtw()
        name = "ACDTW"
    elif c == "2":
        dist = load_alpha(0.0)
        name = "DTW"
    else:
        dist = load_alpha(0.5)
        name = "Alpha-DTW"

    run_dataset_1(name, dist)
    run_dataset_2(name, dist)


if __name__ == "__main__":
    main()
from __future__ import annotations

from itertools import combinations, product
from typing import Callable
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, desc=""):
        return iterable


from src.cython import alpha_dtw, acdtw


"""
Triangle-inequality violation analysis for ACDTW, DTW, and Alpha-DTW.

1) Synthetic datasets (Dataset 1 + Dataset 2)
2) A real UCR dataset (train+test combined)


"""


# ------------------------------------------------
# Synthetic dataset generation
# Dataset 1
# ------------------------------------------------
def d1_class_1(n=50, length=50):
    return np.random.normal(0, 1, (n, length))


def d1_class_2(n=50, length=50):
    X = np.zeros((n, length))
    for i in range(1, length):
        X[:, i] = X[:, i - 1] + np.random.normal(0, 1, n)
    return X


def d1_class_3(n=50, length=50):
    t = np.arange(1, length + 1)
    omega = (4 * np.pi * t) / (length - 1)

    return np.array([
        np.sin(omega + np.random.uniform(0, 2 * np.pi))
        + np.random.normal(0, np.sqrt(0.2), length)
        for _ in range(n)
    ])


# ------------------------------------------------
# Synthetic dataset generation
# Dataset 2
# ------------------------------------------------
def d2_class_1(m, eps, L, l, n=50):
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


def d2_class_2(m, eps, L, n=50):
    length = 4 * m + 1
    curves = []

    for _ in range(n):
        c = np.zeros(length)
        for i in range(1, length - 1):
            c[i] = L + np.random.uniform(0, eps) if i % 2 == 0 else L
        curves.append(c)

    return np.array(curves)


def d2_class_3(m, eps, n=50):
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
    return list(sum([list(combinations(np.where(y == c)[0], 3)) for c in np.unique(y)], []))


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
        print("scalene: skipped (fewer than 3 classes)")
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
    """
    Measure triangle inequality violation.

    Returns the minimum negative deviation from the triangle inequality.
    A negative value indicates a violation.
    """
    vals = [
        d_ij + d_jk - d_ik,
        d_ik + d_jk - d_ij,
        d_ik + d_ij - d_jk,
    ]

    neg = [v for v in vals if v < 0]
    return min(neg) if neg else 0


# ------------------------------------------------
# Distance functions 
# ------------------------------------------------
def acdtw_distance(a, b):
    return float(
        acdtw.acdtw_equal_len_paper(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
        )
    )


def alpha_distance(alpha):
    return lambda a, b: float(
        alpha_dtw.alpha_dtw_distance(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            float(alpha),
        )
    )


# ------------------------------------------------
# Triangle Inequality evaluation 
# ------------------------------------------------
def evaluate(X, triplets, dist, name):
    """
  Evaluate triangle inequality violations.

  Parameters
  ----------
  X : np.ndarray
      Time series dataset
  triplets : list
      Triplets of indices (i, j, k)
  dist : callable
      Distance function
  name : str
      Type of triplet (equilateral, isosceles, scalene)

  Returns
  -------
  float
      Violation rate
  """
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
# Synthetic experiments
# ------------------------------------------------
SEEDS = [1, 15, 20, 38, 40, 45, 53, 68, 75, 86]


def run_dataset_1(name, dist):
    print("\nDataset 1:", name)
    stats = {"equilateral": [], "isosceles": [], "scalene": []}

    for s in SEEDS:
        np.random.seed(s)

        X = np.vstack([
            d1_class_1(),
            d1_class_2(),
            d1_class_3()
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


def run_dataset_2(name, dist):
    print("\nDataset 2:", name)
    stats = {"equilateral": [], "isosceles": [], "scalene": []}

    for s in SEEDS:
        np.random.seed(s)

        X = np.vstack([
            d2_class_1(10, 0.5, 10, 10),
            d2_class_2(10, 0.5, 10),
            d2_class_3(10, 0.5)
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
# UCR loader
# ------------------------------------------------
def load_ucr_dataset(root: str, name: str):
    folder = f"{root}/{name}"
    train = np.genfromtxt(f"{folder}/{name}_TRAIN.tsv")
    test = np.genfromtxt(f"{folder}/{name}_TEST.tsv")

    y_train = train[:, 0].astype(int)
    X_train = train[:, 1:]

    y_test = test[:, 0].astype(int)
    X_test = test[:, 1:]

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    return X, y


def run_ucr_dataset(distance_name, distance_func, dataset_root, dataset_name):
    print(f"\n=== UCR Dataset: {dataset_name} ===")
    print(f"Distance: {distance_name}")

    X, y = load_ucr_dataset(dataset_root, dataset_name)
    classes = np.unique(y)

  
    triplets = {
        "equilateral": get_equilateral(y),
        "isosceles": get_isosceles(y),
    }

    # scalene only if >= 3 classes
    if len(classes) < 3:
        print("Scalene Triplets not exist (Classes less than 3)")
    else:
        triplets["scalene"] = get_scalene(y)

    for t in triplets:
        
        evaluate(X, triplets[t], distance_func, t)



# ------------------------------------------------
# Distance choice
# ------------------------------------------------
def choose_distance() -> tuple[str, Callable[[np.ndarray, np.ndarray], float]]:
    print("Choose the distance:")
    print("1. ACDTW")
    print("2. DTW (alpha = 0)")
    print("3. Alpha-DTW (alpha = 0.5)")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        return "acdtw", acdtw_distance
    if choice == "2":
        return "dtw", alpha_distance(0.0)

    return "alpha_dtw", alpha_distance(0.5)


# ------------------------------------------------
# Main 
# ------------------------------------------------
def main():
    distance_name, distance_func = choose_distance()

    print("\nChoose dataset type:")
    print("1. Synthetic (Dataset 1 + Dataset 2)")
    print("2. UCR Dataset")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        run_dataset_1(distance_name, distance_func)
        run_dataset_2(distance_name, distance_func)
    else:
        dataset_root = input("Enter UCR dataset root path: ").strip()
        dataset_name = input("Enter dataset name (e.g., ECG200): ").strip()
        run_ucr_dataset(distance_name, distance_func, dataset_root, dataset_name)


if __name__ == "__main__":
    main()
    
    
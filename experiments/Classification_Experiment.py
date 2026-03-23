# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:21:11 2026

@author: vaish
"""

"""
Cython-based Experimental Evaluation for DTW, ACDTW, and Alpha-DTW
------------------------------------------------------------------

This script evaluates 1-NN classification performance on UCR datasets
using optimized Cython implementations.

Algorithms:
    1. ACDTW
    2. Classical DTW (Alpha-DTW with alpha = 0)
    3. Alpha-DTW (with tuned alpha)

Notes
-----
- Uses Leave-One-Out Cross Validation (LOOCV) to tune alpha.
- Distance computations are performed using compiled Cython modules.
- Input time series are converted to contiguous float64 arrays.

Author: Vaishnavi Rastogi
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

# Cython modules (compiled .pyd files must be available)
from src.cython import alpha_dtw
from src.cython import acdtw


DATASETS = [
    "Beef", "CBF", "Coffee", "ECG200",
    "FaceFour", "Fish", "GunPoint", "Lightning2", "Lightning7",
    "OliveOil", "ShapeletSim", "Symbols", "SwedishLeaf",
    "SyntheticControl", "Trace", "TwoLeadECG",
]


# ------------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------------
def load_dataset(root: str, name: str):
    """
    Load a UCR dataset from disk.

    Parameters
    ----------
    root : str
        Root directory containing dataset folders.
    name : str
        Dataset name.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    folder = Path(root) / name

    train = np.genfromtxt(folder / f"{name}_TRAIN.tsv")
    test = np.genfromtxt(folder / f"{name}_TEST.tsv")

    y_train = train[:, 0]
    X_train = train[:, 1:]
    y_test = test[:, 0]
    X_test = test[:, 1:]

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------
# Distance Wrappers (Cython)
# ------------------------------------------------------------------
def alpha_dtw_distance(a, b, alpha: float) -> float:
    """
    Wrapper for Cython Alpha-DTW.

    Ensures correct dtype and memory layout.
    """
    return float(
        alpha_dtw.alpha_dtw_distance(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            float(alpha),
        )
    )


def acdtw_distance(a, b) -> float:
    """
    Wrapper for Cython ACDTW.
    """
    return float(
        acdtw.acdtw_equal_len_paper(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
        )
    )


# ------------------------------------------------------------------
# 1-NN Classifier
# ------------------------------------------------------------------
def run_1nn(X_train, y_train, X_test, y_test, distance_fn):
    """
    Perform 1-NN classification.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    distance_fn : callable
        Distance function

    Returns
    -------
    float
        Classification accuracy
    """
    preds = []

    for test in X_test:
        best_dist = np.inf
        best_label = None

        for train, label in zip(X_train, y_train):
            d = distance_fn(test, train)

            if d < best_dist:
                best_dist = d
                best_label = label

        preds.append(best_label)

    return accuracy_score(y_test, preds)


# ------------------------------------------------------------------
# Alpha tuning using LOOCV
# ------------------------------------------------------------------
def tune_alpha(X_train, y_train):
    """
    Tune alpha parameter using Leave-One-Out Cross Validation.

    Returns
    -------
    float
        Best alpha value
    """
    alphas = np.logspace(-5, 0, 100)
    loo = LeaveOneOut()

    best_alpha = None
    best_acc = -1.0

    for alpha in alphas:
        preds = []
        labels = []

        for train_idx, val_idx in loo.split(X_train):
            val = X_train[val_idx[0]]

            best_dist = np.inf
            best_label = None

            for i in train_idx:
                d = alpha_dtw_distance(val, X_train[i], alpha)

                if d < best_dist:
                    best_dist = d
                    best_label = y_train[i]

            preds.append(best_label)
            labels.append(y_train[val_idx[0]])

        acc = accuracy_score(labels, preds)
        print(f"alpha={alpha:.6f} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    print("\nBest alpha:", best_alpha)
    return best_alpha


# ------------------------------------------------------------------
# Main Experiment Pipeline
# ------------------------------------------------------------------
def main():
    dataset_root = input("Enter dataset path: ").strip()

    print("\nChoose algorithm")
    print("1 : ACDTW")
    print("2 : DTW")
    print("3 : Alpha-DTW")

    choice = input("Choice: ").strip()

    results = {}

    for dataset in DATASETS:
        print("\nDataset:", dataset)

        X_train, y_train, X_test, y_test = load_dataset(dataset_root, dataset)

        if choice == "1":
            acc = run_1nn(
                X_train, y_train, X_test, y_test,
                acdtw_distance
            )

        elif choice == "2":
            acc = run_1nn(
                X_train, y_train, X_test, y_test,
                lambda a, b: alpha_dtw_distance(a, b, 0.0),
            )

        else:
            alpha = tune_alpha(X_train, y_train)

            acc = run_1nn(
                X_train, y_train, X_test, y_test,
                lambda a, b: alpha_dtw_distance(a, b, alpha),
            )

        results[dataset] = acc
        print("Accuracy:", acc)

    print("\nFinal Results\n")
    for k, v in results.items():
        print(f"{k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
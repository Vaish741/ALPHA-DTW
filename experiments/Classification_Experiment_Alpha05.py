# -*- coding: utf-8 -*-
"""
1-NN classification with Alpha-DTW (fixed alpha = 0.5) on UCR datasets.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

from src.cython import alpha_dtw


DATASETS = [
    "Adiac", "Beef", "CBF", "CinCECGTorso", "Coffee", "Computers", "ECG200",
    "FaceAll", "FaceFour", "Fish", "GunPoint", "Lightning2", "Lightning7",
    "OliveOil", "Rock", "ShapeletSim", "Symbols", "SwedishLeaf",
    "SyntheticControl", "Trace", "TwoLeadECG",
]


def load_dataset(root: str, name: str):
    folder = Path(root) / name
    train = np.genfromtxt(folder / f"{name}_TRAIN.tsv")
    test = np.genfromtxt(folder / f"{name}_TEST.tsv")

    y_train = train[:, 0]
    X_train = train[:, 1:]
    y_test = test[:, 0]
    X_test = test[:, 1:]

    return X_train, y_train, X_test, y_test


def alpha_dtw_distance(a, b, alpha: float = 0.5) -> float:
    return float(
        alpha_dtw.alpha_dtw_distance(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            float(alpha),
        )
    )


def run_1nn(X_train, y_train, X_test, y_test):
    preds = []
    for test in X_test:
        best_dist = np.inf
        best_label = None
        for train, label in zip(X_train, y_train):
            d = alpha_dtw_distance(test, train, alpha=0.5)
            if d < best_dist:
                best_dist = d
                best_label = label
        preds.append(best_label)
    return accuracy_score(y_test, preds)


def main():
    dataset_root = input("Enter dataset path: ").strip()
    results = {}

    for dataset in DATASETS:
        print("\nDataset:", dataset)
        X_train, y_train, X_test, y_test = load_dataset(dataset_root, dataset)
        acc = run_1nn(X_train, y_train, X_test, y_test)
        results[dataset] = acc
        print("Accuracy:", acc)

    print("\nFinal Results\n")
    for k, v in results.items():
        print(f"{k:20s} {v:.4f}")


if __name__ == "__main__":
    main()

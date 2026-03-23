# ALPHA-DTW 

This repository contains Cython and Python implementations of Alpha-DTW and ACDTW,
along with reproducible experiments for 1-NN classification on UCR datasets and
triangle-inequality analysis on synthetic datsets

## Repository Layout

- `src/cython/` Cython implementations and build script (`alpha_dtw.pyx`, `acdtw.pyx`)
- `src/python/` pure-Python reference implementations
- `experiments/`
- `Dataset/` UCR-format datasets used in experiments


## Requirements

- Python 3.9
- numpy
- scipy
- scikit-learn
- cython


Install:

```bash
pip install numpy scipy scikit-learn cython tqdm
```

## Build Cython Extensions

From the repository root:

```bash
cd src/cython
python setup.py build_ext --inplace
```

This builds `alpha_dtw` and `acdtw` Cython extensions.

## Data

UCR datasets are expected in:

```
Dataset/<DATASET_NAME>/<DATASET_NAME>_TRAIN.tsv
Dataset/<DATASET_NAME>/<DATASET_NAME>_TEST.tsv
```

The included `Dataset/` folder already follows this structure.

## Experiments

### 1-NN Classification (UCR)

```bash
python experiments/Classification_Experiment.py
```

- Choose algorithm: ACDTW, DTW (alpha=0), or Alpha-DTW (alpha tuned by LOOCV).
- Enter dataset root path when prompted (e.g., `C:\Users\vaish\ALPHA-DTW\Dataset`).


### Triangle Inequality Analysis (Synthetic)

```bash
python experiments/Triangle_Inequality_Analysis_On_Synthetic_dataser.py
```

- Evaluates equilateral, isosceles, and scalene triplets.
- Averages results across fixed random seeds.




```bash
python experiments/Classification_Experiment.py > results/classification.txt
python experiments/Triangle_Inequality_Analysis_On_Synthetic_dataser.py > results/triangle.txt
```

## Reproducibility Notes

- Random seeds are fixed inside the experiment scripts.
- Alpha is tuned with LOOCV over `np.logspace(-5, 0, 100)`.
- DTW is Alpha-DTW with `alpha = 0`.


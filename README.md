# ALPHA-DTW
**Alpha-DTW** is an extension of Dynamic Time Warping (DTW) that introduces an additional penalty which takes into account the variation in the values within each time series at the aligned indices. The penalty is controlled by a parameter alpha. The method provides a controllable trade-off between DTW  and Euclidean distance.
This repository provides Python implementations of **Alpha‑DTW**, **DTW**, and **ACDTW** (Cython implementation is also available for fast computation).
The repository includes the following experiments:
- **1‑NN classification** on UCR datasets  
- **Triangle‑inequality violation analysis** on datasets.


---

## 1. Repository Structure

```
ALPHA-DTW/
│
├── src/
│   ├── cython/        # Cython implementations (alpha_dtw.pyx, acdtw.pyx)
│   └── python/        # Python reference implementations
│
├── experiments/
│   ├── Classification_Experiment.py
│   └── Triangle_Inequality_Analysis_On_Dataset.py
│
├── Dataset/           # UCR datasets
├── requirements.txt
└── README.md
```

---

## 2. Requirements

- Python 3.9  
- numpy  
- scipy  
- scikit-learn  
- cython  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Build Cython Extensions

From the repository root:

```bash
cd src/cython
python setup.py build_ext --inplace
```

This compiles:

- `alpha_dtw`  
- `acdtw`

**Note:** Cython extensions must be built before running experiments.

---

## 4. Dataset Format (UCR)

The classification experiment expects datasets in **UCR format**:
 
```
Dataset/<DATASET_NAME>/<DATASET_NAME>_TRAIN.tsv
Dataset/<DATASET_NAME>/<DATASET_NAME>_TEST.tsv
```

Example:

```
Dataset/ECG200/ECG200_TRAIN.tsv
Dataset/ECG200/ECG200_TEST.tsv
```

---
The experiments use datasets from the **UCR Time Series Archive**:

Detailed dataset descriptions are provided here (UCR Archive):  
https://www.cs.ucr.edu/~eamonn/time_series_data_2018/


## 5. Experiments

### 5.1 1‑NN Classification (UCR)

```bash
python experiments/Classification_Experiment.py
```


1. Choose algorithm:
   - ACDTW  
   - DTW (α = 0)  
   - Alpha‑DTW (α tuned by LOOCV)

2. Enter dataset root path (example):

```
C:\Users\vaish\ALPHA-DTW\Dataset
```

---

### 5.2 Triangle Inequality Analysis (Synthetic)

```bash
python experiments/Triangle_Inequality_Analysis_On_Dataset.py
```

This script generates two synthetic datasets and computes violation rates for:

- Equilateral triplets  
- Isosceles triplets  
- Scalene triplets  

Results are averaged over fixed random seeds:

```
SEEDS = [1, 15, 20, 38, 40, 45, 53, 68, 75, 86]
```

---
The analysis has also been performed on real datasets, including ECG200, Lightning7, and GunPoint.

##  6. Reproducibility Notes

- **Alpha** is tuned using LOOCV on the training set:
  ```
  α ∈ logspace(-5, 0, 100)
  ```
- **DTW** is equivalent to **Alpha‑DTW with α = 0**.
- Synthetic experiments use fixed seeds.
- In this study, all datasets chosen from the UCR archive  have time series of equal length.

---



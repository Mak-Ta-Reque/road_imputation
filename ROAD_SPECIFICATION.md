# ROAD Benchmark Pipeline: Inventory & Extensibility

## Datasets Supported
- **cifar10**
- **cifar100**
- **food101**
- **imagenet**

You can add new datasets by updating `experiments/unified/data.py` and registering them in `DATASET_CONFIGS`.

---

## Explanation Methods Available
From `experiments/unified/explanations.py` and pipeline defaults:
- **ig**: Integrated Gradients
- **gb**: Guided Backpropagation
- **ig_sg**: Integrated Gradients + SmoothGrad
- **gb_sg**: Guided Backpropagation + SmoothGrad
- **ig_sq**: Integrated Gradients + SmoothGrad-Squared
- **gb_sq**: Guided Backpropagation + SmoothGrad-Squared
- **ig_var**: Integrated Gradients + VarGrad
- **gb_var**: Guided Backpropagation + VarGrad

You can add new explanation methods by implementing them in `experiments/unified/explanations.py` and updating the `get_explanation_method` function.

---

## Imputation Methods Available
From pipeline defaults and arguments:
- **linear**
- **telea**
- **ns**
- **fixed** (appears in pipeline, not in shell script)

To add new imputation methods, implement them in `experiments/unified/imputations.py`.

---

## Ranking Methods Available
From pipeline defaults and arguments:
- **sort**
- **threshold**

To add new ranking methods, implement them in the relevant ranking logic (likely in `experiments/unified/`).

---

## Pipeline Stages
- **train**
- **explain**
- **benchmark**
- **analyze**
- **all** (runs all stages)

---

## How to Extend
- **New Dataset**: Add loader/class in `data.py`, register in `DATASET_CONFIGS`.
- **New Explanation Method**: Implement in `explanations.py`, add to `get_explanation_method`.
- **New Imputation/Ranking**: Implement in `imputations.py` or relevant file, add to CLI defaults if needed.

---

## Summary Table

| Category         | Available Options                                      | Extensible? | Location                                 |
|------------------|-------------------------------------------------------|-------------|-------------------------------------------|
| Datasets         | cifar10, cifar100, food101, imagenet                  | Yes         | experiments/unified/data.py               |
| Explanations     | ig, gb, ig_sg, gb_sg, ig_sq, gb_sq, ig_var, gb_var    | Yes         | experiments/unified/explanations.py       |
| Imputations      | linear, telea, ns, fixed                              | Yes         | experiments/unified/imputations.py        |
| Rankings         | sort, threshold                                       | Yes         | experiments/unified/ (ranking logic)      |

---

For further details, see the code and README.md.

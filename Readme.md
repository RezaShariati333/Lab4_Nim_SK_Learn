# Lab 4 — Nim Game: Supervised Machine Learning

## Project overview

This project uses supervised machine learning to predict whether the player whose turn it is will lose a Nim position, assuming both players play optimally.

The game contains **three piles**. Each pile can contain between **0 and 9 stones**.

- Number of possible positions: `10 × 10 × 10 = 1,000`
- Input features: `p1`, `p2`, `p3`
- Target label: `loses`
  - `True`: the player whose turn it is will lose with optimal play.
  - `False`: the player whose turn it is can win with optimal play.

The dataset contains 76 losing positions (7.6%) and 924 winning positions (92.4%). Because the classes are imbalanced, accuracy alone is not enough to evaluate a model.

## Requirements

The project uses:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

The required Conda environment is defined in `environment.yml`.

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate nim-ml
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `lab4_nim_ml_sklearn.ipynb` and run the cells from top to bottom using **Shift + Enter**.

## Project structure

The notebook is divided into two main parts.

### Part 1 — Implementation with NumPy

1. Generate all 1,000 possible Nim positions and calculate their labels.
2. Split the dataset into training and test sets (70% / 30%).
3. Create a baseline that always predicts that the current player wins.
4. Implement k-Nearest Neighbors (k-NN) manually using Manhattan distance.
5. Calculate accuracy, precision, and recall manually.

### Part 2 — Implementation with scikit-learn

The same classification task is tested with four models:

- Dummy Classifier (baseline)
- k-Nearest Neighbors (`k=3`, Manhattan distance)
- Decision Tree
- Logistic Regression

The notebook then adds a feature called `nim_sum`:

```python
df["nim_sum"] = df["p1"] ^ df["p2"] ^ df["p3"]
```

This is the bitwise XOR of the three pile sizes. In Nim, a position is losing under optimal play when its XOR value is zero. Adding this feature makes the important structure of the game easier for the models to learn.

The final sections examine the effect of different `k` values on k-NN and display a confusion matrix.

## Results

The following results were obtained by running the notebook with the current `range(10)` dataset and its existing train/test settings.

### Models using only the three pile sizes

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Baseline (always wins) | 0.923 | 0.000 | 0.000 |
| k-NN (k=3) | 0.873 | 0.000 | 0.000 |
| Decision Tree | 0.853 | 0.111 | 0.130 |
| Logistic Regression | 0.923 | 0.000 | 0.000 |

### Models after adding `nim_sum`

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Baseline (always wins) | 0.923 | 0.000 | 0.000 |
| k-NN (k=3) | 0.950 | 0.667 | 0.696 |
| Decision Tree | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 1.000 | 1.000 | 1.000 |

These scores are for the notebook's fixed train/test split; they are not a guarantee of identical results with every split or dataset.

## Interpretation

- **Observation:** With only the three raw pile sizes, the models have difficulty identifying losing positions. The baseline reaches 0.923 accuracy but has zero recall.
- **Reason:** Losing positions are a small minority (7.6%). A model can achieve high accuracy by predicting the majority class while failing to detect losing positions. The relationship between the pile sizes and the winning condition is also not a simple linear pattern.
- **Conclusion:** Accuracy should be considered together with precision and recall. Adding the `nim_sum` feature gives the models information that directly represents the Nim winning condition. In this run, the Decision Tree and Logistic Regression both reached 1.000 on all three reported metrics.

## Evaluation metrics

- **Accuracy:** the proportion of all predictions that are correct.
- **Precision:** among positions predicted as losing, the proportion that really are losing.
- **Recall:** among all truly losing positions, the proportion the model successfully identifies.

## Possible extensions

- Test Decision Tree models with different `max_depth` values.
- Add more features, such as the total number of stones (`p1 + p2 + p3`) or the largest pile (`max(p1, p2, p3)`).
- Add the F1-score to the evaluation tables.
- Increase the pile-size range further and compare the results.

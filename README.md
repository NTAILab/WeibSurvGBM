# WeibSurvGBM: A Survival Analysis Model using Gradient Boosting for Weibull Distribution

WeibSurvGBM is a model designed for survival analysis based on the Weibull distribution. It leverages gradient boosting techniques to predict survival times and perform risk analysis for censored survival data. The model is particularly useful for predicting time-to-event data, which is common in medical research, reliability analysis, and various engineering applications. This repository contains the implementation of the model and code for conducting experiments with survival data.

## Installation

To ensure compatibility and avoid conflicts, it is recommended to set up an isolated Python environment. You can use [conda](https://docs.anaconda.com/miniconda/) for this purpose.

To install `WeibSurvGBM` in development mode after cloning the repository, follow these steps:

```bash
git clone https://github.com/NTAILab/WeibSurvGBM.git
cd WeibSurvGBM
pip install -e .
```

## Package Contents

The package contains several submodules, including:

- `loss` – loss functions and optimizations tailored for survival analysis.
- `model` – the core survival model using gradient boosting techniques.
- `utils` – utility functions for data preprocessing, metrics, and experiments.
- `examples` – example notebooks demonstrating the model's usage and how to apply it to survival data.

## Usage

Example usage is provided in the `examples` directory, including a demonstration of the model's application to survival datasets.

To use the model for survival analysis, follow these steps:

1. Preprocess the dataset, ensuring it contains censored survival times (e.g., time-to-event data) in the format `(delta, time)` where:
   - `delta`: Censoring indicator (1 if the event occurred, 0 if the data is censored).
   - `time`: The observed survival time.
   The target variable `y` should be in the form of a structured NumPy array with the fields `delta` and `time`.

2. Define the model using `WeibSurvGBM`.
3. Train the model and evaluate performance metrics, such as the C-index, for model evaluation.

Here’s an example of using the `WeibSurvGBM` model for survival analysis:

```python
from weib_surv_gbm.model import WeibSurvGBM
from sksurv.datasets import load_veterans_lung_cancer
from sklearn.model_selection import train_test_split

X, y = load_veterans_lung_cancer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = WeibSurvGBM(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

c_index = model.score(X_test, y_test)
print(f'C-index: {c_index}')
```

This will train the `WeibSurvGBM` model on Veterans dataset and provide predictions on test data.

## Citation

If you use this project in your research, please cite it as follows:

...will be later.
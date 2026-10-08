# Iris Species Classification with KNN

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-notebook-F37626)

An end-to-end machine-learning notebook that classifies iris flowers into three species (*setosa*, *versicolor*, *virginica*) from four measurements. The project covers data inspection, cleaning, exploratory analysis, k-nearest-neighbors tuning with cross-validation, a comparison with other models, and a single honest evaluation on a held-out test set.

## Dataset

The [Iris Species dataset](https://www.kaggle.com/datasets/uciml/iris) (150 samples, 3 balanced classes):

| Feature | Description |
|---|---|
| `SepalLengthCm`, `SepalWidthCm` | Sepal size (cm) |
| `PetalLengthCm`, `PetalWidthCm` | Petal size (cm) |
| `Species` | Target class |
| `Id` | Row identifier (used as the index, not as a feature) |

Download `iris.csv` and place it next to the notebook.

## Workflow

1. **Inspect:** shape, types, summary statistics, missing values.
2. **Clean:** inspect duplicate rows and optionally drop them (`DROP_DUPLICATES`) so identical rows cannot land in both the train and test sets.
3. **Outliers:** counted per species with the IQR rule.
4. **EDA:** histograms by species, class counts, scatter plots, box plots, a pairplot, and pooled vs per-species correlation heatmaps.
5. **Split:** stratified 80/20 train/test split. The test set is used once, at the end.
6. **Tune k:** repeated stratified 5-fold cross-validation on the **training set only**, comparing scaled and unscaled features. Scaling sits inside a scikit-learn `Pipeline`, so it is refit in every fold and there is no data leakage.
7. **Compare models:** KNN vs logistic regression, SVM (RBF), and a decision tree, using the same cross-validation.
8. **Evaluate:** accuracy, classification report, and a labelled confusion matrix on the test set.
9. **Deploy-style example:** predict a new flower and save the pipeline with `joblib`.

## Key takeaways

- **Setosa** is separated almost perfectly by the petal features.
- **Versicolor** and **virginica** overlap, so almost all classification errors occur between these two.
- Correlations pooled across species mix between-species differences with within-species relationships, so the notebook also shows them per species.
- The test set has only about 30 samples, so one mistake moves accuracy by about 3 points. The cross-validated score is a more reliable estimate than the single test score.

## Results

Run the notebook to generate the numbers for your copy of the data. The final cell prints the chosen `k`, the cross-validated accuracy, and the held-out test accuracy.

| Metric | Value |
|---|---|
| Best k | _fill in after running_ |
| Cross-validated accuracy (train) | _fill in after running_ |
| Test accuracy | _fill in after running_ |

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook iris_knn_classification.ipynb
```

Make sure `iris.csv` is in the same folder as the notebook.

## Project structure

```text
.
├── iris_knn_classification.ipynb   # full analysis and model
├── iris.csv                        # dataset (download from Kaggle)
└── README.md
```

## Tech stack

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, joblib.

## Limitations and future work

- Iris is small and easy, so these results say little about how KNN performs on harder data.
- Possible extensions: hyperparameter search for the other models (`GridSearchCV`), distance-weighted KNN, feature selection using only the petal features, learning curves, and a small Streamlit app for live predictions.

## License

Add a `LICENSE` file (for example MIT) before publishing. The Iris dataset is publicly available from the UCI Machine Learning Repository.

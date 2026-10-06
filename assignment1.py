
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    make_scorer,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATA
# ==========================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n--- Data types ---")
print(train.dtypes)

# Duplicates
print("\nDuplicate rows - train:", train.duplicated().sum(),
      "| test:", test.duplicated().sum())

# Missing values
print("\nMissing cells - train:", train.isna().sum().sum(), "of", train.size)
print("Missing cells - test:", test.isna().sum().sum(), "of", test.size)
print("\nMissing values per column (train):")
print(train.isna().sum())

# Class balance
print("\nLabel distribution (train):")
print(train["label"].value_counts(normalize=True, dropna=False).round(4))
print("\nLabel distribution (test):")
print(test["label"].value_counts(normalize=True, dropna=False).round(4))

# Numeric summary + IQR outlier check
eda_numeric = train.drop(columns=["label"]).select_dtypes(include="number").columns
print("\nNumeric summary (train):")
print(train[eda_numeric].describe().T.round(3))

print("\nIQR outlier share per numeric column (train, %):")
for col in eda_numeric:
    q1, q3 = train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier = (train[col] < q1 - 1.5 * iqr) | (train[col] > q3 + 1.5 * iqr)
    print(f"  {col:20s} {outlier.mean() * 100:5.1f}%")

# Categorical cardinality + unseen categories in test
eda_categorical = train.drop(columns=["label"]).select_dtypes(exclude="number").columns
print("\nCategorical cardinality (train):")
for col in eda_categorical:
    unseen = set(test[col].dropna()) - set(train[col].dropna())
    print(f"  {col:20s} {train[col].nunique():3d} levels | unseen in test: {len(unseen)}")


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_test = test.drop(columns=["label"])
y_test = test["label"]


# ==========================================
# 4A. CLEAN TARGET VARIABLE
# ==========================================

# Remove rows where the target is missing (cannot be used for training/evaluation)
train_valid = y_train.notna()
test_valid = y_test.notna()
print("\nRows dropped for missing label - train:", (~train_valid).sum(),
      "| test:", (~test_valid).sum())

X_train = X_train.loc[train_valid].copy()
y_train = y_train.loc[train_valid].copy()

X_test = X_test.loc[test_valid].copy()
y_test = y_test.loc[test_valid].copy()

# Normalise label text
y_train = y_train.astype(str).str.strip().str.lower()
y_test = y_test.astype(str).str.strip().str.lower()


# ==========================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL VARIABLES
# ==========================================

numerical_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)


# ==========================================
# 6. PREPROCESSING
# ==========================================

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 7. CROSS-VALIDATION SET-UP
# ==========================================

# 5-fold stratified CV keeps the 76/24 class ratio in every fold.
# F1 on the minority "yes" class is the selection metric because of the imbalance.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
f1_yes = make_scorer(f1_score, pos_label="yes")


# ==========================================
# 8. LOGISTIC REGRESSION (tuned)
# ==========================================

logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        solver="lbfgs",
        tol=1e-4,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

logistic_grid = GridSearchCV(
    logistic_pipeline,
    param_grid={"classifier__C": [0.01, 0.1, 1, 10]},
    scoring=f1_yes,
    cv=cv,
    n_jobs=-1
)
logistic_grid.fit(X_train, y_train)

print("\n--- Logistic Regression tuning (mean CV F1) ---")
for params, score in zip(logistic_grid.cv_results_["params"],
                         logistic_grid.cv_results_["mean_test_score"]):
    print(f"  {params} -> {score:.4f}")
print("Best:", logistic_grid.best_params_,
      f"(CV F1 = {logistic_grid.best_score_:.4f})")

logistic_model = logistic_grid.best_estimator_
print("Iterations used by lbfgs:", logistic_model.named_steps["classifier"].n_iter_[0],
      "of max 1000")

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]


# ==========================================
# 9. RANDOM FOREST (tuned)
# ==========================================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid={
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 12]
    },
    scoring=f1_yes,
    cv=cv,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print("\n--- Random Forest tuning (mean CV F1) ---")
for params, score in zip(rf_grid.cv_results_["params"],
                         rf_grid.cv_results_["mean_test_score"]):
    print(f"  {params} -> {score:.4f}")
print("Best:", rf_grid.best_params_, f"(CV F1 = {rf_grid.best_score_:.4f})")

random_forest_model = rf_grid.best_estimator_

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]


# ==========================================
# 9A. OUT-OF-BAG CHECK ON NUMBER OF TREES
# ==========================================


best_depth = rf_grid.best_params_["classifier__max_depth"]
X_train_prepared = preprocessor.fit(X_train).transform(X_train)

oob_forest = RandomForestClassifier(
    max_depth=best_depth,
    oob_score=True,
    warm_start=True,       
    random_state=RANDOM_STATE,
    n_jobs=-1
)

tree_counts = [50, 100, 200, 300, 400, 500]
oob_scores = []
print("\n--- OOB accuracy vs number of trees ---")
for n in tree_counts:
    oob_forest.set_params(n_estimators=n)
    oob_forest.fit(X_train_prepared, y_train)
    oob_scores.append(oob_forest.oob_score_)
    print(f"  {n:4d} trees -> OOB accuracy {oob_forest.oob_score_:.4f}")


# ==========================================
# 10. EVALUATION
# ==========================================

def evaluate_model(name, y_true, predictions, probabilities):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Accuracy :", round(accuracy_score(y_true, predictions), 4))
    print("Precision:", round(precision_score(y_true, predictions, pos_label="yes"), 4))
    print("Recall   :", round(recall_score(y_true, predictions, pos_label="yes"), 4))
    print("F1 Score :", round(f1_score(y_true, predictions, pos_label="yes"), 4))
    print("ROC-AUC  :", round(roc_auc_score((y_true == "yes").astype(int), probabilities), 4))
    print("\nConfusion Matrix (rows = true no/yes, cols = predicted no/yes):")
    print(confusion_matrix(y_true, predictions, labels=["no", "yes"]))
    print("\nClassification Report:")
    print(classification_report(y_true, predictions))


evaluate_model("LOGISTIC REGRESSION", y_test, logistic_predictions, logistic_probabilities)
evaluate_model("RANDOM FOREST", y_test, rf_predictions, rf_probabilities)


# ==========================================
# 11. COMPARISON TABLE
# ==========================================

y_test_binary = (y_test == "yes").astype(int)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, logistic_predictions),
                 accuracy_score(y_test, rf_predictions)],
    "Precision": [precision_score(y_test, logistic_predictions, pos_label="yes"),
                  precision_score(y_test, rf_predictions, pos_label="yes")],
    "Recall": [recall_score(y_test, logistic_predictions, pos_label="yes"),
               recall_score(y_test, rf_predictions, pos_label="yes")],
    "F1 Score": [f1_score(y_test, logistic_predictions, pos_label="yes"),
                 f1_score(y_test, rf_predictions, pos_label="yes")],
    "ROC-AUC": [roc_auc_score(y_test_binary, logistic_probabilities),
                roc_auc_score(y_test_binary, rf_probabilities)]
})

print("\n\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))


# ==========================================
# 12. FIGURES
# ==========================================

# Fig. 1 (left): ROC curves
fig, ax = plt.subplots(figsize=(5, 4.2))
for label, probs in [("Logistic Regression", logistic_probabilities),
                     ("Random Forest", rf_probabilities)]:
    fpr, tpr, _ = roc_curve(y_test_binary, probs)
    auc = roc_auc_score(y_test_binary, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("fig_roc.png", dpi=200)

# Fig. 1 (right): confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, (label, preds) in zip(axes, [("Logistic Regression", logistic_predictions),
                                     ("Random Forest", rf_predictions)]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, labels=["no", "yes"], ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(label)
fig.tight_layout()
fig.savefig("fig_confusion.png", dpi=200)

# Supplementary: OOB accuracy vs number of trees
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(tree_counts, oob_scores, marker="o")
ax.set_xlabel("Number of trees")
ax.set_ylabel("OOB accuracy")
ax.set_title("Random Forest: OOB accuracy vs ensemble size")
fig.tight_layout()
fig.savefig("fig_oob.png", dpi=200)

plt.show()

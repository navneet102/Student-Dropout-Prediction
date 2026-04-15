"""
train_model.py
--------------
Reproduces the preprocessing and model training pipeline from the notebook
and exports the best model + scaler + pca to model.pkl
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv("student's dropout dataset.csv")
data.rename(columns={"Nacionality": "Nationality"}, inplace=True)

# ── 2. PCA on curricular-unit columns ───────────────────────────────────────
pca_cols = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

pca = PCA(n_components=1)
pca_result = pca.fit_transform(data[pca_cols])
data["Curricular 1st and 2nd sem PCA"] = pca_result

# ── 3. Drop unused columns ───────────────────────────────────────────────────
cols_to_drop = [
    "Nationality",
    "Mother's occupation",
    "Father's qualification",
    *pca_cols,
    "Inflation rate",
    "GDP",
    "Unemployment rate",
]
data.drop(cols_to_drop, axis=1, inplace=True)

# ── 4. Encode target (Dropout=1, everything else=0) ─────────────────────────
data["Target"] = (data["Target"] == "Dropout").astype(int)

# ── 5. Features / target split ──────────────────────────────────────────────
y = np.array(data["Target"])
X_features = data.drop("Target", axis=1)
feature_names = list(X_features.columns)

# ── 6. Scale features ───────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X_features)

# ── 7. Train / validation / test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ── 8. Train models ─────────────────────────────────────────────────────────
trained_models = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_acc = round(accuracy_score(y_test, lr_model.predict(X_test)), 3)
trained_models["Logistic Regression"] = (lr_model, lr_acc)

# Decision Tree (best depth via validation)
print("Training Decision Tree...")
val_accs = []
for d in range(1, 21):
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    val_accs.append(accuracy_score(y_val, t.predict(X_val)))
best_depth = val_accs.index(max(val_accs)) + 1
tree_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree_model.fit(X_train, y_train)
tree_acc = round(accuracy_score(y_test, tree_model.predict(X_test)), 3)
trained_models["Decision Tree"] = (tree_model, tree_acc)

# SVC (best kernel via validation)
print("Training SVC (all kernels)...")
kernels = ["linear", "poly", "rbf", "sigmoid"]
kernel_scores = {}
svm_linear = None
for k in kernels:
    svm = SVC(random_state=42, kernel=k)
    svm.fit(X_train, y_train)
    kernel_scores[k] = svm.score(X_val, y_val)
    if k == "linear":
        svm_linear = svm
best_kernel = max(kernel_scores, key=kernel_scores.get)
svm_model = SVC(random_state=42, kernel=best_kernel)
svm_model.fit(X_train, y_train)
svm_acc = round(accuracy_score(y_test, svm_model.predict(X_test)), 3)
trained_models["SVC"] = (svm_model, svm_acc)
print(f"  Best SVC kernel: {best_kernel}")

# Random Forest (GridSearchCV)
print("Training Random Forest (GridSearchCV)...")
param_grid_rf = {
    "n_estimators": [25, 50, 75, 100],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9],
    "max_leaf_nodes": [3, 6, 9],
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
rf_acc = round(accuracy_score(y_test, best_rf.predict(X_test)), 3)
trained_models["Random Forest"] = (best_rf, rf_acc)
print(f"  Best RF params: {grid_rf.best_params_}")

# KNN (GridSearchCV)
print("Training KNN (GridSearchCV)...")
param_grid_knn = {"n_neighbors": np.arange(1, 25)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
knn_acc = round(accuracy_score(y_test, best_knn.predict(X_test)), 3)
trained_models["KNN"] = (best_knn, knn_acc)
print(f"  Best KNN params: {grid_knn.best_params_}")

# ── 9. Select best model ─────────────────────────────────────────────────────
print("\n-- Model Accuracies --")
for name, (model, acc) in trained_models.items():
    print(f"  {name}: {acc * 100:.1f}%")

best_name = max(trained_models, key=lambda k: trained_models[k][1])
best_model, best_acc = trained_models[best_name]
print(f"\n[BEST] Best model: {best_name} ({best_acc * 100:.1f}%)")

# ── 10. Save artefacts ───────────────────────────────────────────────────────
payload = {
    "model": best_model,
    "model_name": best_name,
    "scaler": scaler,
    "pca": pca,
    "pca_cols": pca_cols,
    "feature_names": feature_names,
    "accuracy": best_acc,
    "all_models": {name: {"model": m, "accuracy": a} for name, (m, a) in trained_models.items()},
}

with open("model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\n[OK] model.pkl saved successfully!")
print(f"  Features: {feature_names}")

import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "kindey stone urine analysis.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

ENCODER_DIR = BASE_DIR / "encoders_scalers"
MODEL_CACHE_DIR = BASE_DIR / "models_cache"

for folder in [
    ARTIFACTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    ENCODER_DIR,
    MODEL_CACHE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))

required_columns = [
    "gravity",
    "ph",
    "osmo",
    "cond",
    "urea",
    "calc",
    "target",
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

data = data[required_columns].dropna()

print("\nDataset information:")
print(data.info())

print("\nTarget distribution:")
print(data["target"].value_counts())


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\nCreating engineered features...")

data["hydrogen_ion"] = 10 ** (-data["ph"])

data["acidic_alkaline"] = (
    data["ph"] < 7.0
).astype(int)

data["calc_osmo"] = (
    data["calc"] * data["osmo"]
)

data["cond_osmo_ratio"] = (
    data["cond"] / data["osmo"]
)

data["urea_gravity_ratio"] = (
    data["urea"] / data["gravity"]
)

feature_columns = [
    "gravity",
    "ph",
    "osmo",
    "cond",
    "urea",
    "calc",
    "hydrogen_ion",
    "acidic_alkaline",
    "calc_osmo",
    "cond_osmo_ratio",
    "urea_gravity_ratio",
]

X = data[feature_columns]
y = data["target"]


# =========================================================
# TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = MinMaxScaler()

# Convert to float before assigning decimal scaled values.
X_train_scaled = X_train.astype(float).copy()
X_test_scaled = X_test.astype(float).copy()

X_train_scaled.iloc[:, 0:6] = scaler.fit_transform(
    X_train.iloc[:, 0:6]
)

X_test_scaled.iloc[:, 0:6] = scaler.transform(
    X_test.iloc[:, 0:6]
)

with open(
    ENCODER_DIR / "min-max-scaler.pickle",
    "wb",
) as file:
    pickle.dump(scaler, file)

print("\nScaler saved.")


# =========================================================
# DEFINE MODELS
# =========================================================

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),

    "svm": SVC(
        probability=True,
        random_state=42,
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5,
    ),

    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),

    "adaboost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}


# =========================================================
# TRAIN AND EVALUATE MODELS
# =========================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "f1_score": f1,
        }
    )

    trained_models[model_name] = model

    joblib.dump(
        model,
        MODELS_DIR / f"{model_name}.joblib",
    )

    with open(
        MODEL_CACHE_DIR / f"{model_name}.pickle",
        "wb",
    ) as file:
        pickle.dump(model, file)

    print(
        f"Accuracy: {accuracy:.3f} | "
        f"F1-score: {f1:.3f}"
    )


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
)

print("\nModel performance:")
print(results_df)


# =========================================================
# MODEL COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(12, 6))

positions = np.arange(len(results_df))
bar_width = 0.35

plt.bar(
    positions - bar_width / 2,
    results_df["accuracy"],
    bar_width,
    label="Accuracy",
)

plt.bar(
    positions + bar_width / 2,
    results_df["f1_score"],
    bar_width,
    label="F1 Score",
)

plt.xticks(
    positions,
    results_df["model"],
    rotation=35,
    ha="right",
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Machine Learning Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "model_comparison.png",
    dpi=150,
)

plt.close()

print("Saved model_comparison.png")


# =========================================================
# LOGISTIC REGRESSION GRAPH
# =========================================================

logistic_model = trained_models["logistic_regression"]

coefficients = logistic_model.coef_[0]

coefficient_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "coefficient": coefficients,
    }
)

coefficient_df["absolute_value"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    by="absolute_value",
    ascending=True,
)

plt.figure(figsize=(11, 7))

plt.barh(
    coefficient_df["feature"],
    coefficient_df["coefficient"],
)

plt.axvline(
    x=0,
    linestyle="--",
)

plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title(
    "Logistic Regression Feature Coefficients"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "logistic_regression.png",
    dpi=150,
)

plt.close()

print("Saved logistic_regression.png")


# =========================================================
# CONFUSION MATRIX
# =========================================================

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

best_predictions = best_model.predict(X_test_scaled)

matrix = confusion_matrix(
    y_test,
    best_predictions,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
)

display.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png",
    dpi=150,
)

plt.close()

print("Saved confusion_matrix.png")


# =========================================================
# DECISION TREE GRAPH
# =========================================================

decision_tree_model = trained_models["decision_tree"]

plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_columns,
    class_names=["No Stone", "Stone"],
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.title("Decision Tree Classification Model")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "decision_tree.png",
    dpi=150,
)

plt.close()

print("Saved decision_tree.png")


# =========================================================
# SAVE RESULTS
# =========================================================

results_df.to_csv(
    ARTIFACTS_DIR / "model_results.csv",
    index=False,
)

print("Saved model_results.csv")

print("\nTraining completed successfully.")
print("Best model:", best_model_name)
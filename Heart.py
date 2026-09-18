import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)


# ============================================================
# 1. Generate a realistic synthetic heart-disease-style dataset
# ============================================================

np.random.seed(7)

n = 600

age = np.random.randint(29, 77, n)

sex = np.random.randint(0, 2, n)
# 1 = Male, 0 = Female

cp = np.random.randint(0, 4, n)
# Chest pain type: 0, 1, 2, 3

trestbps = np.random.normal(131, 17, n).clip(94, 200)
# Resting blood pressure

chol = np.random.normal(246, 51, n).clip(126, 564)
# Cholesterol

fbs = np.random.binomial(1, 0.15, n)
# Fasting blood sugar > 120 mg/dl

thalach = np.random.normal(150, 22, n).clip(71, 202)
# Maximum heart rate

exang = np.random.binomial(1, 0.33, n)
# Exercise-induced angina

oldpeak = np.random.exponential(1.0, n).clip(0, 6.2)
# ST depression

ca = np.random.randint(0, 4, n)
# Number of major vessels


# ============================================================
# Create a synthetic risk score
# ============================================================

risk_score = (
    0.04 * (age - 50)
    + 0.6 * sex
    + 0.5 * cp
    + 0.02 * (trestbps - 130)
    + 0.01 * (chol - 240)
    + 0.4 * fbs
    - 0.03 * (thalach - 150)
    + 0.8 * exang
    + 0.5 * oldpeak
    + 0.5 * ca
    + np.random.normal(0, 1.2, n)
)


# Create binary target
target = (
    risk_score > np.median(risk_score)
).astype(int)


# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame({
    "Age": age,
    "Sex": sex,
    "ChestPainType": cp,
    "RestingBP": trestbps.round(0),
    "Cholesterol": chol.round(0),
    "FastingBS": fbs,
    "MaxHeartRate": thalach.round(0),
    "ExerciseAngina": exang,
    "Oldpeak": oldpeak.round(2),
    "MajorVessels": ca,
    "Target": target
})


# Save dataset
df.to_csv(
    "heart_data_sample.csv",
    index=False
)

print("Dataset created successfully.")
print(f"Total samples: {len(df)}")
print()


# ============================================================
# 2. Preprocess
# ============================================================

X = df.drop(
    columns=["Target"]
)

y = df["Target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)


# Standardization for Logistic Regression
scaler = StandardScaler()

X_train_s = scaler.fit_transform(
    X_train
)

X_test_s = scaler.transform(
    X_test
)


# ============================================================
# 3. Train Logistic Regression
# ============================================================

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_s,
    y_train
)


# Predictions
y_pred_log = log_model.predict(
    X_test_s
)

# Probability predictions
y_prob_log = log_model.predict_proba(
    X_test_s
)[:, 1]


# ============================================================
# 4. Train Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=7
)

rf_model.fit(
    X_train,
    y_train
)


# Predictions
y_pred_rf = rf_model.predict(
    X_test
)


# ============================================================
# 5. Evaluation Function
# ============================================================

def report(name, y_true, y_pred):

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}"
    )


# ============================================================
# 6. Compare Models
# ============================================================

report(
    "Logistic Regression",
    y_test,
    y_pred_log
)

report(
    "Random Forest",
    y_test,
    y_pred_rf
)


# ============================================================
# 7. Classification Reports
# ============================================================

print()
print("=" * 50)
print("Classification Report - Logistic Regression")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred_log,
        target_names=[
            "No Disease",
            "Disease"
        ],
        zero_division=0
    )
)


print()
print("=" * 50)
print("Classification Report - Random Forest")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=[
            "No Disease",
            "Disease"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. Confusion Matrix - Logistic Regression
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred_log
)


plt.figure(
    figsize=(5, 4.5)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["No Disease", "Disease"]
)

plt.yticks(
    [0, 1],
    ["No Disease", "Disease"]
)


# Add values inside confusion matrix
for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > cm.max() / 2
                else "black"
            ),
            fontsize=14
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    "hd_confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. ROC Curve - Logistic Regression
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob_log
)

roc_auc = auc(
    fpr,
    tpr
)


plt.figure(
    figsize=(5.5, 5)
)

plt.plot(
    fpr,
    tpr,
    color="#2E75B6",
    linewidth=2,
    label=f"ROC Curve (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Logistic Regression"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "hd_roc_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 10. Random Forest Feature Importance
# ============================================================

importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values()


plt.figure(
    figsize=(6, 5)
)

importances.plot(
    kind="barh",
    color="#C00000"
)

plt.xlabel(
    "Importance"
)

plt.title(
    "Feature Importance - Random Forest"
)

plt.tight_layout()

plt.savefig(
    "hd_feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. Print Feature Importance
# ============================================================

print()
print("=" * 50)
print("Random Forest Feature Importance")
print("=" * 50)

print(
    importances.sort_values(
        ascending=False
    )
)


# ============================================================
# 12. Logistic Regression Coefficients
# ============================================================

log_coefficients = pd.Series(
    log_model.coef_[0],
    index=X.columns
).sort_values(
    key=abs,
    ascending=False
)


print()
print("=" * 50)
print("Logistic Regression Coefficients")
print("=" * 50)

print(
    log_coefficients
)


# ============================================================
# 13. Final Output
# ============================================================

print()
print("=" * 50)
print("DONE")
print("=" * 50)

print("Generated files:")

print("1. heart_data_sample.csv")
print("2. hd_confusion_matrix.png")
print("3. hd_roc_curve.png")
print("4. hd_feature_importance.png")

print()
print(f"ROC-AUC: {roc_auc:.4f}")

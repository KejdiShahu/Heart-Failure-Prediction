import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np

RESULT_FOLDER = "results"

# 1. Load and Encode
df = pd.read_csv("heart.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Helper: print metrics + return predictions
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<12} {'Train':>8} {'Test':>8}")
    print(f"  {'-' * 30}")
    for metric, fn in [
        ("Accuracy", accuracy_score),
        ("Precision", lambda y, p: precision_score(y, p, zero_division=0)),
        ("Recall", recall_score),
        ("F1", f1_score),
    ]:
        print(
            f"  {metric:<12} {fn(y_tr, y_pred_train):>8.2f} {fn(y_te, y_pred_test):>8.2f}"
        )

    print(
        classification_report(y_te, y_pred_test, target_names=["No Disease", "Disease"])
    )
    return y_pred_test


# Helper: plot confusion matrix
def plot_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")


# 2. Decision Tree
param_grid = {
    "max_depth": [3],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 2, 5],
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

print(f"Best DT Parameters: {grid_search.best_params_}")
dt_preds = evaluate_model("Decision Tree", best_clf, X_train, y_train, X_test, y_test)

# 3. Visualize the Decision Tree
plt.figure(figsize=(25, 12))
plot_tree(
    best_clf,
    feature_names=X.columns.tolist(),
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Visualization", fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULT_FOLDER}/decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()

# 4. Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=300)
log_reg.fit(X_train_scaled, y_train)
lr_preds = evaluate_model(
    "Logistic Regression", log_reg, X_train_scaled, y_train, X_test_scaled, y_test
)

# Top predictors
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": log_reg.coef_[0]})
print("\n  Top 5 Predictors (by |coefficient|):")
print(
    coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
    .head(5)
    .to_string(index=False)
)

# 5. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = evaluate_model("Naive Bayes", nb_model, X_train, y_train, X_test, y_test)

# 6. Confusion Matrix Grid (all 3 models)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_cm(axes[0], y_test, dt_preds, "Decision Tree")
plot_cm(axes[1], y_test, lr_preds, "Logistic Regression")
plot_cm(axes[2], y_test, nb_preds, "Naive Bayes")
fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULT_FOLDER}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# 7. Metric Comparison Bar Chart
models = ["Decision Tree", "Logistic Regression", "Naive Bayes"]
preds = [dt_preds, lr_preds, nb_preds]
metrics = {
    "Accuracy": [accuracy_score(y_test, p) for p in preds],
    "Precision": [precision_score(y_test, p, zero_division=0) for p in preds],
    "Recall": [recall_score(y_test, p) for p in preds],
    "F1": [f1_score(y_test, p) for p in preds],
}

x = np.arange(len(models))
width = 0.2
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, (metric, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i])
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Test Set Metrics", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_FOLDER}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

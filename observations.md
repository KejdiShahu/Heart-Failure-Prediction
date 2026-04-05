## Observations

This study evaluated three classification models — Decision Tree, Logistic Regression, and Naive Bayes —
on the Heart Disease dataset after applying a standard preprocessing pipeline including one-hot encoding
of categorical features and train/test splitting (80/20, random state 42). Logistic Regression additionally
required feature scaling via `StandardScaler`.

### Model Performance

All three models achieved comparable and strong test-set performance, with accuracy ranging from **85–86%**
and F1-scores between **0.85–0.87**, suggesting the dataset is well-suited for linear and probabilistic
decision boundaries.

| Model               | Accuracy | Precision | Recall | F1   |
|---------------------|----------|-----------|--------|------|
| Decision Tree       | 0.85     | 0.90      | 0.84   | 0.87 |
| Logistic Regression | 0.85     | 0.90      | 0.84   | 0.87 |
| Naive Bayes         | 0.86     | 0.91      | 0.84   | 0.87 |

### Key Findings

**Naive Bayes** edged out the other two models slightly on test accuracy (86%) and precision (0.91),
despite being the simplest model and receiving no feature scaling — suggesting the features are
reasonably conditionally independent given the class label. **Decision Tree**, tuned via GridSearchCV
(`criterion=entropy`, `max_depth=3`, `min_samples_leaf=1`), matched Logistic Regression exactly on
test metrics, indicating the dataset's decision boundary is well-captured at shallow depth without
overfitting. **Logistic Regression** showed a mild train/test accuracy gap (0.87 → 0.85), hinting at
slight overfitting, though the gap remains minor; its top predictors — `ST_Slope_Flat` (+0.66),
`ChestPainType_NAP` (−0.63), and `Sex_M` (+0.54) — align well with established clinical risk factors
for heart disease. Across all models, recall for the Disease class was consistent at 0.84, meaning
roughly 16% of true disease cases were missed. In a clinical context, reducing false negatives would
be a priority, suggesting future work could optimize for recall specifically via threshold adjustment
or class weighting.

### Class Imbalance Note

The test set contains 107 Disease vs. 77 No Disease cases (~58/42 split). The models perform better on
the majority Disease class across all metrics, which is expected. The `No Disease` precision (0.80) being
lower than `Disease` precision (0.90–0.91) reflects this imbalance and warrants attention in real-world deployment.
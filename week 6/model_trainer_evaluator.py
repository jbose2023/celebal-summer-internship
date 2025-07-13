import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Shape:", X.shape)
print("Class Distribution:\n", y.value_counts())

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Target Class Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 3: Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

# Step 5: Train and evaluate models
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n----- {name} -----")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Training
for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        evaluate_model(name, model, X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test)

# Step 6: Hyperparameter Tuning

# GridSearchCV for Random Forest
rf_params = {
    'n_estimators': [100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)
print("Best Random Forest Params:", grid_rf.best_params_)

# RandomizedSearchCV for SVM
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
random_svm = RandomizedSearchCV(SVC(probability=True), svm_params, n_iter=5, cv=5, scoring='f1', random_state=42)
random_svm.fit(X_train_scaled, y_train)
print("Best SVM Params:", random_svm.best_params_)

# Step 7: Final Evaluation of Best Models
evaluate_model("Tuned Random Forest", grid_rf.best_estimator_, X_test, y_test)
evaluate_model("Tuned SVM", random_svm.best_estimator_, X_test_scaled, y_test)

# Step 8: Plot ROC Curve
def plot_roc(model, X_test, y_test, name):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.figure(figsize=(8,6))
plot_roc(grid_rf.best_estimator_, X_test, y_test, "Random Forest")
plot_roc(random_svm.best_estimator_, X_test_scaled, y_test, "SVM")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Step 9: Save Best Model
joblib.dump(grid_rf.best_estimator_, "best_model.pkl")
print("\nModel saved as best_model.pkl")

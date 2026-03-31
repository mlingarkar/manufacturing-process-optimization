import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

#load data from manufacturing defect data file
df = pd.read_csv(r"C:\Users\Megan\OneDrive\Documents\Python_Practice\manufacturing_defect_data.csv")

# make sure Defect Rate is numeric
df["Defect Rate"] = (
    df["Defect Rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float)
)

print("\n--- Dataset Loaded for Machine Learning ---")
print(df.head())
print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")


#targets
# Regression target:
# Predict exact defect rate percentage
y_reg = df["Defect Rate"]

# Classification target:
# Flag jobs with unusually high defect risk
high_defect_threshold = df["Defect Rate"].quantile(0.75)
df["High Defect Risk"] = (df["Defect Rate"] >= high_defect_threshold).astype(int)
y_clf = df["High Defect Risk"]

print(f"\nHigh defect threshold (75th percentile): {high_defect_threshold:.2f}%")


#feature selection
# Use only inputs known before or during production setup.
# Exclude leakage / outcome columns.
feature_columns = [
    "Customer",
    "Industry",
    "Shift",
    "Operator Level",
    "Material",
    "Department",
    "Machine",
    "Process Type",
    "Complexity Score",
    "Tool Condition",
    "Rush Order",
    "Machine Downtime",
    "Inspection Type",
    "Outside Process",
    "Outside Process Type",
    "Hardware",
    "Hardware Type",
    "Deburr",
    "Setup Minutes",
    "Cycle Time Seconds",
    "Total Parts",
    "Thread"
]

X = df[feature_columns]

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

print("\n--- Features Used ---")
print(feature_columns)


#reprocessesing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_columns),
        ("num", numeric_transformer, numeric_columns)
    ]
)


#regression model
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

regression_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

regression_model.fit(X_train_reg, y_train_reg)
y_pred_reg = regression_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = mean_squared_error(y_test_reg, y_pred_reg) ** 0.5
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n--- Regression Model: Predict Defect Rate ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


#classification model
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

classification_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

classification_model.fit(X_train_clf, y_train_clf)
y_pred_clf = classification_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)

print("\n--- Classification Model: Predict High Defect Risk ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf))


#feature importance
#get feature names after one-hot encoding
ohe = regression_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
encoded_cat_names = ohe.get_feature_names_out(categorical_columns)
all_feature_names = list(encoded_cat_names) + numeric_columns

feature_importance = pd.Series(
    regression_model.named_steps["model"].feature_importances_,
    index=all_feature_names
).sort_values(ascending=False)

print("\n--- Top 15 Most Important Features for Defect Rate Prediction ---")
print(feature_importance.head(15))


#save predictions
results = X_test_reg.copy()
results["Actual Defect Rate"] = y_test_reg.values
results["Predicted Defect Rate"] = y_pred_reg
results["Prediction Error"] = results["Actual Defect Rate"] - results["Predicted Defect Rate"]

results.to_csv("ml_predictions_output.csv", index=False)
print("\nSaved: ml_predictions_output.csv")


#visual feature of actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, edgecolor="black")
plt.xlabel("Actual Defect Rate (%)")
plt.ylabel("Predicted Defect Rate (%)")
plt.title("Actual vs Predicted Defect Rate")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


#visual features
top_features = feature_importance.head(12).sort_values()

plt.figure(figsize=(10, 6))
top_features.plot(kind="barh", edgecolor="black")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top Features Driving Defect Rate Prediction")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - High Defect Risk")
plt.colorbar()
plt.xticks([0, 1], ["Low Risk", "High Risk"])
plt.yticks([0, 1], ["Low Risk", "High Risk"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()
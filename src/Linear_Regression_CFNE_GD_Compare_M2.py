import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = (
    "C:/Users/harin/PythonProject/placement_prediction/"
    "dataset/final_preprocess_M2.csv"
)

IMAGE_FOLDER = (
    "C:/Users/harin/PythonProject/placement_prediction/"
    "outputs/Linear_Regression_CFNE_GD_Compare_M2"
)

RESULT_FILE = (
    "C:/Users/harin/PythonProject/placement_prediction/"
    "dataset/CFNE_GD_Comparison_M2.csv"
)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\n====================================================")
print("       LINEAR REGRESSION - CFNE vs GD")
print("====================================================")

print("\nDataset path:")
print(DATASET_PATH)

print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 3. CHECK DATASET EXISTS
# ============================================================

if not os.path.isfile(DATASET_PATH):
    print("\nERROR: Dataset file not found!")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)


# ============================================================
# 4. LOAD DATASET
# ============================================================

data = pd.read_csv(DATASET_PATH)

print("\n================ DATASET INFORMATION ================\n")

print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

print(data.isnull().sum())


# ============================================================
# 6. REPLACE INFINITY WITH NaN
# ============================================================

data = data.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = data.select_dtypes(
    include=np.number
).columns

categorical_columns = data.select_dtypes(
    exclude=np.number
).columns


# Fill numerical columns using median
for column in numeric_columns:

    if data[column].isnull().sum() > 0:

        median_value = data[column].median()

        data[column] = data[column].fillna(median_value)


# Fill categorical columns using mode
for column in categorical_columns:

    if data[column].isnull().sum() > 0:

        mode_value = data[column].mode()

        if len(mode_value) > 0:
            data[column] = data[column].fillna(mode_value[0])


# ============================================================
# 8. REMOVE ROWS STILL CONTAINING NaN
# ============================================================

data = data.dropna()


print("\n================ AFTER CLEANING ================\n")

print("Dataset shape:", data.shape)

print("\nRemaining NaN values:")
print(data.isnull().sum().sum())


# ============================================================
# 9. EXTRACT FEATURES AND TARGET
# ============================================================

# Your original code uses the LAST column as target.

X = data.iloc[:, :-1].copy()

y = data.iloc[:, -1].copy()

TARGET_NAME = data.columns[-1]


print("\n================ TARGET INFORMATION ================\n")

print("Target column:", TARGET_NAME)

print("Number of features:", X.shape[1])

print("Number of samples:", X.shape[0])


# ============================================================
# 10. CONVERT CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# Convert everything to numerical values
X = X.astype(float)


# Convert target to numerical values if necessary
if not pd.api.types.is_numeric_dtype(y):

    y = pd.factorize(y)[0]

else:

    y = y.astype(float).values


# ============================================================
# 11. FINAL NaN / INFINITY CHECK
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())

X = X.values.astype(float)

y = np.asarray(y, dtype=float)


print("\n================ FINAL DATA CHECK ================\n")

print("X shape:", X.shape)

print("y shape:", y.shape)

print("NaN in X:", np.isnan(X).sum())

print("NaN in y:", np.isnan(y).sum())

print("Infinity in X:", np.isinf(X).sum())

print("Infinity in y:", np.isinf(y).sum())


# ============================================================
# 12. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n================ TRAIN TEST SPLIT ================\n")

print("Training samples:", len(X_train))

print("Testing samples :", len(X_test))


# ============================================================
# 13. FEATURE SCALING
# ============================================================

# Scaling is important for Gradient Descent.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 14. ADD BIAS COLUMN
# ============================================================

# Add column of 1s for intercept.

X_train_bias = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_bias = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]


# ============================================================
# 15. CLOSED FORM NORMAL EQUATION (CFNE)
# ============================================================

print("\n====================================================")
print("       CLOSED FORM NORMAL EQUATION")
print("====================================================")


# Formula:
#
# theta = (X^T X)^(-1) X^T y
#
# pinv is used instead of inverse because it is
# numerically safer when X^T X is singular.

theta_cfne = np.linalg.pinv(
    X_train_bias.T @ X_train_bias
) @ X_train_bias.T @ y_train


# Prediction

y_pred_cfne = X_test_bias @ theta_cfne


# Metrics

mse_cfne = mean_squared_error(
    y_test,
    y_pred_cfne
)

mae_cfne = mean_absolute_error(
    y_test,
    y_pred_cfne
)

r2_cfne = r2_score(
    y_test,
    y_pred_cfne
)


print("\nCFNE Results:")

print("MSE :", mse_cfne)

print("MAE :", mae_cfne)

print("R2  :", r2_cfne)


# ============================================================
# 16. GRADIENT DESCENT
# ============================================================

print("\n====================================================")
print("             GRADIENT DESCENT")
print("====================================================")


# Hyperparameters

learning_rate = 0.01

epochs = 1000


# Number of training examples

m = X_train_bias.shape[0]


# Number of features including bias

n = X_train_bias.shape[1]


# Initialize weights

theta_gd = np.zeros(n)


# Store loss values

loss_history = []


# ============================================================
# 17. GRADIENT DESCENT TRAINING
# ============================================================

for epoch in range(epochs):

    # Prediction
    y_train_pred = X_train_bias @ theta_gd

    # Error
    error = y_train_pred - y_train

    # Cost / MSE
    cost = np.mean(error ** 2)

    # Store cost
    loss_history.append(cost)

    # Gradient
    gradient = (
        2 / m
    ) * (
        X_train_bias.T @ error
    )

    # Update weights
    theta_gd = theta_gd - (
        learning_rate * gradient
    )


# ============================================================
# 18. GRADIENT DESCENT PREDICTION
# ============================================================

y_pred_gd = X_test_bias @ theta_gd


# ============================================================
# 19. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    y_pred_gd
)

mae_gd = mean_absolute_error(
    y_test,
    y_pred_gd
)

r2_gd = r2_score(
    y_test,
    y_pred_gd
)


print("\nGradient Descent Results:")

print("MSE :", mse_gd)

print("MAE :", mae_gd)

print("R2  :", r2_gd)


# ============================================================
# 20. COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Method": [
        "Closed Form Normal Equation",
        "Gradient Descent"
    ],

    "MSE": [
        mse_cfne,
        mse_gd
    ],

    "MAE": [
        mae_cfne,
        mae_gd
    ],

    "R2 Score": [
        r2_cfne,
        r2_gd
    ]

})


print("\n====================================================")
print("              MODEL COMPARISON")
print("====================================================")

print(comparison)


# ============================================================
# 21. SAVE COMPARISON RESULTS
# ============================================================

comparison.to_csv(
    RESULT_FILE,
    index=False
)

print("\nComparison results saved to:")

print(RESULT_FILE)


# ============================================================
# 22. GRAPH 1 - GD LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel("Epoch")

plt.ylabel("Mean Squared Error")

plt.title(
    "Gradient Descent Loss Curve"
)

plt.grid(True)

loss_graph = os.path.join(
    IMAGE_FOLDER,
    "GD_Loss_Curve.png"
)

plt.savefig(
    loss_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 23. GRAPH 2 - ACTUAL VS CFNE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_cfne,
    alpha=0.6
)

# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred_cfne.min()
)

maximum = max(
    y_test.max(),
    y_pred_cfne.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title(
    "Actual vs Predicted - CFNE"
)

plt.grid(True)

cfne_graph = os.path.join(
    IMAGE_FOLDER,
    "Actual_vs_Predicted_CFNE.png"
)

plt.savefig(
    cfne_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 24. GRAPH 3 - ACTUAL VS GD
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_gd,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    y_pred_gd.min()
)

maximum = max(
    y_test.max(),
    y_pred_gd.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title(
    "Actual vs Predicted - Gradient Descent"
)

plt.grid(True)

gd_graph = os.path.join(
    IMAGE_FOLDER,
    "Actual_vs_Predicted_GD.png"
)

plt.savefig(
    gd_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 25. GRAPH 4 - METRIC COMPARISON
# ============================================================

methods = [
    "CFNE",
    "Gradient Descent"
]


# ---------------- MSE ----------------

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    [mse_cfne, mse_gd]
)

plt.xlabel("Method")

plt.ylabel("MSE")

plt.title(
    "MSE Comparison"
)

plt.grid(
    axis="y"
)

mse_graph = os.path.join(
    IMAGE_FOLDER,
    "MSE_Comparison.png"
)

plt.savefig(
    mse_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ---------------- MAE ----------------

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    [mae_cfne, mae_gd]
)

plt.xlabel("Method")

plt.ylabel("MAE")

plt.title(
    "MAE Comparison"
)

plt.grid(
    axis="y"
)

mae_graph = os.path.join(
    IMAGE_FOLDER,
    "MAE_Comparison.png"
)

plt.savefig(
    mae_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ---------------- R2 ----------------

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    [r2_cfne, r2_gd]
)

plt.xlabel("Method")

plt.ylabel("R2 Score")

plt.title(
    "R2 Score Comparison"
)

plt.grid(
    axis="y"
)

r2_graph = os.path.join(
    IMAGE_FOLDER,
    "R2_Comparison.png"
)

plt.savefig(
    r2_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 26. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({

    "Actual": y_test,

    "CFNE_Predicted": y_pred_cfne,

    "GD_Predicted": y_pred_gd

})


prediction_file = os.path.join(
    IMAGE_FOLDER,
    "CFNE_GD_Predictions_M2.csv"
)

prediction_results.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n====================================================")
print("                 FINAL RESULTS")
print("====================================================")

print("\nTarget column:", TARGET_NAME)

print("\n--------------- CFNE ----------------")

print("MSE :", mse_cfne)

print("MAE :", mae_cfne)

print("R2  :", r2_cfne)


print("\n--------------- Gradient Descent ----------------")

print("MSE :", mse_gd)

print("MAE :", mae_gd)

print("R2  :", r2_gd)


print("\n====================================================")

print("All processing completed successfully!")

print("====================================================")


print("\nFiles generated:")

print("\n1. Comparison:")
print(RESULT_FILE)

print("\n2. Predictions:")
print(prediction_file)

print("\n3. Graphs:")
print(IMAGE_FOLDER)

print("\n====================================================")
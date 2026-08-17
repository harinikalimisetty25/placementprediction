import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATH
# ============================================================

DATASET_PATH = "C:/Users/harin/PythonProject/placement_prediction/dataset/final_preprocess_M2.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\n================ DATASET INFORMATION ================\n")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())


# ============================================================
# 4. REPLACE INFINITY WITH NaN
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Fill numerical columns with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


# Fill non-numerical columns with mode
categorical_columns = df.select_dtypes(exclude=np.number).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# ============================================================
# 6. CHECK NaN AGAIN
# ============================================================

print("\n================ AFTER NaN HANDLING ================\n")

print(df.isnull().sum())

print("\nTotal remaining NaN values:",
      df.isnull().sum().sum())


# ============================================================
# 7. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "PlacementStatus"


# Check whether target column exists
if TARGET_COLUMN not in df.columns:
    print("\nERROR: Target column not found!")
    print("\nAvailable columns are:")
    print(df.columns.tolist())
    exit()


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


print("\n================ FEATURES AND TARGET ================\n")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget column:", TARGET_COLUMN)


# ============================================================
# 9. CONVERT CATEGORICAL FEATURES
# ============================================================

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)


# Convert Boolean values to integers
X = X.astype(float)


# ============================================================
# 10. FINAL NaN CHECK
# ============================================================

print("\n================ FINAL CHECK ================\n")

print("NaN values in X:", X.isnull().sum().sum())
print("NaN values in y:", y.isnull().sum())

print("Infinite values in X:",
      np.isinf(X).sum().sum())


# ============================================================
# 11. REMOVE ANY REMAINING INVALID VALUES
# ============================================================

# Replace infinity
X = X.replace([np.inf, -np.inf], np.nan)

# Fill any remaining NaN
X = X.fillna(X.median())

# If target contains NaN, remove those rows
valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


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

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 13. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 14. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


print("\n================ MODEL TRAINED ================\n")

print("Linear Regression model trained successfully!")


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 16. CALCULATE METRICS
# ============================================================

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


# ============================================================
# 17. DISPLAY METRICS
# ============================================================

print("\n====================================================")
print("             LINEAR REGRESSION RESULTS")
print("====================================================")

print("Mean Squared Error (MSE):", mse)

print("Mean Absolute Error (MAE):", mae)

print("R² Score:", r2)

print("====================================================")


# ============================================================
# 18. MODEL COEFFICIENTS
# ============================================================

print("\n================ MODEL COEFFICIENTS ================\n")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print(coefficients)


# ============================================================
# 19. ACTUAL VS PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n================ ACTUAL VS PREDICTED ================\n")

print(results.head(10))


# ============================================================
# 20. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted - Linear Regression")

plt.grid(True)

plt.show()


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

results.to_csv(
    "C:/Users/harin/PythonProject/placement_prediction/dataset/Linear_Regression_Predictions_M2.csv",
    index=False
)

print("\nPrediction results saved successfully.")


# ============================================================
# 22. FINISHED
# ============================================================

print("\n================ COMPLETE ================\n")

print("Linear Regression completed successfully.")
print("MSE :", mse)
print("MAE :", mae)
print("R²  :", r2)
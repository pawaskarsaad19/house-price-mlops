import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("data/train.csv")

# Select only important numeric features
features = ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea"]

df = df[features + ["SalePrice"]]

# Fill missing values
df = df.fillna(df.mean())

# Split data
X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Simple model trained!")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)
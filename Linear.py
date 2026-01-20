# Step 1: import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: loading Dataset

df = pd.read_csv("AdvertisingBudgetandSales.csv")
df.columns = ['ID', 'TV', 'Radio', 'Newspaper', 'Sales']

# Step 3: explore data
print(df.head())
print(df.info())

# Step 4: feature selection


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Step 5: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: predictions
y_pred = model.predict(X_test)

# Step 8: evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Predict Future Sales

future_sales = model.predict([[200, 50, 30]])
print("Predicted Sales:", future_sales[0])

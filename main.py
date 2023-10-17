import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from useful_package import polynom_3, hyperbola

# Generate sample data
X = np.linspace(-10, 10, 1000).reshape(-1, 1)

# Data for hyperbola (actually a cubic function)
X_hyperbola, y_hyperbola = hyperbola(-10, 10)

# Data for 3rd-degree polynomial using the polynom_3 function
y_polynom_3 = polynom_3([4, 3, 2, 1], X)

# Split data for training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_hyperbola_train, y_hyperbola_test = y_hyperbola[:split], y_hyperbola[split:]
y_polynom_3_train, y_polynom_3_test = y_polynom_3[:split], y_polynom_3[split:]

# Train RandomForestRegressor for each function
rf_hyperbola = RandomForestRegressor(n_estimators=100).fit(X_train, y_hyperbola_train.ravel())
rf_polynom_3 = RandomForestRegressor(n_estimators=100).fit(X_train, y_polynom_3_train.ravel())

# Make predictions
y_hyperbola_pred = rf_hyperbola.predict(X_test)
y_polynom_3_pred = rf_polynom_3.predict(X_test)

# Calculate MSE for the predictions
mse_hyperbola = mean_squared_error(y_hyperbola_test, y_hyperbola_pred)
mse_polynom_3 = mean_squared_error(y_polynom_3_test, y_polynom_3_pred)

print(f"MSE for Hyperbola (Cubic Function): {mse_hyperbola}")
print(f"MSE for 3rd-degree Polynomial: {mse_polynom_3}")

if __name__ == "__main__":
    pass

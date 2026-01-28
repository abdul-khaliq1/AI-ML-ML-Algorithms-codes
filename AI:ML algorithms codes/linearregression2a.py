import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.2, 1.8, 2.6, 3.2, 3.8])

# Mean values
x_mean = np.mean(x)
y_mean = np.mean(y)

# Regression parameters
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

# Print regression equation
print("Linear Regression Equation:")
print(f"y = {m:.2f}x + {b:.2f}")

# Prediction for 7 weeks
x_7 = 7
y_7 = m * x_7 + b

# Print prediction in terminal
print("\nPrediction:")
print(f"For 7 weeks, predicted sales = {y_7:.2f} thousand")

# Regression line
x_line = np.linspace(1, 7, 100)
y_line = m * x_line + b

# Plot
plt.figure()
plt.scatter(x, y)
plt.plot(x_line, y_line)
plt.scatter(x_7, y_7)

# Label predicted point
plt.text(
    x_7,
    y_7,
    f"({x_7}, {y_7:.2f})",
    fontsize=10,
    verticalalignment="bottom"
)

plt.xlabel("Weeks")
plt.ylabel("Sales (in thousands)")
plt.title("Sales Prediction for 7 Weeks")
plt.grid(True)
plt.show()

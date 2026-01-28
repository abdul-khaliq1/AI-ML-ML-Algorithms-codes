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

# Prediction for 12 weeks
x_12 = 12
y_12 = m * x_12 + b

# Print prediction in terminal
print("\nPrediction:")
print(f"For 12 weeks, predicted sales = {y_12:.2f} thousand")

# Regression line up to 12 weeks
x_line = np.linspace(1, 12, 100)
y_line = m * x_line + b

# Plot
plt.figure()
plt.scatter(x, y)
plt.plot(x_line, y_line)
plt.scatter(x_12, y_12)

# Label predicted point (FIXED LINE)
plt.text(
    x_12,
    y_12,
    f"({x_12}, {y_12:.2f})",
    fontsize=10,
    verticalalignment="bottom"
)

plt.xlabel("Weeks")
plt.ylabel("Sales (in thousands)")
plt.title("Sales Prediction for 12 Weeks")
plt.grid(True)
plt.show()

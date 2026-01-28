import numpy as np
import matplotlib.pyplot as plt

# DATA
x = np.array([8, 10, 13])
y = np.array([10, 13, 16])

# MWAN VALUES
x_bar = np.mean(x)
y_bar = np.mean(y)

# SLOPE(m)
m = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar)**2)

# Intercept (b)
b = y_bar - m * x_bar

# Prediction
x_new = 20
y_new = m * x_new + b

print("Slope (m):", m)
print("Intercept (b):", b)
print("Predicted value at x = 20:", y_new)

# Graph
plt.scatter(x, y, label="Given Data")
plt.plot(x, m*x + b, label="Regression Line")
plt.scatter(x_new, y_new, label="Prediction (x=20)")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Graph")
plt.legend()
plt.show()

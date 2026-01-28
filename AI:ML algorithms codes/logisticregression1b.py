import numpy as np
import matplotlib.pyplot as plt


# Data (x = 6)
X = np.array([2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1])


# Logistic regression parameters
a0 = -1.5
a1 = 0.6


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# X=6
print("\nLogistic Regression Output (Up to x = 6)")
print("---------------------------------------")
print("Study Hours | Actual | Probability")
print("----------------------------------")

for xi, yi in zip(X, y):
    prob = sigmoid(a0 + a1 * xi)
    print(f"{xi:^11} | {yi:^6} | {prob:.3f}")


# Separate x = 6 result
x = 6
p = sigmoid(a0 + a1 * x)

print("\nSeparate Result")
print("---------------")
print(f"x = {x}")
print(f"Probability of Passing = {p:.3f}")


# Graph
X_curve = np.linspace(1, 9, 100)
y_curve = sigmoid(a0 + a1 * X_curve)

plt.figure()
plt.scatter(X, y, label="Actual Data (up to 6)")
plt.plot(X_curve, y_curve, label="Logistic Curve")
plt.scatter(x, p, label="x = 6", marker='s')

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.grid()
plt.show()

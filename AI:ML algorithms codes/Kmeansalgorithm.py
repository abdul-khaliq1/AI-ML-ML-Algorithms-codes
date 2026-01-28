import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


print("K-MEANS CLUSTERING (K = 2)\n")

print("Given Data:")
data = {
    "C1": (20, 500),
    "C2": (40, 1000),
    "C3": (30, 800),
    "C4": (18, 300),
    "C5": (28, 1200),
    "C6": (35, 1400),
    "C7": (45, 1800),
}
for k, v in data.items():
    print(f"{k}: Age={v[0]}, Amount={v[1]}")

print("\nStep 1: Initial Centroids")
print("K = 1 -> C1 (20, 500)")
print("K = 2 -> C2 (40, 1000)")

print("\nStep 2: After assigning C3 and C4")
print("K = 1 -> C1, C4")
print("K = 2 -> C2, C3")

print("\nStep 3: After assigning C5 and C6")
print("K = 1 -> C1, C4")
print("K = 2 -> C2, C3, C5, C6")

print("\nStep 4: Final Assignment")
print("K = 1 -> C1, C4")
print("K = 2 -> C2, C3, C5, C6, C7")

# Final centroids
K1_final = np.array([[20, 500], [18, 300]])
K2_final = np.array([
    [40, 1000],
    [30, 800],
    [28, 1200],
    [35, 1400],
    [45, 1800]
])
centroid1 = K1_final.mean(axis=0)
centroid2 = K2_final.mean(axis=0)

print("\nFinal Centroids:")
print(f"K = 1 centroid = ({centroid1[0]:.1f}, {centroid1[1]:.1f})")
print(f"K = 2 centroid = ({centroid2[0]:.1f}, {centroid2[1]:.1f})")

# DIAGRAM
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# S1
K1 = np.array([[20, 500]])
K2 = np.array([[40, 1000]])

ax = axes[0, 0]
ax.scatter(K1[:, 0], K1[:, 1])
ax.scatter(K2[:, 0], K2[:, 1])
ax.text(20.3, 520, "C1")
ax.text(40.3, 1020, "C2")
ax.add_patch(Ellipse((20, 500), 6, 300, fill=False))
ax.add_patch(Ellipse((40, 1000), 6, 300, fill=False))
ax.text(18, 650, "K = 1")
ax.text(38, 1150, "K = 2")
ax.set_title("Step 1")
ax.axis("off")

K1 = np.array([[20, 500], [18, 300]])
K2 = np.array([[40, 1000], [30, 800]])

ax = axes[0, 1]
ax.scatter(K1[:, 0], K1[:, 1])
ax.scatter(K2[:, 0], K2[:, 1])
for i, l in enumerate(["C1", "C4"]):
    ax.text(K1[i, 0] + 0.3, K1[i, 1] + 20, l)
for i, l in enumerate(["C2", "C3"]):
    ax.text(K2[i, 0] + 0.3, K2[i, 1] + 20, l)
ax.add_patch(Ellipse((19, 400), 7, 550, fill=False))
ax.add_patch(Ellipse((35, 900), 12, 600, fill=False))
ax.text(16, 700, "K = 1")
ax.text(33, 1150, "K = 2")
ax.set_title("Step 2")
ax.axis("off")


K2 = np.array([[40, 1000], [30, 800], [28, 1200], [35, 1400]])

ax = axes[1, 0]
ax.scatter(K1[:, 0], K1[:, 1])
ax.scatter(K2[:, 0], K2[:, 1])
for i, l in enumerate(["C1", "C4"]):
    ax.text(K1[i, 0] + 0.3, K1[i, 1] + 20, l)
for i, l in enumerate(["C2", "C3", "C5", "C6"]):
    ax.text(K2[i, 0] + 0.3, K2[i, 1] + 20, l)
ax.add_patch(Ellipse((19, 400), 7, 550, fill=False))
ax.add_patch(Ellipse((34, 1150), 20, 1000, fill=False))
ax.text(16, 700, "K = 1")
ax.text(30, 1600, "K = 2")
ax.set_title("Step 3")
ax.axis("off")

# S4
K2 = K2_final

ax = axes[1, 1]
ax.scatter(K1[:, 0], K1[:, 1])
ax.scatter(K2[:, 0], K2[:, 1])
for i, l in enumerate(["C1", "C4"]):
    ax.text(K1[i, 0] + 0.3, K1[i, 1] + 20, l)
for i, l in enumerate(["C2", "C3", "C5", "C6", "C7"]):
    ax.text(K2[i, 0] + 0.3, K2[i, 1] + 20, l)
ax.add_patch(Ellipse((19, 400), 7, 550, fill=False))
ax.add_patch(Ellipse((36, 1200), 32, 1600, fill=False))  # C7 INSIDE
ax.text(16, 700, "K = 1")
ax.text(30, 1850, "K = 2")
ax.set_title("Final Step")
ax.axis("off")

plt.suptitle("K-Means Clustering (K = 2) – Step-wise Diagram", fontsize=14)
plt.tight_layout()
plt.show()

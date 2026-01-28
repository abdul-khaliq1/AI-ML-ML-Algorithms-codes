import numpy as np
import matplotlib.pyplot as plt

# =============================
# GIVEN DATA (6 INSTANCES)
# =============================
points = {
    "C1": (185, 72),
    "C2": (170, 56),
    "C3": (168, 60),
    "C4": (179, 68),
    "C5": (182, 72),
    "C6": (188, 77)
}

# Initial centroids (from notebook)
centroid_K1 = np.array(points["C1"])
centroid_K2 = np.array(points["C2"])

print("INITIAL CENTROIDS")
print("K = 1 :", centroid_K1)
print("K = 2 :", centroid_K2)

print("\nEUCLIDEAN DISTANCES\n")

# Distance calculation
for k, v in points.items():
    p = np.array(v)
    d1 = np.linalg.norm(p - centroid_K1)
    d2 = np.linalg.norm(p - centroid_K2)
    print(f"{k} -> d(K=1) = {d1:.3f}, d(K=2) = {d2:.3f}")

# Final clusters (as fixed)
K1 = ["C1", "C4", "C5", "C6"]
K2 = ["C2", "C3"]

print("\nFINAL CLUSTERS")
print("K = 1 :", K1)
print("K = 2 :", K2)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-Means Clustering (K = 2) – Step-wise Diagram", fontsize=16)

# STEP 1
ax = axs[0, 0]
ax.set_title("Step 1: Initial Clusters")

ax.scatter(*points["C1"])
ax.text(points["C1"][0]+0.4, points["C1"][1]+0.4, "C1")

ax.scatter(*points["C2"])
ax.text(points["C2"][0]+0.4, points["C2"][1]+0.4, "C2")

ax.add_patch(plt.Circle((185, 72), 4, fill=False))
ax.add_patch(plt.Circle((170, 56), 4, fill=False))

ax.text(185, 78, "K = 1", ha="center")
ax.text(170, 62, "K = 2", ha="center")
ax.axis("off")

# ---------- STEP 2 ----------
ax = axs[0, 1]
ax.set_title("Step 2: After C3 & C4")

for c in ["C2", "C3"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

for c in ["C1", "C4"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

ax.add_patch(plt.Circle((169, 58), 6, fill=False))
ax.add_patch(plt.Circle((182, 70), 5, fill=False))

ax.text(169, 66, "K = 2", ha="center")
ax.text(182, 77, "K = 1", ha="center")
ax.axis("off")

# ---------- STEP 3 ----------
ax = axs[1, 0]
ax.set_title("Step 3: After C5 & C6")

for c in ["C2", "C3"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

for c in ["C1", "C4", "C5", "C6"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

ax.add_patch(plt.Circle((169, 58), 6, fill=False))
ax.add_patch(plt.Circle((183, 72), 7, fill=False))

ax.text(169, 66, "K = 2", ha="center")
ax.text(183, 81, "K = 1", ha="center")
ax.axis("off")

# ---------- STEP 4 ----------
ax = axs[1, 1]
ax.set_title("Step 4: Final Clusters")

for c in ["C2", "C3"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

for c in ["C1", "C4", "C5", "C6"]:
    ax.scatter(*points[c])
    ax.text(points[c][0]+0.4, points[c][1]+0.4, c)

ax.add_patch(plt.Circle((169, 58), 6, fill=False))
ax.add_patch(plt.Circle((183, 72), 7, fill=False))

ax.text(169, 66, "K = 2", ha="center")
ax.text(183, 81, "K = 1", ha="center")
ax.axis("off")

plt.tight_layout()
plt.show()

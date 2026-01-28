import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([
    [8.0, 160],   # Movie 1
    [6.2, 170],   # Movie 2
    [7.2, 168],   # Movie 3
    [8.2, 155]    # Movie 4
])

# Labels
genres = np.array(["Action", "Action", "Comedy", "Comedy"])

# New movie
new_movie = np.array([7.4, 114])


# Euclidean distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Calculate distances
distances = []
for i in range(len(X)):
    d = euclidean_distance(new_movie, X[i])
    distances.append((d, genres[i], X[i]))

# Sort distances
distances.sort(key=lambda x: x[0])

# Select K
k = 3
nearest = distances[:k]

# Majority voting
labels = [g for _, g, _ in nearest]
prediction = max(set(labels), key=labels.count)

print("Distances (sorted):")
for d, g, point in distances:
    print(f"Point {point} -> Distance = {d:.2f}, Genre = {g}")

print("\nK Nearest Neighbors:")
for d, g, _ in nearest:
    print(f"Distance = {d:.2f}, Genre = {g}")

print("\nPredicted Genre:", prediction)

# GRAPH
for i in range(len(X)):
    if genres[i] == "Action":
        plt.scatter(X[i, 0], X[i, 1], marker='o')
        plt.text(X[i, 0]+0.03, X[i, 1]+1, "Action")
    else:
        plt.scatter(X[i, 0], X[i, 1], marker='s')
        plt.text(X[i, 0]+0.03, X[i, 1]+1, "Comedy")

# New movie
plt.scatter(new_movie[0], new_movie[1], marker='X', s=150)
plt.text(new_movie[0]+0.03, new_movie[1]+1, "New Movie")

plt.xlabel("IMDb Rating")
plt.ylabel("Duration (minutes)")
plt.title("KNN Movie Genre Classification")
plt.show()

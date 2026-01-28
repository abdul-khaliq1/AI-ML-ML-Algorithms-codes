import numpy as np

# Step 1: Given data

X = np.array([
    [4, 11],
    [8, 4],
    [13, 5],
    [7, 14]
])

print("Original Data:\n", X)


# Step 2: Mean centering

mean = np.mean(X, axis=0)
X_centered = X - mean

print("\nMean of features:", mean)
print("\nMean Centered Data:\n", X_centered)


# Step 3: Covariance matrix

cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)


# Step 4: Eigenvalues & Eigenvectors

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)


# Step 5: Sort eigenvalues (descending)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)


# Step 6: First Principal Component

pc1 = eigenvectors[:, 0]
print("\nFirst Principal Component (PC1):\n", pc1)


# Step 7: Project data onto PC1

pc1_scores = X_centered @ pc1
print("\nFirst Principal Component Scores:\n", pc1_scores)

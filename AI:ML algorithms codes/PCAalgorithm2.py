import numpy as np
# Given data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

print("Original Data:\n", X)

# Step 1: Mean centering
mean = np.mean(X, axis=0)
X_centered = X - mean

print("\nMean:", mean)
print("\nMean Centered Data:\n", X_centered)

# Step 2: Covariance matrix
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 3: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 4: Select largest eigenvalue (Principal Component)
idx = np.argmax(eigenvalues)
principal_vector = eigenvectors[:, idx]

print("\nPrincipal Eigenvector (PC1):\n", principal_vector)

# Step 5: Project data (2D → 1D)
Z = X_centered.dot(principal_vector)
print("\nReduced 1D Data:\n", Z)

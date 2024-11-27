import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load the joke rating data
data = pd.read_csv("./data/train.csv")

# Create a user-item rating matrix
rating_matrix = data.pivot(index="user_id", columns="joke_id", values="Rating").fillna(0)

# Ensure the matrix is numeric
rating_matrix = rating_matrix.astype(float)

# Convert to a sparse matrix
rating_matrix_sparse = csr_matrix(rating_matrix.values)

# Perform SVD
# Set k to the number of latent factors you want to extract
k = 120
U, sigma, VT = svds(rating_matrix_sparse, k=k)

# Convert sigma to a diagonal matrix
sigma_diag = np.diag(sigma)

# Calculate user and item latent vectors
user_latent_vectors = np.dot(U, sigma_diag)  # Shape: (num_users, k)
item_latent_vectors = VT.T  # Shape: (num_items, k)

# Print the results
print("User latent vectors:")
print(user_latent_vectors)

print("\nItem latent vectors:")
print(item_latent_vectors)

np.save("./data/user_latent_vectors_svd.npy", user_latent_vectors)
np.save("./data/item_latent_vectors_svd.npy", item_latent_vectors)

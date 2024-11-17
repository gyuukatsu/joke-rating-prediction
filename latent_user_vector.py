import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
import numpy as np

# Load the joke rating data
data = pd.read_csv("./data/train.csv")

# Assuming the data has columns 'user_id', 'joke_id', and 'rating'
user_joke_matrix = data.pivot(index="user_id", columns="joke_id", values="Rating").fillna(0)

# Choose the number of latent factors
n_latent_factors = 100

# Using SVD for matrix factorization
svd = TruncatedSVD(n_components=n_latent_factors)
user_latent_vectors_svd = svd.fit_transform(user_joke_matrix)

# Using NMF for matrix factorization
# nmf = NMF(n_components=n_latent_factors, init="random", random_state=0)
# user_latent_vectors_nmf = nmf.fit_transform(user_joke_matrix)

# Save the latent user vectors
np.save("./data/user_latent_vectors_svd.npy", user_latent_vectors_svd)
# np.save("./data/user_latent_vectors_nmf.npy", user_latent_vectors_nmf)

print("Latent user vectors have been generated and saved.")

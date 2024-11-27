import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

jokes = pd.read_csv("./data/jokes.csv")
rationales = json.load(open("./data/rationales.json"))

sentences = []
for i in range(len(rationales["rationales"])):
    # sentence = jokes["joke_text"][i]
    sentence = jokes["joke_text"][i] + " " + rationales["rationales"][i]
    # sentence = rationales["rationales"][i]
    sentences.append(sentence)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
print(embeddings.shape)

# np.save("./data/joke_embeddings.npy", embeddings)
np.save("./data/joke_rationale_embeddings.npy", embeddings)
# np.save("./data/rationale_embeddings.npy", embeddings)

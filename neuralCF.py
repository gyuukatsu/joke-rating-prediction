import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim


class NCF(nn.Module):
    def __init__(self, user_embedding, joke_embedding, dropout_rate=0.3):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding.from_pretrained(
            torch.tensor(user_embedding, dtype=torch.float32), freeze=True
        )
        # self.user_embedding = nn.Embedding(user_embedding.shape[0], user_embedding.shape[1])
        self.joke_embedding = nn.Embedding.from_pretrained(
            torch.tensor(joke_embedding, dtype=torch.float32), freeze=True
        )
        user_emb_dim = user_embedding.shape[1]
        joke_emb_dim = joke_embedding.shape[1]
        self.ncf_layers = nn.Sequential(
            nn.Linear(user_emb_dim + joke_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )
        # self.output_scale = nn.Tanh()

    def forward(self, user, joke):
        user_embedded = self.user_embedding(user)
        joke_embedded = self.joke_embedding(joke)
        x = torch.cat([user_embedded, joke_embedded], dim=-1)
        x = self.ncf_layers(x)
        # x = self.output_scale(x) * 10  # Scale the output to be between -10 and 10
        return x


def neuralCF_train(data, user_embedding, joke_embedding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split the data into train and test
    train, test = train_test_split(data, test_size=0.1, shuffle=True, random_state=42)
    X_train = train.drop("Rating", axis=1)
    y_train = train["Rating"]
    X_test = test.drop("Rating", axis=1)
    y_test = test["Rating"]

    model = NCF(user_embedding, joke_embedding).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    train_dataset = TensorDataset(
        torch.tensor(X_train["user_id"].values, dtype=torch.long),
        torch.tensor(X_train["joke_id"].values, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(X_test["user_id"].values, dtype=torch.long),
        torch.tensor(X_test["joke_id"].values, dtype=torch.long),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    num_epochs = 100
    early_stopping_patience = 3
    best_loss = float("inf")
    epoch_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user, joke, rating in train_loader:
            user, joke, rating = user.to(device), joke.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, joke).squeeze()
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for user, joke, rating in test_loader:
                user, joke, rating = user.to(device), joke.to(device), rating.to(device)
                output = model(user, joke).squeeze()
                loss = torch.sqrt(criterion(output, rating))  # RMSE
                total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Test RMSE: {avg_test_loss:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "./model/ncf.pth")
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve == early_stopping_patience:
                print("Early stopping")
                break

        # scheduler.step(avg_test_loss)


def neuralCF_inference(data, user_embedding, joke_embedding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NCF(user_embedding, joke_embedding).to(device)
    model.load_state_dict(torch.load("./model/ncf.pth"))
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(data["user_id"].values, dtype=torch.long),
        torch.tensor(data["joke_id"].values, dtype=torch.long),
    )

    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    predictions = []

    with torch.no_grad():
        for user, joke in test_loader:
            user, joke = user.to(device), joke.to(device)
            output = model(user, joke).squeeze()
            output = torch.clamp(output, min=-10, max=10)
            predictions.extend(output.cpu().detach().numpy())

    data["Rating"] = predictions
    data.drop(["user_id", "joke_id"], axis=1, inplace=True)
    data.to_csv("./data/submission_ncf.csv", index=False)
    print("Inference completed.")


if __name__ == "__main__":
    train_data = pd.read_csv("./data/train.csv")
    train_data["joke_id"] = train_data["joke_id"] - 1
    train_data["user_id"] = train_data["user_id"] - 1
    assert train_data["user_id"].min() >= 0 and train_data["joke_id"].min() >= 0, "Negative IDs detected"

    user_embedding_matrix = np.load("./data/trained_user_embeddings.npy")
    joke_embedding_matrix = np.load("./data/trained_joke_embeddings.npy")
    # joke_embedding_matrix = np.load("./data/rationale_embeddings.npy")
    # user_embedding_matrix = np.zeros((train_data["user_id"].nunique(), 32))

    # for user_id in train_data["user_id"].unique():
    #     user_data = train_data[train_data["user_id"] == user_id]
    #     joke_ids = user_data["joke_id"].values
    #     ratings = user_data["Rating"].values
    #     ratings_normalized = ratings / np.linalg.norm(ratings)
    #     weighted_joke_embeddings = joke_embedding_matrix[joke_ids] * ratings[:, np.newaxis]
    #     user_embedding_matrix[user_id] = weighted_joke_embeddings.sum(axis=0)

    neuralCF_train(train_data, user_embedding_matrix, joke_embedding_matrix)

    test_data = pd.read_csv("./data/test.csv")
    test_data["joke_id"] = test_data["joke_id"] - 1
    test_data["user_id"] = test_data["user_id"] - 1
    assert test_data["user_id"].min() >= 0 and test_data["joke_id"].min() >= 0, "Negative IDs detected"

    neuralCF_inference(test_data, user_embedding_matrix, joke_embedding_matrix)

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ContrastiveTwoTowerModel(nn.Module):
    def __init__(self, user_embedding, joke_embedding, dropout_rate=0.3):
        super(ContrastiveTwoTowerModel, self).__init__()

        self.user_embedding = nn.Embedding.from_pretrained(
            torch.tensor(user_embedding, dtype=torch.float32), freeze=True
        )
        # self.user_embedding = nn.Embedding(user_embedding.shape[0], user_embedding.shape[1])
        self.joke_embedding = nn.Embedding.from_pretrained(
            torch.tensor(joke_embedding, dtype=torch.float32), freeze=True
        )
        user_emb_dim = user_embedding.shape[1]
        joke_emb_dim = joke_embedding.shape[1]
        self.user_fc = nn.Sequential(
            nn.Linear(user_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
        )
        self.joke_fc = nn.Sequential(
            nn.Linear(joke_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
        )
        self.prediction_fc = nn.Sequential(nn.Linear(64 + 64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, user, joke):
        user_embedded = self.user_embedding(user)
        joke_embedded = self.joke_embedding(joke)

        user_features = self.user_fc(user_embedded)
        joke_features = self.joke_fc(joke_embedded)

        combined = torch.cat([user_features, joke_features], dim=1)
        rating_prediction = self.prediction_fc(combined)
        return user_features, joke_features, rating_prediction


def hybrid_loss(user_features, joke_features, predicted_rating, actual_rating, margin=1.0):
    # Contrastive loss (optional: normalize embeddings for cosine similarity)
    user_features = F.normalize(user_features, dim=1)
    joke_features = F.normalize(joke_features, dim=1)
    similarity = torch.cosine_similarity(user_features, joke_features, dim=1)

    contrastive = F.mse_loss(similarity, actual_rating / 10)  # Scale ratings to [-1, 1]

    # Regression loss
    regression = F.mse_loss(predicted_rating.squeeze(), actual_rating)

    # Combine the losses
    return 0.5 * contrastive + 0.5 * regression


def twoTower_train(data, user_embedding_matrix, joke_embedding_matrix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, test = train_test_split(data, test_size=0.1, shuffle=True, random_state=42)
    X_train = train.drop("Rating", axis=1)
    y_train = train["Rating"]
    X_test = test.drop("Rating", axis=1)
    y_test = test["Rating"]

    model = ContrastiveTwoTowerModel(user_embedding_matrix, joke_embedding_matrix, dropout_rate=0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_dataset = TensorDataset(
        torch.tensor(X_train["user_id"].values, dtype=torch.int32),
        torch.tensor(X_train["joke_id"].values, dtype=torch.int32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(X_test["user_id"].values, dtype=torch.int32),
        torch.tensor(X_test["joke_id"].values, dtype=torch.int32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    num_epochs = 100
    early_stopping_patience = 3
    best_test_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user, joke, rating in train_loader:
            user, joke, rating = user.to(device), joke.to(device), rating.to(device)
            optimizer.zero_grad()
            user_features, joke_features, predicted_rating = model(user, joke)
            loss = hybrid_loss(user_features, joke_features, predicted_rating, rating)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for user, joke, rating in test_loader:
                user, joke, rating = user.to(device), joke.to(device), rating.to(device)
                user_features, joke_features, predicted_rating = model(user, joke)
                loss = hybrid_loss(user_features, joke_features, predicted_rating, rating)
                total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), "./model/two_tower_contrastive.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print("Early stopping")
                break

        # Adjust learning rate
        scheduler.step(avg_test_loss)


def twoTower_inference(data, user_embedding_matrix, joke_embedding_matrix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = ContrastiveTwoTowerModel(user_embedding_matrix, joke_embedding_matrix, dropout_rate=0).to(device)
    model.load_state_dict(torch.load("./model/two_tower_contrastive.pth"))
    model.eval()

    # Prepare test dataset
    test_dataset = TensorDataset(
        torch.tensor(data["user_id"].values, dtype=torch.int32),
        torch.tensor(data["joke_id"].values, dtype=torch.int32),
    )
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    predictions = []

    with torch.no_grad():
        for user, joke in test_loader:
            user, joke = user.to(device), joke.to(device)
            _, _, predicted_rating = model(user, joke)
            predictions.extend(predicted_rating.squeeze().cpu().numpy())

    data["Rating"] = predictions
    data.drop(["user_id", "joke_id"], axis=1, inplace=True)
    data.to_csv("./data/submission_two_tower_contrastive.csv", index=False)
    print("Inference complete. Predictions saved to './data/submission_two_tower_contrastive.csv'")


if __name__ == "__main__":
    # Load train data
    train_data = pd.read_csv("./data/train.csv")
    train_data["user_id"] = train_data["user_id"] - 1
    train_data["joke_id"] = train_data["joke_id"] - 1

    # user_embedding_matrix = np.load("./data/user_latent_vectors_svd.npy")
    joke_embedding_matrix = np.load("./data/rationale_embeddings.npy")
    user_embedding_matrix = np.zeros((train_data["user_id"].nunique(), joke_embedding_matrix.shape[1]))

    for user_id in train_data["user_id"].unique():
        user_data = train_data[train_data["user_id"] == user_id]
        joke_ids = user_data["joke_id"].values
        ratings = user_data["Rating"].values
        ratings_normalized = ratings / np.linalg.norm(ratings)
        weighted_joke_embeddings = joke_embedding_matrix[joke_ids] * ratings[:, np.newaxis]
        user_embedding_matrix[user_id] = weighted_joke_embeddings.sum(axis=0)

    twoTower_train(train_data, user_embedding_matrix, joke_embedding_matrix)

    # Load test data
    test_data = pd.read_csv("./data/test.csv")
    test_data["user_id"] = test_data["user_id"] - 1
    test_data["joke_id"] = test_data["joke_id"] - 1
    assert test_data["user_id"].min() >= 0 and test_data["joke_id"].min() >= 0, "Negative IDs detected"

    twoTower_inference(test_data, user_embedding_matrix, joke_embedding_matrix)

# tmux attach -t joke-rating-prediction

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim


class DeepFM(nn.Module):
    def __init__(self, num_users, num_jokes, embedding_dim, joke_embedding):
        super(DeepFM, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.joke_embedding = nn.Embedding.from_pretrained(
            torch.tensor(joke_embedding, dtype=torch.float32), freeze=False
        )
        self.fm_first_order_user = nn.Embedding(num_users, 1)
        self.fm_first_order_joke = nn.Embedding(num_jokes, 1)
        self.fm_second_order_user = nn.Embedding(num_users, embedding_dim)
        self.fm_second_order_joke = nn.Embedding(num_jokes, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.output_scale = nn.Tanh()

    def forward(self, user, joke):
        user_embedded = self.user_embedding(user)
        joke_embedded = self.joke_embedding(joke)

        fm_first_order_user = self.fm_first_order_user(user).squeeze()
        fm_first_order_joke = self.fm_first_order_joke(joke).squeeze()
        fm_first_order = fm_first_order_user + fm_first_order_joke

        fm_second_order_user = self.fm_second_order_user(user)
        fm_second_order_joke = self.fm_second_order_joke(joke)
        fm_second_order = torch.sum(fm_second_order_user * fm_second_order_joke, dim=1)

        x = torch.cat([user_embedded, joke_embedded], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze()

        output = fm_first_order + fm_second_order + x
        output = self.output_scale(output) * 10  # Scale the output to be between -10 and 10
        return output


def deepFM_train(data, joke_embedding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, test = train_test_split(data, test_size=0.2)
    X_train = train.drop("Rating", axis=1)
    y_train = train["Rating"]
    X_test = test.drop("Rating", axis=1)
    y_test = test["Rating"]

    num_users = data["user_id"].nunique()
    num_jokes = data["joke_id"].nunique()
    embedding_dim = joke_embedding.shape[1]

    model = DeepFM(num_users, num_jokes, embedding_dim, joke_embedding).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.tensor(X_train["user_id"].values, dtype=torch.long),
        torch.tensor(X_train["joke_id"].values, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(X_test["user_id"].values, dtype=torch.long),
        torch.tensor(X_test["joke_id"].values, dtype=torch.long),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_epochs = 100
    early_stopping_patience = 3
    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user, joke, rating in train_loader:
            user, joke, rating = user.to(device), joke.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, joke)
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
                output = model(user, joke)
                loss = criterion(output, rating)
                total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "./model/deepfm.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print("Early stopping")
                break


def deepFM_inference(data, joke_embedding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_users = data["user_id"].nunique()
    num_jokes = data["joke_id"].nunique()
    embedding_dim = joke_embedding.shape[1]

    model = DeepFM(num_users, num_jokes, embedding_dim, joke_embedding).to(device)
    model.load_state_dict(torch.load("./model/deepfm.pth"))
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(data["user_id"].values, dtype=torch.long),
        torch.tensor(data["joke_id"].values, dtype=torch.long),
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    predictions = []

    with torch.no_grad():
        for user, joke in test_loader:
            user, joke = user.to(device), joke.to(device)
            output = model(user, joke)
            predictions.extend(output.cpu().detach().numpy())

    data["Rating"] = predictions
    data.drop(["user_id", "joke_id"], axis=1, inplace=True)
    data.to_csv("./data/submission_deepfm.csv", index=False)


if __name__ == "__main__":
    train_data = pd.read_csv("./data/train.csv")
    train_data["joke_id"] = train_data["joke_id"] - 1
    train_data["user_id"] = train_data["user_id"] - 1
    assert train_data["user_id"].min() >= 0 and train_data["joke_id"].min() >= 0, "Negative IDs detected"

    test_data = pd.read_csv("./data/test.csv")
    test_data["joke_id"] = test_data["joke_id"] - 1
    test_data["user_id"] = test_data["user_id"] - 1
    assert test_data["user_id"].min() >= 0 and test_data["joke_id"].min() >= 0, "Negative IDs detected"

    joke_embedding = np.load("./data/joke_rationale_embeddings.npy")

    deepFM_train(train_data, joke_embedding)
    deepFM_inference(test_data, joke_embedding)

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim


class NCF(nn.Module):
    def __init__(self, num_users, num_jokes, embedding_dim, joke_embedding):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.joke_embedding = nn.Embedding.from_pretrained(
            torch.tensor(joke_embedding, dtype=torch.float32), freeze=False  # Make it trainable
        )
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.output_scale = nn.Tanh()

    def forward(self, user, joke):
        user_embedded = self.user_embedding(user)
        joke_embedded = self.joke_embedding(joke)
        x = torch.cat([user_embedded, joke_embedded], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.output_scale(x) * 10  # Scale the output to be between -10 and 10
        return x


def neuralCF_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = pd.read_csv("./data/train.csv")
    data["joke_id"] = data["joke_id"] - 1
    data["user_id"] = data["user_id"] - 1
    assert data["user_id"].min() >= 0 and data["joke_id"].min() >= 0, "Negative IDs detected"

    # Split the data into train and test
    train, test = train_test_split(data, test_size=0.2)
    X_train = train.drop("Rating", axis=1)
    y_train = train["Rating"]
    X_test = test.drop("Rating", axis=1)
    y_test = test["Rating"]

    joke_embedding = np.load("./data/joke_embeddings.npy")

    num_users = data["user_id"].nunique()
    num_jokes = data["joke_id"].nunique()
    embedding_dim = joke_embedding.shape[1]

    model = NCF(num_users, num_jokes, embedding_dim, joke_embedding).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.tensor(X_train["user_id"].values, dtype=torch.long),
        torch.tensor(X_train["joke_id"].values, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_epochs = 10
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

    model.eval()
    test_dataset = TensorDataset(
        torch.tensor(X_test["user_id"].values, dtype=torch.long),
        torch.tensor(X_test["joke_id"].values, dtype=torch.long),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_loss = 0
    for user, joke, rating in test_loader:
        user, joke, rating = user.to(device), joke.to(device), rating.to(device)
        output = model(user, joke).squeeze()
        loss = criterion(output, rating)
        total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_loader)}")

    # Save the model
    torch.save(model.state_dict(), "./model/ncf.pth")


def neuralCF_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    real_test = pd.read_csv("./data/test.csv")
    real_test["joke_id"] = real_test["joke_id"] - 1
    real_test["user_id"] = real_test["user_id"] - 1
    assert real_test["user_id"].min() >= 0 and real_test["joke_id"].min() >= 0, "Negative IDs detected"

    joke_embedding = np.load("./data/joke_embeddings.npy")

    num_users = real_test["user_id"].nunique()
    num_jokes = real_test["joke_id"].nunique()
    embedding_dim = joke_embedding.shape[1]

    model = NCF(num_users, num_jokes, embedding_dim, joke_embedding).to(device)
    model.load_state_dict(torch.load("./model/ncf.pth"))
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(real_test["user_id"].values, dtype=torch.long),
        torch.tensor(real_test["joke_id"].values, dtype=torch.long),
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    predictions = []

    for user, joke in test_loader:
        user, joke = user.to(device), joke.to(device)
        output = model(user, joke).squeeze()
        predictions.extend(output.cpu().detach().numpy())

    real_test["Rating"] = predictions
    real_test.drop(["user_id", "joke_id"], axis=1, inplace=True)
    real_test.to_csv("./data/submission_ncf.csv", index=False)


if __name__ == "__main__":
    neuralCF_train()
    neuralCF_inference()

    # tmux attach -t wideDeepRec

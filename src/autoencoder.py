import os
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from nih_dataset import NIHDataset, get_split_indices

data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()

# keep only X-rays whose label is exactly "No Finding" (train for learning, val for checking)
normal_train = [i for i in train_idx if data.df["Finding Labels"][i] == "No Finding"]
normal_val = [i for i in val_idx if data.df["Finding Labels"][i] == "No Finding"]


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # squeezer: each Conv with stride 2 halves the size: 224 -> 112 -> 56 -> 28 -> 14
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )

        # rebuilder: each ConvTranspose doubles the size: 14 -> 28 -> 56 -> 112 -> 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # 3 channels back, like the input
        )

    def forward(self, x):
        small = self.encoder(x)      # squeeze
        return self.decoder(small)   # rebuild


# runs only when you run this file directly (python src/autoencoder.py)
if __name__ == "__main__":
    print(len(normal_train), len(normal_val))

    train_loader = DataLoader(Subset(data, normal_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(data, normal_val), batch_size=32, shuffle=False)

    model = AutoEncoder()
    loss_function = nn.MSELoss()   # how different is the rebuild from the original?
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):

        # TRAINING: the model practises copying normal X-rays
        model.train()
        running_loss = 0

        for images, _ in train_loader:        # the "_" is the label, which we don't need
            rebuilt = model(images)
            loss = loss_function(rebuilt, images)   # the target is the picture itself

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # VALIDATION: normal X-rays it has never practised on, no learning
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, _ in val_loader:
                val_loss += loss_function(model(images), images).item()

        val_loss = val_loss / len(val_loader)
        print("Epoch:", epoch + 1, "Train loss:", avg_loss, "Val loss:", val_loss)

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/autoencoder.pth")
    print("Saved model to results/autoencoder.pth")
import os
import torch
from torch.utils.data import Subset, DataLoader
from nih_dataset import NIHDataset, FINDINGS, get_split_indices
from model import XrayModel

# data + patient split from nih_dataset.py
data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()

train_loader = DataLoader(Subset(data, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(data, val_idx), batch_size=32, shuffle=False)

# rare-problem weights, worked out from TRAIN rows only
rows = data.df.loc[train_idx, "Finding Labels"]
labels = torch.tensor(
    [[1 if f in t.split("|") else 0 for f in FINDINGS] for t in rows],
    dtype=torch.float32
)
positives = labels.sum(dim=0).clamp(min=1)    # X-rays that HAVE each problem
negatives = len(labels) - positives           # X-rays that don't
pos_weight = (negatives / positives).clamp(max=10)  # rarer problem -> bigger weight, capped at 10

# model with 14 outputs, one per problem
model = XrayModel(num_outputs=14,unfreeze_layer4=True)
loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam([
    {"params": model.model.fc.parameters(), "lr": 0.0001},      # new layer, starts random, can learn faster
    {"params": model.model.layer4.parameters(), "lr": 0.00001}  # pretrained block, only gentle nudges
])

epochs = 3

for epoch in range(epochs):

    # TRAINING: the model learns
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # VALIDATION: only checking, no learning
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += loss_function(outputs, labels).item()

    val_loss = val_loss / len(val_loader)
    print("Epoch:", epoch + 1, "Train loss:", avg_loss, "Val loss:", val_loss)

# save the trained model
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/model_nih_ft.pth")
print("Saved model to results/model_nih_ft.pth")
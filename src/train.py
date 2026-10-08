import os
import torch
from dataset import get_loaders
from model import XrayModel

# get the data and the class weights from dataset.py
train_loader, val_loader, weights = get_loaders()

# build the model from model.py
model = XrayModel()

# loss with class weights, so NORMAL mistakes count for more
loss_function = torch.nn.CrossEntropyLoss(weight=weights)

# only the new last layer is trainable, so only give it to the optimizer
optimizer = torch.optim.Adam(model.model.fc.parameters(), lr=0.0001)

epochs = 3

for epoch in range(epochs):

    # TRAINING: the model learns
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()   # clear old nudge directions
        loss.backward()         # work out new nudge directions
        optimizer.step()        # nudge the weights

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # VALIDATION: only checking, no learning
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print("Epoch:", epoch + 1, "Avg loss:", avg_loss, "Val accuracy:", val_accuracy)

# save the trained model (make the folder first if it doesn't exist)
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/model.pth")
print("Saved model to results/model.pth")
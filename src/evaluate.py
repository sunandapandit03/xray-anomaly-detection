import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from dataset import transform
from model import XrayModel

# test images, same resize + normalize as before, nothing random
test_data = datasets.ImageFolder("data/chest_xray/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# build the model, then load the numbers we saved in train.py
model = XrayModel()
model.load_state_dict(torch.load("results/model.pth"))
model.eval()  # testing mode

all_preds = []
all_labels = []

with torch.no_grad():  # only checking, no learning
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # position of the bigger score = the answer
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

correct = sum(p == l for p, l in zip(all_preds, all_labels))
print("Accuracy:", correct / len(all_labels))
print(confusion_matrix(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))  # said PNEUMONIA, was it right?
print("Recall:", recall_score(all_labels, all_preds))        # sick patients caught
print("F1:", f1_score(all_labels, all_preds))                # blend of the two
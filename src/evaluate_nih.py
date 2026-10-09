import torch
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score
from nih_dataset import NIHDataset, FINDINGS, get_split_indices
from model import XrayModel

# TEST X-rays only (patients the model never saw in training or validation)
data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()
test_loader = DataLoader(Subset(data, test_idx), batch_size=32, shuffle=False)

# build the model, then load the fine-tuned numbers saved by train_nih.py
model = XrayModel(num_outputs=14)
model.load_state_dict(torch.load("results/model_nih_ft.pth"))
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():  # only checking, no learning
    for images, labels in test_loader:
        outputs = model(images)
        all_probs.append(torch.sigmoid(outputs))  # scores -> chance between 0 and 1
        all_labels.append(labels)

all_probs = torch.cat(all_probs).numpy()    # glue all batches into one big table
all_labels = torch.cat(all_labels).numpy()

# one AUROC per problem, and how many X-rays in the test set have it
scores = []
for i, name in enumerate(FINDINGS):
    n = int(all_labels[:, i].sum())
    if n == 0:
        print(name, "no examples in this set")  # AUROC needs at least one sick X-ray
    else:
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        scores.append(auc)
        print(name, round(auc, 3), "(", n, "examples )")

print("Average AUROC:", round(sum(scores) / len(scores), 3))
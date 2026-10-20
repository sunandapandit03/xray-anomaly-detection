import torch
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score
from nih_dataset import NIHDataset, FINDINGS, get_split_indices
from model import XrayModel

MODEL_FILE = "results/model_nih_ft8.pth"   # which saved model to score
SPLIT = "test"                              # "val" = practice set, "test" = the real exam (use rarely)

data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()
indices = val_idx if SPLIT == "val" else test_idx
loader = DataLoader(Subset(data, indices), batch_size=32, shuffle=False)

# build the model, then load the saved numbers
model = XrayModel(num_outputs=14)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():  # only checking, no learning
    for images, labels in loader:
        outputs = model(images)
        all_probs.append(torch.sigmoid(outputs))  # scores -> chance between 0 and 1
        all_labels.append(labels)

all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

# one AUROC per problem, and how many X-rays in this set have it
scores = []
for i, name in enumerate(FINDINGS):
    n = int(all_labels[:, i].sum())
    if n == 0:
        print(name, "no examples in this set")
    else:
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        scores.append(auc)
        print(name, round(auc, 3), "(", n, "examples )")

print("Set:", SPLIT, "| Model:", MODEL_FILE)
print("Average AUROC:", round(sum(scores) / len(scores), 3))
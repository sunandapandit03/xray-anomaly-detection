import torch
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score
from nih_dataset import NIHDataset, get_split_indices
from autoencoder import AutoEncoder

# TEST X-rays only (patients the autoencoder never saw)
data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()
test_loader = DataLoader(Subset(data, test_idx), batch_size=32, shuffle=False)

# build the autoencoder, then load the numbers saved by autoencoder.py
model = AutoEncoder()
model.load_state_dict(torch.load("results/autoencoder.pth"))
model.eval()

# anomaly score = how badly the copy differs from the original, one number per X-ray
scores = []
with torch.no_grad():
    for images, _ in test_loader:
        rebuilt = model(images)
        error = ((rebuilt - images) ** 2).mean(dim=(1, 2, 3))  # average over all pixels
        scores.extend(error.tolist())

# truth for each test X-ray: 0 = normal ("No Finding"), 1 = abnormal (has any finding)
truth = [0 if data.df["Finding Labels"][i] == "No Finding" else 1 for i in test_idx]

normal_scores = [s for s, t in zip(scores, truth) if t == 0]
abnormal_scores = [s for s, t in zip(scores, truth) if t == 1]

print("Test X-rays:", len(truth), "abnormal:", sum(truth))
print("Average error, normal:", sum(normal_scores) / len(normal_scores))
print("Average error, abnormal:", sum(abnormal_scores) / len(abnormal_scores))
print("AUROC:", roc_auc_score(truth, scores))
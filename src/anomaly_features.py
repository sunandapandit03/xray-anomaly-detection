import torch
from torch.utils.data import Subset, DataLoader
from torchvision import models
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from nih_dataset import NIHDataset, get_split_indices

data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()

# the "what normal looks like" collection: normal TRAIN X-rays only
normal_train = [i for i in train_idx if data.df["Finding Labels"][i] == "No Finding"]

# ResNet18 with ImageNet knowledge only
# Identity() removes the last layer, so the output is the 512-number fingerprint
model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Identity()
model.eval()


def get_features(indices):
    # run X-rays through ResNet and collect their fingerprints
    loader = DataLoader(Subset(data, indices), batch_size=32, shuffle=False)
    out = []
    with torch.no_grad():
        for images, _ in loader:
            out.append(model(images))
    return torch.cat(out).numpy()


normal_features = get_features(normal_train)   # fingerprints of normal X-rays
test_features = get_features(test_idx)         # CHANGED: the real exam, test X-rays

# for each test X-ray: average distance to its 5 closest normal fingerprints
knn = NearestNeighbors(n_neighbors=5).fit(normal_features)
distances, _ = knn.kneighbors(test_features)
scores = distances.mean(axis=1)

# CHANGED: truth for the test X-rays: 0 = normal ("No Finding"), 1 = abnormal (has any finding)
truth = [0 if data.df["Finding Labels"][i] == "No Finding" else 1 for i in test_idx]

normal_scores = [s for s, t in zip(scores, truth) if t == 0]
abnormal_scores = [s for s, t in zip(scores, truth) if t == 1]

print("Test X-rays:", len(truth), "abnormal:", sum(truth))
print("Average distance, normal:", sum(normal_scores) / len(normal_scores))
print("Average distance, abnormal:", sum(abnormal_scores) / len(abnormal_scores))
print("AUROC:", roc_auc_score(truth, scores))
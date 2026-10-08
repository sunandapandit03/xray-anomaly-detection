import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_loaders():
    train_data = datasets.ImageFolder("data/chest_xray/train", transform=transform)

    indices = list(range(len(train_data)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=train_data.targets,
        random_state=42
    )

    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(train_data, val_idx), batch_size=32, shuffle=False)

    train_labels = [train_data.targets[i] for i in train_idx]
    counts = torch.bincount(torch.tensor(train_labels))
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()

    return train_loader, val_loader, weights


if __name__ == "__main__":
    train_loader, val_loader, weights = get_loaders()
    print(len(train_loader.dataset), len(val_loader.dataset))
    print(weights)
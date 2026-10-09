import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# the 14 problems, in a fixed order (box 1 always means the same problem)
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

CSV_PATH = "data/nih_sample/nih/sample/sample_labels.csv"
IMAGE_FOLDER = "data/nih_sample/nih/sample/images/"

# every image goes through these 3 steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # make all images the same size
    transforms.ToTensor(),           # picture -> numbers between 0 and 1
    transforms.Normalize(            # adjust numbers the way ResNet expects
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class NIHDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, image_folder=IMAGE_FOLDER, findings=FINDINGS):
        # runs once: just store the CSV table, the picture folder and the problem names
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.findings = findings

    def __len__(self):
        # "how many X-rays do you have?" -> one per CSV row
        return len(self.df)

    def __getitem__(self, i):
        # "give me X-ray number i" -> the picture and its 14 ticks
        name = self.df["Image Index"][i]
        text = self.df["Finding Labels"][i]

        # open the picture, make it 3 channels (ResNet needs that), run the pipeline
        image = Image.open(self.image_folder + name).convert("RGB")
        image = transform(image)

        # text like "Cardiomegaly|Edema" -> 14 numbers of 0 and 1
        parts = text.split("|")
        label = [1 if f in parts else 0 for f in self.findings]
        label = torch.tensor(label, dtype=torch.float32)  # decimals, the new loss needs them

        return image, label


def get_split_indices(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    # split PATIENTS, not images, so one person never lands in two groups
    patients = df["Patient ID"].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.3, random_state=42)  # 70% train
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)      # 15% val, 15% test

    # row numbers of every X-ray that belongs to each group of patients
    train_idx = df.index[df["Patient ID"].isin(train_p)].tolist()
    val_idx = df.index[df["Patient ID"].isin(val_p)].tolist()
    test_idx = df.index[df["Patient ID"].isin(test_p)].tolist()

    return train_idx, val_idx, test_idx


# runs only when you run this file directly, quick self-check
if __name__ == "__main__":
    data = NIHDataset()
    image, label = data[4]
    print(len(data), image.shape)
    print(label)

    train_idx, val_idx, test_idx = get_split_indices()
    print(len(train_idx), len(val_idx), len(test_idx))
    print(len(train_idx) + len(val_idx) + len(test_idx))
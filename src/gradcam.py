import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nih_dataset import NIHDataset, FINDINGS, get_split_indices
from model import XrayModel

data = NIHDataset()
train_idx, val_idx, test_idx = get_split_indices()

# unfreeze_layer4=True so PyTorch can trace the Effusion score back through layer4
model = XrayModel(num_outputs=14, unfreeze_layer4=True)
model.load_state_dict(torch.load("results/model_nih_ft.pth"))
model.eval()

# find the first TEST X-ray that truly has Effusion
effusion = FINDINGS.index("Effusion")
for i in test_idx:
    image, label = data[i]
    if label[effusion] == 1:
        break
name = data.df["Image Index"][i]

# a "hook" saves layer4's output (the 7x7 grid) whenever the model runs
saved = {}
def save_output(module, inp, out):
    saved["grid"] = out
model.model.layer4.register_forward_hook(save_output)

# run the X-ray through the model and take the Effusion score
outputs = model(image.unsqueeze(0))
score = outputs[0, effusion]

# for each square of the grid: how much does the Effusion score change if it changes?
grads = torch.autograd.grad(score, saved["grid"])[0]
grid = saved["grid"].detach()

# average importance of each of the 512 feature maps, then blend them into one 7x7 heat map
weights = grads.mean(dim=(2, 3), keepdim=True)
heat = F.relu((weights * grid).sum(dim=1, keepdim=True))   # relu keeps only "pushed the score UP"

# stretch the 7x7 map to 224x224 and scale it to 0-1
heat = F.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
heat = (heat / (heat.max() + 1e-8)).detach()

# undo Normalize so the X-ray looks normal on screen
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
picture = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

# draw the X-ray, then the heat map on top (red = looked here)
plt.imshow(picture)
plt.imshow(heat.numpy(), cmap="jet", alpha=0.4)
plt.axis("off")
plt.title("Effusion - " + name)

os.makedirs("results", exist_ok=True)
plt.savefig("results/gradcam_effusion.png")
print("Saved results/gradcam_effusion.png")
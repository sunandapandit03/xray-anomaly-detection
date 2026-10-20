import torch
import gradio as gr
from nih_dataset import FINDINGS, transform
from model import XrayModel

# load the trained Stage 2 model once, when the app starts
model = XrayModel(num_outputs=14)
model.load_state_dict(torch.load("results/model_nih_ft8.pth"))
model.eval()


def predict(picture):
    # same steps as in training: 3 channels, resize, normalize
    image = transform(picture.convert("RGB"))

    # ask the model; sigmoid turns its 14 scores into numbers between 0 and 1
    with torch.no_grad():
        scores = torch.sigmoid(model(image.unsqueeze(0)))[0]

    # gradio wants a dictionary: finding name -> score
    return {name: float(s) for name, s in zip(FINDINGS, scores)}


demo = gr.Interface(
    fn=predict,                                           # what to run on the upload
    inputs=gr.Image(type="pil", label="Upload a chest X-ray"),
    outputs=gr.Label(num_top_classes=5, label="Most likely findings"),
    title="Chest X-ray findings (student demo)",
    description="Research demo only. Not a medical tool. The scores rank findings, they are not exact probabilities.",
)

demo.launch()
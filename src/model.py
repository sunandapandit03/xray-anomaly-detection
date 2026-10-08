# # A neural network is one type of machine-learning model

# import torch
# # PyTorch is the library we'll use to build and train our neural network

# import torch.nn as nn
# # torch.nn contains tools for building neural networks

# class CNN(nn.Module):
#     # I'm creating a neural network called CNN,
#     # nn.Module is PyTorch's basic class for creating neural networks

#     def __init__(self):
#         # __init__() is the setup function for our CNN

#         super().__init__()
#         # Initialize the parent PyTorch neural-network class

#         #LAYER1
#         self.conv1 = nn.Conv2d(3, 16, 3) #1 is channel(x-ray has only 1 channel),16 are filters(16 different pattern detectors),3 is filter size
#         #Conv2d->It looks at small areas of the image and tries to detect patterns

#         #LAYER2
#         self.relu = nn.ReLU()           #ReLU = Rectified Linear Unit(ReLU basically removes negative values and keeps positive ones)

#         #LAYER3
#         self.pool=nn.MaxPool2d(2)       #Pooling reduces the size of the information while trying to keep the important parts

#         self.conv2=nn.Conv2d(16,32,3)   #The second convolution layer can take those patterns and look for more complex patterns

#         #LAYER4
#         self.flatten = nn.Flatten()

#         #LAYER5
#         self.fc = nn.Linear(380192,2) #A Linear layer connects numbers together and learns how important they are for the final prediction.

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x



import torch.nn as nn
from torchvision import models


class XrayModel(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained ResNet18 that already knows edges, textures, shapes
        self.model = models.resnet18(weights="DEFAULT")

        # freeze everything so training can't change the pretrained knowledge
        for param in self.model.parameters():
            param.requires_grad = False

        # swap the last layer: 512 numbers in, 2 scores out (NORMAL, PNEUMONIA)
        # made AFTER freezing, so this new layer stays trainable
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        # image goes through ResNet, 2 scores come out
        return self.model(x)


# runs only when you run this file directly, quick self-check
if __name__ == "__main__":
    import torch

    model = XrayModel()
    fake_batch = torch.zeros(4, 3, 224, 224)  # 4 blank images, just to test the shape
    print(model(fake_batch).shape)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable)
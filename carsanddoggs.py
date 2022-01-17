from locale import normalize
import torch
import torchvision

from torchvision import transforms
from PIL import Image
from os import listdir


normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    )
transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


train_data_list=[]
for f in listdir("catdog/train/"):
    img = Image.open("catdog/train/" + f)
    img_tensor = transform(img)
    train_data_lost.append(img_tensor)

    if len(train_data_list) >= 256:
        break

train_data = torch.stack(train_data_list)
print(train_data)



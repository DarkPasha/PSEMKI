

from locale import normalize
from turtle import forward
import torch
import torchvision

from torchvision import transforms
from PIL import Image
import os

import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
transforms = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


def test():
    model = model.load_state_dict(torch.load('meinNetz.pt'))
    model.eval()
    files = os.listdir('PetImages/tests/')
    print("enter filename")
    f = input("")
    #f = random.choice(files)
    img = Image.open('PetImages/tests/' + f).convert('RGB')
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze(0)
    img_eval_tensor = img_eval_tensor.cuda()
    #print(img_eval_tensor.shape)
    data = Variable(img_eval_tensor)
    out = model(data)
    print("ergebnis", out)
    print(out.data.max(1, keepdim = True)[1]) 
    print("Hund") if((out.data.max(1, keepdim = True)[1])==1) else print("Katze")
    img.show()
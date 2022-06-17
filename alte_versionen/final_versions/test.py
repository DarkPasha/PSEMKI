from importlib.util import module_for_loader
from sklearn import model_selection
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import ImageFont
from PIL import ImageDraw

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
transforms = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6 , kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 12 , kernel_size = 5)
        self.conv3 = nn.Conv2d(12, 18 , kernel_size = 5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size = 5)
        self.conv5 = nn.Conv2d(24, 56, kernel_size = 2)
        self.fc1 = nn.Linear(1400, 1000)
        self.fc2 = nn.Linear(1000, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)  
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, 1400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)



def test():
    model = Netz()
    model = model.load_state_dict(torch.load('meinNetz2.pt'))
    model.eval()
    files = os.listdir('PetImages/tests/')
    f = input("Please enter the filename of the picture you whish to test: ")
    
    directoryPath = 'PetImages/tests/' + f

    if os.path.isfile(directoryPath):      
        #f = random.choice(files)
        img = Image.open('PetImages/tests/' + f).convert('RGB')
        img_eval_tensor = transforms(img)
        img_eval_tensor.unsqueeze(0)
        img_eval_tensor = img_eval_tensor
        #print(img_eval_tensor.shape)
        data = Variable(img_eval_tensor)
        out = model(data)
        #print("ergebnis", out)
        #print(out.data.max(1, keepdim = True)[1]) 
        print("Tierart: Hund") if((out.data.max(1, keepdim = True)[1])==1) else print("Tierart: Katze")
        img.show()
    
    else:          
        print("Filename does not exist!")
        test()

while True:
    test()
    print("Do you want to continue? If yes, please state with YES:")
    if input("") == "YES":
        True
    else:
        break
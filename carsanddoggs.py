#Package Import

from locale import normalize
from turtle import forward
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

#Anpassen der Bilder auf feste Größe

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


train_data_list=[]
target_list = []
train_data=[]


files = listdir("")

#Zufallsreinfolge für die Bilder, damit die KI keine Muster darin erkennt 
#Außerdem Bearbeitungder Bilder, damit die KI ein neuronales Netz entwickeln kann

for i in range(len(listdir("C:\Users\Dani\Desktop\archive\PetImages"))):
    f = random.choice(files)
    files.remove(f)
    img = Image.open("C:\Users\Dani\Desktop\archive\PetImages" + f)
    img_tensor = transform(img)
    train_data_list.append(img_tensor)
    isCat = 1 if "cat" in f else 0
    isDog = 1 if "dog" in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    
#Trainingsepochen:

    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        break

print(train_data_list)
print(target_list)

#Erstellen eines neuronalen Netzes

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6 , kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 12 , kernel_size = 5)
        self.conv3 = nn.Conv2d(12, 18 , kernel_size = 5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size = 5)
        self.fc1 = nn.Linear(3456, 1000)
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
        x = x.view(-1, 14112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

model = Netz()
model.cuda()

#Optimierung des Netzes der KI

optimizer = optim.Adam(model.parameters(), lr = 0.01)
def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()



        batch_id = batch_id + 1

#Testklasse der KI / Ausführung der KI

def test():
    model.eval()
    files = listdir('C:\Users\Dani\Desktop\archive\PetImages')
    f = random.choice(files)
    img = Image.open('C:\Users\Dani\Desktop\archive\PetImages' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim = True)[1])
    x = input("")

#Ausführung der KI mit 30 Trainingsepochen

for epoch in range(1, 30):
    train(epoch)
    test()
#Package Import
#Authors: Emirhan, Daniel
#Final version

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

import matplotlib.pyplot as plt

#Anpassen der Bilder auf feste Größe

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
transforms = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


#train_data_list=[]
#target_list = []
#train_data=[]

test_data_list=[]
test_target_list = []
test_data=[]



#Zufallsreinfolge für die Bilder, damit die KI keine Muster darin erkennt 
#Außerdem Bearbeitungder Bilder, damit die KI ein neuronales Netz entwickeln kann
#print(files)
 
def readBatchsize(batch_size, files):
    train_data_list=[]
    target_list = []
    train_data= []
    
    for i in range(batch_size):
        f = random.choice(files)
        files.remove(f)
        #print(f)
        #print(i)
        img = Image.open("PetImages/training_data/" + f) 
        img_tensor = transforms(img)
        train_data_list.append(img_tensor)
        isCat = 1 if "cat" in f else 0
        isDog = 1 if "dog" in f else 0
        target = [isCat, isDog]
        target_list.append(target)

    train_data.append((torch.stack(train_data_list), target_list))
    return train_data

# if len(train_data_list) >= 7:
#    train_data.append((torch.stack(train_data_list), target_list))
#    train_data_list = []
    

# load test data
testFiles = os.listdir("PetImages/test_data/")
#print("testFiles", files)
for i in range(100):
    f = random.choice(testFiles)
    testFiles.remove(f)
    img = Image.open("PetImages/test_data/" + f) 
    img_tensor = transforms(img)
    test_data_list.append(img_tensor)
    isCat = 1 if "cat" in f else 0
    isDog = 1 if "dog" in f else 0
    target = [isCat, isDog]
    test_target_list.append(target)
    #print(f)
#Trainingsepochen:
test_data.append((torch.stack(test_data_list), test_target_list))
test_data_list = []
    



#print(train_data_list)
#print(target_list)
#print(train_data)
#Erstellen eines neuronalen Netzes

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

model = Netz()
model.cuda()

#Optimierung des Netzes der KI

def error_criterion(out,target):
    #print(out)
    out_max_vals, out_max_indices = torch.max(out,1)
    target_max_vals, target_max_indices = torch.max(target,1)
    #print(out_max_indices)
    #print(target_max_indices)
   # train_error = (out_max_indices != target).sum().data[0]/target_max_indices.size()[0]
    train_error = torch.abs(out_max_indices - target_max_indices).sum().data 
    return train_error

optimizer = optim.Adam(model.parameters(), lr = 0.0005)
def train(epoch, train_data):
    model.train()
    batch_id = 0
    running_train_error= 0.0
    for data, target in train_data:
        #print("training")
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        #print("loss", loss)
        loss.backward()
        optimizer.step()
       # print(out)
        #print(target)
        running_train_error = error_criterion(out,target)
        #print("running_train_error", running_train_error)
        batch_id = batch_id + 1
    return running_train_error




def testModelWithTestData():
    print("testing with testData")
    #print(test_data)
    running_train_error= 0.0
    for data, target in test_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        out = model(data)
        #print("ergebnis", out.data.max(1, keepdim = True)[1]) 
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        running_train_error = error_criterion(out,target)/data.shape[0]
        #print("loss", loss)
    return running_train_error


arr = []
arr2 = []


#Ausführung der KI mit 30 Trainingsepochen

for epoch in range(1, 10):
    print("epoch", epoch)
    i = 0
    files = os.listdir("PetImages/training_data/")
    total_train_error= 0.0
    for batchNumber in range(1, 200):
        train_data = readBatchsize(50, files)
        total_train_error += train(epoch, train_data)
        #print(f)
        #print(len(files))
    torch.save(model, 'meinNetz2.pt')
    test_error = testModelWithTestData()
    print("test_error")
    print(test_error)
    print("totatl_train_error")
    print(total_train_error/10000)
    arr.append((total_train_error/10000).item())
    arr2.append(test_error.item())
    #print(arr)


#plot Graph

x1 = range(1,len(arr)+1)
y1 = arr

x2 =  range(1,len(arr2)+1)
y2 = arr2

plt.ylim(0,0.5)
plt.plot(x1, y1, label = "training_error") 
plt.plot(x2, y2, label = "test_error")
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Error Rate AI - lr 0.0005 - 10 epochs')
plt.legend()
plt.savefig('Error_Rate_AI_3.png')
plt.show()

#Testklasse der KI / Ausführung der KI

def test():
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
    #x = input("")

while True:
    test()

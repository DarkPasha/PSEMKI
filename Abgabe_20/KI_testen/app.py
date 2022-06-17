#Authors: Kartik
#Final version

from cProfile import label
from fileinput import filename
import os
import tkinter as tk
from turtle import onclick
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import tkinter.filedialog as fd

#label_file_explorer = tk.Label()
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
transforms = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])

#print ("In the beginning")
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


def ausführen():
    file = fd.askopenfilename(title='Choose a file of any type', filetypes=[("All files", "*.*")])
    filename = os.path.abspath(file)
    #os.startfile(os.path.abspath(file))
    model = Netz()
    #model = model.load_state_dict(torch.load('meinNetz2.pt'))
    model = torch.load('KI_Programm\\NetzTest.pt')
    model.eval()
    f = filename    
    directoryPath = filename

    if os.path.isfile(directoryPath):      
        #f = random.choice(files)
        img = Image.open(directoryPath).convert('RGB')
        img_eval_tensor = transforms(img)
        img_eval_tensor.unsqueeze(0)
        img_eval_tensor = img_eval_tensor
        #print(img_eval_tensor.shape)
        data = Variable(img_eval_tensor)
        out = model(data)
        #print("ergebnis", out)
        #print(out.data.max(1, keepdim = True)[1]) 
        if((out.data.max(1, keepdim =True)[1])==1):
            tierart = tk.Label(window,
							text = "Dein Tier ist: Hund",
							width = 100, height = 4,
							fg = "blue") 
        else: 
            tierart = tk.Label(window,
							text = "Dein Tier ist: Katze",
							width = 100, height = 4,
							fg = "blue")

        tierart.grid(column = 1, row = 3)
        #img.show()
        
    else:          
        print("Filename does not exist!")
        ausführen() 


if __name__ == "__main__":
    # Create the root window
    window = tk.Tk()
    # Set window title
    window.title('File Explorer')
    # Set window size
    window.geometry("500x500")
    #Set window background color
    window.config(background = "white")
    # Create a File Explorer label
    label_file_explorer = tk.Label(window,
							text = "File Explorer using Tkinter",
							width = 100, height = 4,
							fg = "blue")


    tk.Button(window, text='Open a file', width=20, command=ausführen).place(x=30, y=50)

    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1)
    
    # Let the window wait for any events
    window.mainloop()
    
   
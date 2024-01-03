import numpy as np
import torch
import torch.nn as nn           
import torch.optim as optim      
import torch.nn.functional as F  
from torch.utils.data import DataLoader
import scipy.io
import matplotlib.pyplot as plt
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 5
learning_rate = 0.003

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# Q6.1.3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512*5*5, 625),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(625, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512*5*5)
        x = self.fc_layers(x)
        return x

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

losses = []
accuracy = []

def train(epoch):
    model.train() 
    iteration = 0
    correct = 0
    train_loss = 0
    for ep in range(epoch):
        for _, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            train_loss += loss.item()
            pred = output.max(1)[1] 
            correct += pred.eq(target).sum().item()
            
        train_loss /= iteration
        train_acc = 100*correct/(iteration*batch_size)
        losses.append(train_loss)
        accuracy.append(train_acc)

        print('Epoch: {} \t Accuracy: {:.4f}% \t Loss: {:.4f}'.format(
            ep, train_acc, train_loss))

if __name__ == '__main__':
    train(50)
    plt.plot(range(len(losses)), losses, label="training")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.xlim(0, len(losses) - 1)
    plt.ylim(0, None)
    plt.grid()
    plt.show()

    plt.plot(range(len(accuracy)), accuracy, label="training")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.xlim(0, len(accuracy) - 1)
    plt.ylim(0, None)
    plt.grid()
    plt.show()
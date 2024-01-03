import torch
import torch.nn as nn           
import torch.optim as optim      
import torch.nn.functional as F  
from torch.utils.data import DataLoader
import scipy.io
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
learning_rate = 0.003
hidden_size = 64

# Q6.1.1
train_data = scipy.io.loadmat('hw4/data/nist36_train.mat')
train_x, train_y = train_data["train_data"], train_data["train_labels"]
train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
trainset = torch.utils.data.TensorDataset(train_x, train_y)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(train_x.shape[1], hidden_size)
        self.s = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, train_y.shape[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.s(x)
        x = self.fc2(x)
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
            target = target.max(1)[1]
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
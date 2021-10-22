# MNIST数据集分类
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, re):
        re = self.pool(F.relu(self.conv1(re)))
        re = self.pool(F.relu(self.conv2(re)))
        re = re.view(-1, 6*4*4)
        re = F.relu(self.fc1(re))
        re = self.fc2(re)
        re = F.log_softmax(re, dim=1)
        return re


def mnist_model_train(mnist_model):
    mnist_model.train()
    for index, (data, target) in enumerate(train_loader, 0):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = mnist_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            print('Train: %d/%d\tLoss: %.4f' % (index*len(data), len(train_loader.dataset), loss.item()))
    print('Training finished.')


def mnist_model_test(mnist_model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    mnist_model.eval()
    loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = mnist_model(data)
        loss += criterion(output, target).item()
        # output.max(1, keepdim=True)--->返回每一行中最大的元素并返回索引，返回了两个数组
        # output.max(1, keepdim=True)[1] 就是取第二个数组，取索引数组。
        _, prediction = torch.max(output.data, 1)
        correct += (prediction == target).sum().item()
        c = (prediction == target).squeeze()
        for i in range(8):
            label = target[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    accuracy = correct / len(test_loader.dataset)
    print('Average loss: %.4f\tTotal accuracy: %.4f%%' % (loss, 100*accuracy))
    print('Class accuracy:')
    for i in range(10):
        print('Accuracy of class %d is %.4f%%' % (i, 100*class_correct[i]/class_total[i]))


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    print('MNIST train_set downloaded.')
    test_set = torchvision.datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)
    print('MNIST test_set downloaded.')

    model = MnistNet()
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    mnist_model_train(model)
    mnist_model_test(model)

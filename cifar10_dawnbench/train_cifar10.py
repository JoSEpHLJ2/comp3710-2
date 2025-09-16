import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import argparse

# 启用cuDNN基准测试
torch.backends.cudnn.benchmark = True

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def get_data_loaders(batch_size=512, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True)

    return trainloader, testloader

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 超参数
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    
    trainloader, testloader = get_data_loaders(batch_size=batch_size)
    
    model = ResNet18().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # 训练开始
    start_time = time.time()
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if i % 20 == 19:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], '
                     f'Loss: {running_loss/20:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                running_loss = 0.0
        
        scheduler.step()
        
        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"新的最佳准确率: {best_accuracy:.2f}% - 模型已保存")
    
    training_time = time.time() - start_time
    print(f'总训练时间: {training_time:.2f}秒')
    print(f'最佳准确率: {best_accuracy:.2f}%')
    
    return best_accuracy, training_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAWNBench CIFAR-10 Training')
    parser.add_argument('--batch-size', type=int, default=512, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')  # 改为100
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')     # 建议降低到0.1
    
    args = parser.parse_args()
    
    accuracy, time_taken = train_model(args)
    
    # 保存结果
    with open('training_results.txt', 'w') as f:
        f.write(f'准确率: {accuracy:.2f}%\n')
        f.write(f'训练时间: {time_taken:.2f}秒\n')
        f.write(f'批量大小: {args.batch_size}\n')
        f.write(f'学习率: {args.lr}\n')
        f.write(f'训练轮数: {args.epochs}\n')
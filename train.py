import torch
from torch import nn
from net import MYLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据转换为tensor
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, )

# 测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# GPU
# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else 'cpu'


# 调用net里面定义的模型 将模型数据转到GPU
model = MYLeNet5().to(device)

# 定义一个损失函数
lose_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率 每隔十轮变回0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def trian(dataLoader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataLoader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)   # 输出值计算损失函数
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred)/output.shape[0] # 批次精确度

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        # 单轮累加
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n+1 #总轮次
    #平均精确度和loss
    print("train_loss:"+str(loss/n))
    print("train_acc:"+str(current/n))


# 验证函数 (不用传播)
def val(dataLoader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataLoader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)  # 输出值计算损失函数
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]  # 批次精确度
            # 单轮累加
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1  # 总轮次
            # 平均精确度和loss
        print("test_loss:" + str(loss / n))
        print("test_acc:" + str(current / n))

        return current/n

# 训练
epoch = 50
min_acc = 0  # 最小精确度

for t in range(epoch):
    print(f'epoch{t+1}\n----------------------')
    trian(train_dataloader, model, lose_fn, optimizer)
    a = val(test_dataloader, model, lose_fn)
    # 保存最好模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.makedirs('save_model')
        min_acc = a
        print('save best model start')
        torch.save(model.state_dict(), 'save_model/best_model.pth')

print('end')
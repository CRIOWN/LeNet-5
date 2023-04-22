import torch
from net import MYLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

#test

# 数据转换为tensor
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=16, shuffle=True, )

# 测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# GPU
# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型 将模型数据转到GPU
model = MYLeNet5().to(device)

# 加载模型 填入地址
model.load_state_dict(torch.load("D:/STUDY/Python/Code/LeNet-5/save_model/best_model.pth"))

#
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# tensor转换为图片， 可视化
show = ToPILImage()

# 进去验证
for i in range(20):
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')


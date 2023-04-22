import torch


device = torch.device("cuda")
x = torch.zeros(3, 4, dtype=torch.int, device=device)
print("-----------")
print(x)
print("-----------")
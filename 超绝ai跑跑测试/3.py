import torch
print(torch.cuda.is_available())       # 应该输出 True
print(torch.cuda.get_device_name(0))   # 应该输出你的显卡型号，如 NVIDIA GeForce RTX 4070
import random
import warnings
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")

# 加载原始的MNIST数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
original_dataset = torchvision.datasets.MNIST(root=r'Raw_data', train=True,
                                              download=False, transform=transform)

# 将数据集按标签分开成10份
class_data = [[] for _ in range(10)]
for image, label in original_dataset:
    class_data[int(label)].append((image, label))


# 分配数据集函数，default代表指定数量的相应标签,返回两个列表
def MNIST_Split(default=3000, seed=666):
    random.seed(seed)  # 随机数种子
    custom_dataset, remaining_dataset = [], []

    for i in range(len(class_data)):
        sampled_index = random.sample(list(range(len(class_data[i]))), default)
        unsampled_index = [idx for idx in range(len(class_data[i])) if idx not in sampled_index]
        custom_dataset.append([class_data[i][j] for j in sampled_index])
        remaining_dataset.extend([class_data[i][j] for j in unsampled_index])

    random.shuffle(remaining_dataset)  # 打乱剩余列表
    sp = int(len(original_dataset) / 10 - default)

    return custom_dataset, remaining_dataset, sp


custom_dataset, remaining_dataset, sp = MNIST_Split()

dataloaders = [
    torch.utils.data.DataLoader(custom_dataset[i] + remaining_dataset[sp * i:sp * (i + 1)], batch_size=64, shuffle=True)
    for i in range(len(custom_dataset))]


# 客户端标签分布统计函数
def stat(dataset):
    label_list = []
    for tup in dataset:
        label_list.append(tup[1])
    vc = pd.value_counts(pd.Series(label_list))
    return vc

# stat(dataloaders[0].dataset)

import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=None):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.Conv_Block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Res_Block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.Res_Block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Down_Sample2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.Res_Block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Down_Sample3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.Conv_Block1(x)
        x = self.Res_Block1(x) + x
        x = self.Res_Block2(x) + self.Down_Sample2(x)
        x = self.Res_Block3(x) + self.Down_Sample3(x)
        x = self.Global_avg_pool(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 创建DenseNet模型
    model = ResNet(in_channels=1, num_classes=10)

    # 打印模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")  # 78090

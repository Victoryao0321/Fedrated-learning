import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1):
        super(MBConvBlock, self).__init__()
        self.expand_conv = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.expand_bn = nn.BatchNorm2d(in_channels * expansion_factor)
        self.depthwise_conv = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3,
                                        stride=stride, padding=1, groups=in_channels * expansion_factor,
                                        bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels * expansion_factor)
        self.SEBlock = SEBlock(in_channels * expansion_factor, reduction_ratio=4)
        self.project_conv = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.use_residual = (in_channels == out_channels) and (stride == 1)
        if (in_channels == out_channels) and (stride == 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        x = self.relu(self.depthwise_bn(self.depthwise_conv(x)))
        x = self.SEBlock(x)
        x = self.relu(self.project_bn(self.project_conv(x)))
        if self.use_residual:
            x = x + self.shortcut(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=None):
        self.input_size = input_size
        super(EfficientNet, self).__init__()
        self.Conv_Block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.mbconv1_1 = MBConvBlock(in_channels=16, out_channels=16, expansion_factor=4, stride=1)
        self.mbconv1_2 = MBConvBlock(in_channels=16, out_channels=16, expansion_factor=4, stride=1)
        self.mbconv2_1 = MBConvBlock(in_channels=16, out_channels=32, expansion_factor=4, stride=2)
        self.mbconv2_2 = MBConvBlock(in_channels=32, out_channels=32, expansion_factor=4, stride=1)
        self.mbconv3_1 = MBConvBlock(in_channels=32, out_channels=64, expansion_factor=4, stride=2)
        self.mbconv3_2 = MBConvBlock(in_channels=64, out_channels=64, expansion_factor=4, stride=1)

        self.Global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.Conv_Block(x)
        x = self.mbconv1_1(x)
        x = self.mbconv1_2(x)
        x = self.mbconv2_1(x)
        x = self.mbconv2_2(x)
        x = self.mbconv3_1(x)
        x = self.mbconv3_2(x)
        x = self.Global_avg_pool(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 创建DenseNet模型
    model = EfficientNet(in_channels=1, num_classes=10)

    # 打印模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")  # 132906

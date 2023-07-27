import torch
import torch.nn as nn


class Dense_Block(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(Dense_Block, self).__init__()
        block = []
        channel = in_channels
        for i in range(num_layers):
            block.append(self._make_conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def _make_conv_block(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=None):
        super(DenseNet, self).__init__()
        self.input_size = input_size
        self.Conv_Block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.Dense_Block1 = Dense_Block(in_channels=16, growth_rate=8, num_layers=3)
        self.Dense_Block2 = Dense_Block(in_channels=40, growth_rate=8, num_layers=3)
        self.Transition1 = nn.Sequential(
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1),
            nn.AvgPool2d(2, 2)
        )
        self.Transition2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.AvgPool2d(2, 2)
        )
        self.Global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.Conv_Block1(x)
        x = self.Dense_Block1(x)
        x = self.Transition1(x)
        x = self.Dense_Block2(x)
        x = self.Transition2(x)
        x = self.Global_avg_pool(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 创建DenseNet模型
    model = DenseNet(in_channels=1, num_classes=10)

    # 打印模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")  # 22834

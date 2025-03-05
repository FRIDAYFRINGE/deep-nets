import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu6(self.bn1(self.depthwise(x)))
        x = F.relu6(self.bn2(self.pointwise(x)))
        return x

# MobileNet V1 Model
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(in_channels, out_channels, stride):
            """Standard 3x3 Convolution with BatchNorm and ReLU6"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),   # Conv 1: 224x224x3 -> 112x112x32
            DepthwiseSeparableConv(32, 64, 1),  # Conv 2: 112x112x32 -> 112x112x64
            DepthwiseSeparableConv(64, 128, 2), # Conv 3: 112x112x64 -> 56x56x128
            DepthwiseSeparableConv(128, 128, 1),# Conv 4: 56x56x128 -> 56x56x128
            DepthwiseSeparableConv(128, 256, 2),# Conv 5: 56x56x128 -> 28x28x256
            DepthwiseSeparableConv(256, 256, 1),# Conv 6: 28x28x256 -> 28x28x256
            DepthwiseSeparableConv(256, 512, 2),# Conv 7: 28x28x256 -> 14x14x512
            # 5x DepthwiseSeparableConv with (512, 512) and stride 1 (Conv 8-12)
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],
            DepthwiseSeparableConv(512, 1024, 2),# Conv 13: 14x14x512 -> 7x7x1024
            DepthwiseSeparableConv(1024, 1024, 1),# Conv 14: 7x7x1024 -> 7x7x1024
            nn.AvgPool2d(7)                      # Global average pooling
        )
        self.fc = nn.Linear(1024, num_classes)   # Fully connected layer for classification

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)  # Flatten
        x = self.fc(x)        # Fully connected layer
        return x

# Example usage
if __name__ == "__main__":
    model = MobileNetV1(num_classes=1000)
    print(model)
    # Example forward pass with a dummy input (batch size 1, 3 channels, 224x224)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Expected output: torch.Size([1, 1000])


total_para = sum(p.numel() for p in model.parameters() if p.requires_grad )
print(total_para)
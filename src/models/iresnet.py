"""
InsightFace 风格的 IResNet (Improved ResNet) backbone。

参考：nizhib/pytorch-insightface 中的 iresnet 实现，
用于和 FaceNet NN2 做单变量对照实验。

与标准 torchvision ResNet 的主要区别：
- 使用 pre-activation 风格的 IBasicBlock（BN -> Conv -> BN -> PReLU -> Conv -> BN）。
- 第一层为 3x3 stride=1，没有 7x7 大卷积和 maxpool。
- 最终输出先 AdaptiveAvgPool2d(1) 得到 512-D 特征，再投影到 embedding_dim 并 L2 归一化。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


__all__ = ["iresnet50", "iresnet100", "IResNet100"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class IBasicBlock(nn.Module):
    """InsightFace 改进的 BasicBlock（pre-activation + PReLU）。"""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IResNet(nn.Module):
    """InsightFace IResNet，输出 L2 归一化的 embedding。"""

    def __init__(
        self,
        layers: list[int],
        embedding_dim: int = 128,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.inplanes = 64
        self.dropout_p = dropout
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-05, momentum=0.9)
        self.prelu1 = nn.PReLU(64)

        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.features = nn.BatchNorm1d(embedding_dim, eps=2e-05, momentum=0.9)

        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * IBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * IBasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=2e-05, momentum=0.9),
            )

        layers = [
            IBasicBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        ]
        self.inplanes = planes * IBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                if self.use_checkpoint and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """推理接口，等价于 forward。"""
        return self.forward(x)


def iresnet50(
    embedding_dim: int = 128, dropout: float = 0.0, use_checkpoint: bool = False
) -> IResNet:
    """IResNet50: [3, 4, 14, 3]。"""
    return IResNet(
        [3, 4, 14, 3],
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    )


def iresnet100(
    embedding_dim: int = 128, dropout: float = 0.0, use_checkpoint: bool = False
) -> IResNet:
    """IResNet100: [3, 13, 30, 3]。"""
    return IResNet(
        [3, 13, 30, 3],
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    )


# 兼容现有 train.py 的 MODEL_REGISTRY 命名
IResNet100 = iresnet100

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Normalize(nn.Module):
    """L2 归一化到单位球面，输出 128-D embedding。"""

    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class BasicConv2d(nn.Module):
    """Conv + BN + ReLU 基础模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    """简化的 Inception 模块（类似 GoogLeNet / FaceNet NN2-NN4）。"""

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int, pool_proj: int):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class FaceNet(nn.Module):
    """
    FaceNet 风格的 Inception backbone + 128-D embedding + L2 归一化。

    支持通过 `num_inception_blocks` 和 `feature_dim` 灵活构造 NN2/NN3/NN4 级别的模型。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_inception_blocks: int = 3,
        dropout: float = 0.6,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem：快速降采样
        self.stem = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        )

        # Inception 堆叠
        channels = [192, 256, 288, 288]
        self.inception_blocks = nn.ModuleList()
        in_ch = 192
        for i in range(num_inception_blocks):
            out_ch = channels[min(i + 1, len(channels) - 1)]
            # 保持通道数近似平衡
            ch1x1 = out_ch // 4
            ch3x3red = out_ch // 8
            ch3x3 = out_ch // 4
            ch5x5red = out_ch // 32
            ch5x5 = out_ch // 16
            pool_proj = out_ch // 4
            self.inception_blocks.append(
                InceptionBlock(in_ch, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
            )
            in_ch = ch1x1 + ch3x3 + ch5x5 + pool_proj
            if i < num_inception_blocks - 1:
                self.inception_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_ch, embedding_dim, bias=False)
        self.normalize = L2Normalize(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.inception_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.normalize(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """推理接口，等价于 forward。"""
        return self.forward(x)


def NN2(embedding_dim: int = 128, dropout: float = 0.6) -> FaceNet:
    """Inception 风格大模型，输入 224×224。"""
    return FaceNet(embedding_dim=embedding_dim, num_inception_blocks=4, dropout=dropout)


def NN3(embedding_dim: int = 128, dropout: float = 0.6) -> FaceNet:
    """输入 160×160 的中等模型。"""
    return FaceNet(embedding_dim=embedding_dim, num_inception_blocks=3, dropout=dropout)


def NN4(embedding_dim: int = 128, dropout: float = 0.6) -> FaceNet:
    """输入 96×96 的轻量模型，适合快速 baseline。"""
    return FaceNet(embedding_dim=embedding_dim, num_inception_blocks=2, dropout=dropout)


def NNS1(embedding_dim: int = 128, dropout: float = 0.4) -> FaceNet:
    """更小的模型，输入约 165×165。"""
    return FaceNet(embedding_dim=embedding_dim, num_inception_blocks=2, dropout=dropout)


def NNS2(embedding_dim: int = 128, dropout: float = 0.4) -> FaceNet:
    """最小模型，输入约 140×116。"""
    return FaceNet(embedding_dim=embedding_dim, num_inception_blocks=1, dropout=dropout)

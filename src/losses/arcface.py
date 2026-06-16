import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss).

    仅替换 Triplet Loss 时使用，保持模型其余部分不变：
    模型仍然输出 embedding，本模块在 embedding 后添加一个可学习的
    分类权重矩阵 W（shape: num_classes x embedding_dim），并施加
    additive angular margin。

    参考实现：
      - Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
      - https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # 预计算 margin 相关常量
        self.register_buffer("cos_m", torch.tensor(math.cos(margin)))
        self.register_buffer("sin_m", torch.tensor(math.sin(margin)))
        self.register_buffer("th", torch.tensor(math.cos(math.pi - margin)))
        self.register_buffer("mm", torch.tensor(math.sin(margin) * margin))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: (B, D) 模型输出的 embedding（是否归一化均可）。
            labels:     (B,)   身份标签。

        Returns:
            loss: 标量 ArcFace loss。
            stats: 包含训练统计信息的字典。
        """
        # L2 归一化 embedding 与分类权重
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # cos(theta) = x^T w
        cos_t = F.linear(embeddings, weight)  # (B, num_classes)

        # sin(theta) for cos(theta + m)
        sin_t = torch.sqrt(1.0 - torch.clamp(cos_t * cos_t, min=0.0, max=1.0) + 1e-6)

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cos_t * self.cos_m - sin_t * self.sin_m

        # 标准 ArcFace 的 margin 截断（非 easy-margin）
        phi = torch.where(cos_t > self.th, phi, cos_t - self.mm)

        # 仅对目标类施加 margin
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=cos_t.dtype)
        logits = self.scale * (one_hot * phi + (1.0 - one_hot) * cos_t)

        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total = labels.size(0)
            acc = correct / total

        stats = {
            "num_triplets": total,  # 占位，保持与 TripletLoss 统计字典兼容
            "num_valid": correct,
            "frac_valid": acc,
            "d_ap_mean": 0.0,
            "d_an_mean": 0.0,
            "accuracy": acc,
        }
        return loss, stats

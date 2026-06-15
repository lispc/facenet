import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    FaceNet 论文中的 Triplet Loss。

    L = Σ max(d(a, p)^2 - d(a, n)^2 + alpha, 0)
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        triplets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: (N, D) 已 L2 归一化的 embeddings。
            labels: (N,) 身份标签。
            triplets: (M, 3) 预选的 (anchor, positive, negative) 索引。若为 None，
                      调用方需在外部完成 mining 后传入。

        Returns:
            loss: 标量 loss
            stats: 包含有效 triplet 数量、hard/semi-hard 数量等统计信息
        """
        if triplets is None:
            raise ValueError("TripletLoss 需要外部 mining 模块提供 triplets")

        anchors = embeddings[triplets[:, 0]]
        positives = embeddings[triplets[:, 1]]
        negatives = embeddings[triplets[:, 2]]

        # 因为 embeddings 已 L2 归一化，这里用 1 - cosine = squared L2 / 2
        # 论文使用 squared Euclidean distance，这里保持一致
        d_ap = torch.sum((anchors - positives) ** 2, dim=1)
        d_an = torch.sum((anchors - negatives) ** 2, dim=1)

        losses = F.relu(d_ap - d_an + self.margin)

        valid_mask = losses > 1e-6
        num_valid = valid_mask.sum().item()
        num_triplets = triplets.size(0)

        if num_valid == 0:
            # 没有有效 triplet 时返回 0 loss，避免 NaN
            loss = losses.sum() * 0.0
        else:
            loss = losses.mean()

        stats = {
            "num_triplets": num_triplets,
            "num_valid": num_valid,
            "frac_valid": num_valid / max(num_triplets, 1),
            "d_ap_mean": d_ap.mean().item(),
            "d_an_mean": d_an.mean().item(),
        }
        return loss, stats

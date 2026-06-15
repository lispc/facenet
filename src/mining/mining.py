import torch


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    计算已 L2 归一化 embeddings 之间的 squared Euclidean distance。

    Args:
        embeddings: (N, D)

    Returns:
        distances: (N, N)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=1e-12)
    return distances


def all_triplets_mask(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造所有合法的 anchor-positive-negative 索引 mask。

    Returns:
        anchor_idx: (N, N, N)
        positive_idx: (N, N, N)
        negative_idx: (N, N, N)
        valid_mask: (N, N, N) bool，True 表示 (i, j, k) 是合法 triplet
    """
    n = labels.size(0)
    indices = torch.arange(n, device=labels.device)
    anchor_idx = indices.view(n, 1, 1).expand(n, n, n)
    positive_idx = indices.view(1, n, 1).expand(n, n, n)
    negative_idx = indices.view(1, 1, n).expand(n, n, n)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    labels_negative = ~labels_equal

    i_equal_j = anchor_idx == positive_idx
    i_equal_k = anchor_idx == negative_idx
    j_equal_k = positive_idx == negative_idx

    # valid: anchor != positive, anchor != negative, positive != negative
    valid_indices = (~i_equal_j) & (~i_equal_k) & (~j_equal_k)
    # positive: labels[i] == labels[j]
    valid_positives = labels_equal.unsqueeze(2).expand(n, n, n)
    # negative: labels[i] != labels[k]
    valid_negatives = labels_negative.unsqueeze(2).expand(n, n, n)

    valid_mask = valid_indices & valid_positives & valid_negatives
    return anchor_idx, positive_idx, negative_idx, valid_mask


def _build_pos_idx_matrix(idxs: torch.Tensor) -> torch.Tensor:
    """
    给定同一身份的 K 个索引，构造 (K, K-1) 的 positive 索引矩阵：
    第 i 行表示 anchor idxs[i] 对应的 K-1 个 positive 索引。
    """
    k = idxs.size(0)
    rows = []
    for i in range(k):
        rows.append(torch.cat([idxs[:i], idxs[i + 1 :]]))
    return torch.stack(rows, dim=0)


def semi_hard_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    max_triplets: int | None = None,
) -> torch.Tensor:
    """
    在线 semi-hard negative mining（O(N^2 * K) 内存实现）。

    按身份分组处理，避免 O(N^3) 的全量 (N,N,N) 矩阵，因此可以支持更大的 batch。
    对每个 anchor-positive 对，选择满足 d(a,p) < d(a,n) < d(a,p)+margin 的 negative；
    若多个则随机选一个。

    Args:
        embeddings: (N, D) 已 L2 归一化
        labels: (N,)
        margin: triplet margin
        max_triplets: 返回的最大 triplet 数量，None 表示不限制

    Returns:
        triplets: (M, 3) 索引 [anchor, positive, negative]
    """
    n = labels.size(0)
    device = labels.device

    # 按身份分组（PKBatchSampler 生成的 batch 中同一身份连续出现，但这里不依赖连续性）
    labels_list = labels.tolist()
    identity_to_indices: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels_list):
        identity_to_indices.setdefault(lab, []).append(idx)

    distances = pairwise_distances(embeddings)  # (N, N)
    all_triplets: list[torch.Tensor] = []

    full_index = torch.arange(n, device=device)

    for idxs in identity_to_indices.values():
        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        k_c = idxs_t.size(0)
        if k_c < 2:
            continue

        # 该身份下每个 anchor 对应的 positive 索引矩阵 (K, K-1)
        pos_idx = _build_pos_idx_matrix(idxs_t)  # (K, K-1)

        # 负样本索引：除本身份外的所有样本
        neg_mask = torch.ones(n, dtype=torch.bool, device=device)
        neg_mask[idxs_t] = False
        neg_idx = full_index[neg_mask]  # (M,)
        if neg_idx.size(0) == 0:
            continue

        # d(a, p) 与 d(a, n)
        d_ap = distances[idxs_t.unsqueeze(1), pos_idx]  # (K, K-1)
        d_an = distances[idxs_t.unsqueeze(1), neg_idx.unsqueeze(0)]  # (K, M)

        # semi-hard 条件矩阵 (K, K-1, M)
        cond = (d_ap.unsqueeze(2) < d_an.unsqueeze(1)) & (
            d_an.unsqueeze(1) < d_ap.unsqueeze(2) + margin
        )

        # 每个 anchor-positive 对随机选一个满足条件的 negative
        rand = torch.rand(cond.shape, device=device)
        rand[~cond] = -1.0
        sel_neg_local = rand.argmax(dim=2)  # (K, K-1)
        has = cond.any(dim=2)  # (K, K-1)

        if not has.any():
            continue

        anchor_flat = idxs_t.unsqueeze(1).expand(-1, k_c - 1)[has]
        positive_flat = pos_idx[has]
        negative_flat = neg_idx[sel_neg_local[has]]

        all_triplets.append(
            torch.stack([anchor_flat, positive_flat, negative_flat], dim=1)
        )

    if all_triplets:
        triplets = torch.cat(all_triplets, dim=0)
    else:
        triplets = torch.empty((0, 3), dtype=torch.long, device=device)

    if max_triplets is not None and triplets.size(0) > max_triplets:
        perm = torch.randperm(triplets.size(0), device=device)[:max_triplets]
        triplets = triplets[perm]
    return triplets


def semi_hard_mining_legacy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    max_triplets: int | None = None,
) -> torch.Tensor:
    """
    旧的 O(N^3) 全向量化 semi-hard mining，仅在小 batch（N <= ~256）时使用。
    """
    n = labels.size(0)
    device = labels.device

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    valid_pos = labels_equal & (~eye)
    valid_neg = ~labels_equal

    distances = pairwise_distances(embeddings)
    d_ap = distances.unsqueeze(2)
    d_an = distances.unsqueeze(1)
    semi_mask = (
        valid_pos.unsqueeze(2)
        & valid_neg.unsqueeze(1)
        & (d_ap < d_an)
        & (d_an < d_ap + margin)
    )

    rand = torch.rand(n, n, n, device=device)
    rand[~semi_mask] = -1.0
    neg_idx = rand.argmax(dim=2)

    has_semi = semi_mask.any(dim=2)
    valid_pairs = has_semi & valid_pos
    anchor_idx, positive_idx = torch.nonzero(valid_pairs, as_tuple=True)
    negative_idx = neg_idx[anchor_idx, positive_idx]

    triplets = torch.stack([anchor_idx, positive_idx, negative_idx], dim=1)

    if max_triplets is not None and triplets.size(0) > max_triplets:
        perm = torch.randperm(triplets.size(0), device=device)[:max_triplets]
        triplets = triplets[perm]
    return triplets


def hard_mining(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_triplets: int | None = None,
) -> torch.Tensor:
    """
    在线 hard negative mining。

    对每个 anchor，选 hardest positive（同身份最远）和 hardest negative（不同身份最近）。

    Args:
        embeddings: (N, D)
        labels: (N,)
        max_triplets: 返回的最大 triplet 数量

    Returns:
        triplets: (M, 3)
    """
    distances = pairwise_distances(embeddings)
    n = labels.size(0)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    # 排除自身
    eye = torch.eye(n, device=labels.device, dtype=torch.bool)
    labels_equal = labels_equal & (~eye)

    #  hardest positive: 同身份中距离最大的
    positive_distances = distances.clone()
    positive_distances[~labels_equal] = -1.0
    hardest_positive_idx = positive_distances.argmax(dim=1)  # (N,)

    #  hardest negative: 不同身份中距离最小的
    negative_mask = ~labels_equal & (~eye)
    negative_distances = distances.clone()
    negative_distances[~negative_mask] = float("inf")
    hardest_negative_idx = negative_distances.argmin(dim=1)  # (N,)

    anchor_idx = torch.arange(n, device=labels.device)
    triplets = torch.stack([anchor_idx, hardest_positive_idx, hardest_negative_idx], dim=1)

    # 过滤掉 hardest negative 不存在的情况（比如 batch 中只有一个身份）
    has_negative = negative_mask.any(dim=1)
    triplets = triplets[has_negative]

    if max_triplets is not None and triplets.size(0) > max_triplets:
        perm = torch.randperm(triplets.size(0), device=triplets.device)[:max_triplets]
        triplets = triplets[perm]

    return triplets

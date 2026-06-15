"""
使用 InsightFace 风格的 `.bin` 验证文件评估模型。

`.bin` 文件格式：pickle 保存的 (list_of_image_bytes, issame_list)。
图片按对排列：0-1 是一对，2-3 是一对，以此类推。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models.facenet import NN2, NN3, NN4, NNS1, NNS2


MODEL_REGISTRY = {
    "nn2": NN2,
    "nn3": NN3,
    "nn4": NN4,
    "nns1": NNS1,
    "nns2": NNS2,
}


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def evaluate(model, bin_path: str, transform, device, nrof_folds: int = 10, eval_batch_size: int = 64):
    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    # issame_list might be list of bool or int
    issame = np.array(issame_list, dtype=bool)
    nrof_pairs = len(issame)

    from PIL import Image
    import io

    images = []
    for img_bytes in bins:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(transform(img))
    images = torch.stack(images, dim=0)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), eval_batch_size):
            batch = images[i : i + eval_batch_size].to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # (2N, D)
    assert len(embeddings) == 2 * nrof_pairs

    # 计算每对的相似度
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    scores = np.sum(emb1 * emb2, axis=1)  # cosine similarity（已 L2 归一化）

    # 10-fold 交叉验证
    fold_size = nrof_pairs // nrof_folds
    indices = np.arange(nrof_pairs)
    np.random.seed(42)
    np.random.shuffle(indices)

    accs = []
    for fold in range(nrof_folds):
        test_mask = np.zeros(nrof_pairs, dtype=bool)
        test_mask[fold * fold_size : (fold + 1) * fold_size] = True
        train_mask = ~test_mask

        train_scores = scores[indices[train_mask]]
        train_labels = issame[indices[train_mask]]
        test_scores = scores[indices[test_mask]]
        test_labels = issame[indices[test_mask]]

        best_acc = 0.0
        best_thresh = 0.0
        for thresh in np.linspace(-1, 1, 200):
            acc = ((train_scores > thresh) == train_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        test_acc = ((test_scores > best_thresh) == test_labels).mean()
        accs.append(test_acc)
        print(f"Fold {fold + 1}: acc={test_acc * 100:.2f}%, thresh={best_thresh:.3f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\nMean accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet on InsightFace .bin files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bin_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="nn4", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input_size", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--nrof_folds", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(embedding_dim=args.embedding_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    transform = build_transform(args.input_size)
    evaluate(model, args.bin_path, transform, device, args.nrof_folds)


if __name__ == "__main__":
    main()

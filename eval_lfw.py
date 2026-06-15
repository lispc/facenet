import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data.lfw import LFWDataset, load_lfw_pairs
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


def evaluate(model, lfw_dataset, pairs_folds, device):
    model.eval()
    fold_accs = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for fold_idx, fold_pairs in enumerate(pairs_folds):
            scores = []
            labels = []
            for path1, path2, is_same in fold_pairs:
                img1 = lfw_dataset.get(path1).unsqueeze(0).to(device)
                img2 = lfw_dataset.get(path2).unsqueeze(0).to(device)
                emb1 = model(img1)
                emb2 = model(img2)
                sim = (emb1 * emb2).sum(dim=1).cpu().item()
                scores.append(sim)
                labels.append(is_same)

            scores = np.array(scores)
            labels = np.array(labels)

            best_acc = 0.0
            best_thresh = 0.0
            for thresh in np.linspace(-1, 1, 200):
                preds = scores > thresh
                acc = (preds == labels).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh

            fold_accs.append(best_acc)
            all_scores.extend(scores)
            all_labels.extend(labels)
            print(f"Fold {fold_idx + 1}: acc={best_acc * 100:.2f}%, thresh={best_thresh:.3f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\nLFW mean accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet on LFW")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lfw_root", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--model", type=str, default="nn4", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input_size", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(embedding_dim=args.embedding_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    transform = build_transform(args.input_size)
    lfw_dataset = LFWDataset(args.lfw_root, transform=transform)
    pairs_folds = load_lfw_pairs(args.pairs)

    evaluate(model, lfw_dataset, pairs_folds, device)


if __name__ == "__main__":
    main()

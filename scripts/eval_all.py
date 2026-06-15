"""一键评估 LFW / CFP-FP / AgeDB-30 三个标准 benchmark。"""
import argparse
from pathlib import Path

import sys

# Allow importing eval_bin.py from project root when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_bin import build_transform, evaluate, MODEL_REGISTRY

import torch


DATASETS = {
    "lfw": "lfw.bin",
    "cfp_fp": "cfp_fp.bin",
    "agedb_30": "agedb_30.bin",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet on LFW/CFP-FP/AgeDB-30")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_root", type=str, default="./data/casia-webface/eval")
    parser.add_argument("--model", type=str, default="nn2", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--nrof_folds", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(embedding_dim=args.embedding_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    transform = build_transform(args.input_size)
    eval_root = Path(args.eval_root)

    print(f"Checkpoint: {args.checkpoint}\nModel: {args.model}  Input: {args.input_size}x{args.input_size}")
    for name, filename in DATASETS.items():
        bin_path = eval_root / filename
        if not bin_path.exists():
            print(f"{name}: {bin_path} not found, skip")
            continue
        print(f"\n--- {name.upper()} ---")
        mean_acc, std_acc = evaluate(model, str(bin_path), transform, device, args.nrof_folds)
        print(f"{name}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

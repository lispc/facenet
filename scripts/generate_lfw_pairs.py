"""
从合成人脸数据生成类似 LFW pairs.txt 的测试对，用于验证 eval_lfw.py。
"""

import argparse
import random
from pathlib import Path


def generate_pairs(data_root: str, output: str, num_folds: int = 10, pairs_per_fold: int = 300):
    root = Path(data_root)
    identities = sorted([d.name for d in root.iterdir() if d.is_dir()])
    imgs_by_id = {}
    for identity in identities:
        imgs = sorted([p.name for p in (root / identity).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if len(imgs) >= 2:
            imgs_by_id[identity] = imgs

    valid_identities = list(imgs_by_id.keys())
    if len(valid_identities) < 2:
        raise ValueError("Need at least 2 identities with 2+ images")

    lines = [str(num_folds)]
    rng = random.Random(42)

    for _ in range(num_folds):
        # matched pairs
        for _ in range(pairs_per_fold // 2):
            identity = rng.choice(valid_identities)
            img1, img2 = rng.sample(imgs_by_id[identity], 2)
            # extract index like identity_NNNN.jpg -> NNNN
            idx1 = int(Path(img1).stem.split("_")[-1])
            idx2 = int(Path(img2).stem.split("_")[-1])
            lines.append(f"{identity}\t{idx1}\t{idx2}")

        # mismatched pairs
        for _ in range(pairs_per_fold // 2):
            id1, id2 = rng.sample(valid_identities, 2)
            img1 = rng.choice(imgs_by_id[id1])
            img2 = rng.choice(imgs_by_id[id2])
            idx1 = int(Path(img1).stem.split("_")[-1])
            idx2 = int(Path(img2).stem.split("_")[-1])
            lines.append(f"{id1}\t{idx1}\t{id2}\t{idx2}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {num_folds * pairs_per_fold} pairs -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/synthetic_faces")
    parser.add_argument("--output", type=str, default="./data/synthetic_faces/pairs.txt")
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--pairs_per_fold", type=int, default=600)
    args = parser.parse_args()
    generate_pairs(args.data_root, args.output, args.num_folds, args.pairs_per_fold)


if __name__ == "__main__":
    main()

"""
生成合成人脸数据集，用于代码冒烟测试。

不依赖真实数据集即可验证：
  - 模型前向/反向
  - Triplet Loss + online mining
  - DataLoader + PKBatchSampler
  - 4 卡 DDP 训练流程
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_identity_images(
    identity_dir: Path,
    num_images: int,
    image_size: int = 112,
    base_color: tuple | None = None,
):
    """为一个身份生成若干张相似但略有噪声的图片。"""
    identity_dir.mkdir(parents=True, exist_ok=True)
    if base_color is None:
        base_color = tuple(np.random.randint(50, 200, size=3).tolist())

    for i in range(num_images):
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * np.array(base_color, dtype=np.uint8)
        # 加入随机噪声和区块，模拟不同人脸
        noise = np.random.randint(-30, 30, size=(image_size, image_size, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 随机画两个“眼睛”
        x1, y1 = np.random.randint(20, image_size // 2 - 10, size=2)
        x2, y2 = np.random.randint(image_size // 2 + 10, image_size - 20, size=2)
        r = np.random.randint(3, 7)
        img[max(0, y1 - r) : y1 + r, max(0, x1 - r) : x1 + r] = 0
        img[max(0, y2 - r) : y2 + r, max(0, x2 - r) : x2 + r] = 0

        Image.fromarray(img).save(identity_dir / f"{identity_dir.name}_{i:04d}.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/synthetic_faces")
    parser.add_argument("--num_identities", type=int, default=200)
    parser.add_argument("--images_per_identity", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for identity_id in tqdm(range(args.num_identities), desc="Generating identities"):
        identity_dir = output_dir / f"id_{identity_id:05d}"
        base_color = tuple(np.random.randint(50, 200, size=3).tolist())
        generate_identity_images(
            identity_dir,
            num_images=args.images_per_identity,
            image_size=args.image_size,
            base_color=base_color,
        )

    print(f"Generated {args.num_identities} identities in {output_dir}")


if __name__ == "__main__":
    main()

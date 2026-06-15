"""
将 ImageFolder 格式的人脸数据集转换为 LMDB。

输入结构：
  root/
    identity_0/
      img1.jpg
      img2.jpg
    identity_1/
      ...

输出结构：
  output.lmdb/
    data.mdb
    lock.mdb
"""

import argparse
import os
import pickle
from pathlib import Path

import lmdb
from PIL import Image
from tqdm import tqdm


def image_to_bytes(img_path: Path, quality: int = 95) -> bytes:
    """统一转成 JPEG bytes，避免不同格式混用。"""
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def folder_to_lmdb(
    root: str,
    output_path: str,
    map_size: int | None = None,
    quality: int = 95,
):
    import io

    root = Path(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有图片路径和标签
    samples = []
    identity_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for label, identity_dir in enumerate(identity_dirs):
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((str(img_path), label))

    if len(samples) == 0:
        raise ValueError(f"No images found under {root}")

    # 估算 map_size：原始图片总大小 * 1.5 + 元数据（pickle 有 overhead）
    if map_size is None:
        total_size = sum(os.path.getsize(p) for p, _ in samples)
        map_size = int(total_size * 1.6) + 1 * 1024 * 1024 * 1024  # +1GB metadata

    env = lmdb.open(str(output_path), map_size=map_size)
    labels = []

    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(tqdm(samples, desc="Writing LMDB")):
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            # 如果不是 jpg，统一转 jpg
            if Path(img_path).suffix.lower() not in (".jpg", ".jpeg"):
                img_bytes = image_to_bytes(Path(img_path), quality=quality)

            value = pickle.dumps((label, img_bytes))
            key = f"{idx:08d}".encode("ascii")
            txn.put(key, value)
            labels.append(label)

        txn.put(b"__len__", str(len(samples)).encode("ascii"))
        txn.put(b"__labels__", pickle.dumps(labels))

    env.close()
    print(f"Wrote {len(samples)} samples to {output_path}")
    print(f"Map size: {map_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="ImageFolder root")
    parser.add_argument("--output", type=str, required=True, help="LMDB output path")
    parser.add_argument("--map_size", type=int, default=None, help="LMDB map size in bytes")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    folder_to_lmdb(args.root, args.output, map_size=args.map_size, quality=args.quality)


if __name__ == "__main__":
    main()

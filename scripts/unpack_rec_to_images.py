"""
将 MXNet ImageRecordIO `.rec` / `.idx` 格式人脸数据集解包成按身份组织的图片文件夹。

这是业界常见做法（InsightFace 的 mx_recordio_2_images.py 也是同样思路）：
  - 一次转换，之后训练直接读 ImageFolder，速度快；
  - 避免训练时依赖 mxnet；
  - 磁盘空间充足时（如本机 425 GB）完全可行。

输出结构：
  output_dir/
    images/
      0000000/
        0000000_0000.jpg
        ...
      0000001/
        ...
    label.txt   # images/0000000/0000000_0000.jpg 0
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.data.recordio import MXRecordIO, unpack_record


def unpack_rec(
    rec_root: str,
    output_dir: str,
    quality: int = 95,
):
    rec_root = Path(rec_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    imgrec = MXRecordIO(rec_root / "train.idx", rec_root / "train.rec")
    imgidx = list(imgrec.keys)

    label_lines = []
    identity_counts: dict[int, int] = {}

    for idx in tqdm(imgidx, desc="Unpacking records"):
        s = imgrec.read_idx(idx)
        header, img_bytes = unpack_record(s)
        lbl = header.label
        if hasattr(lbl, "__len__") and not isinstance(lbl, (str, bytes)):
            label = int(lbl.flat[0])
        else:
            label = int(lbl)

        identity_dir = images_dir / f"{label:07d}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        img_count = identity_counts.get(label, 0)
        img_name = f"{label:07d}_{img_count:04d}.jpg"
        img_path = identity_dir / img_name
        identity_counts[label] = img_count + 1

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        rel_path = f"images/{label:07d}/{img_name}"
        label_lines.append(f"{rel_path} {label}")

    with open(output_dir / "label.txt", "w") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"Unpacked {len(imgidx)} images to {images_dir}")
    print(f"Identities: {len(identity_counts)}")
    print(f"Label file: {output_dir / 'label.txt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_root", type=str, required=True, help="Directory containing train.rec and train.idx")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    unpack_rec(args.rec_root, args.output_dir, quality=args.quality)


if __name__ == "__main__":
    main()

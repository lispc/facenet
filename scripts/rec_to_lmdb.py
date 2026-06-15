"""
将 MXNet ImageRecordIO `.rec` / `.idx` 格式人脸数据集直接转换为 LMDB。

无需先解压成图片文件夹，节省中间磁盘占用。
转换过程中用 cv2 解码并重新编码成 JPEG，确保所有样本都是有效图像。
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# 让脚本从项目根目录找到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lmdb
from tqdm import tqdm

from src.data.recordio import MXRecordIO, unpack_record


def rec_to_lmdb(
    rec_root: str,
    output_path: str,
    map_size: int | None = None,
    quality: int = 95,
):
    rec_root = Path(rec_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imgrec = MXRecordIO(rec_root / "train.idx", rec_root / "train.rec")
    imgidx = list(imgrec.keys)

    # 估算 map_size：rec 大小 * 1.6 + 元数据（pickle 有一定 overhead）
    if map_size is None:
        rec_size = (rec_root / "train.rec").stat().st_size
        map_size = int(rec_size * 1.6) + 1 * 1024 * 1024 * 1024

    env = lmdb.open(str(output_path), map_size=map_size)
    labels = []
    skipped = 0

    with env.begin(write=True) as txn:
        for idx in tqdm(imgidx, desc="Converting .rec to LMDB"):
            s = imgrec.read_idx(idx)
            header, img_bytes = unpack_record(s)
            lbl = header.label
            if hasattr(lbl, "__len__") and not isinstance(lbl, (str, bytes)):
                label = int(lbl.flat[0])
            else:
                label = int(lbl)

            # 用 cv2 解码并重新编码，确保 JPEG 有效
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img_bytes = encoded.tobytes()

            value = pickle.dumps((label, img_bytes))
            key = f"{len(labels):08d}".encode("ascii")
            txn.put(key, value)
            labels.append(label)

        txn.put(b"__len__", str(len(labels)).encode("ascii"))
        txn.put(b"__labels__", pickle.dumps(labels))

    env.close()
    print(f"Wrote {len(labels)} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} corrupted samples")
    print(f"Map size: {map_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_root", type=str, required=True, help="Directory containing train.rec and train.idx")
    parser.add_argument("--output", type=str, required=True, help="LMDB output path")
    parser.add_argument("--map_size", type=int, default=None, help="LMDB map size in bytes")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    rec_to_lmdb(args.rec_root, args.output, map_size=args.map_size, quality=args.quality)


if __name__ == "__main__":
    main()

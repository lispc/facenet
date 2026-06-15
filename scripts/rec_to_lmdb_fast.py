"""
将 MXNet ImageRecordIO `.rec` / `.idx` 格式人脸数据集直接转换为 LMDB（快速版）。

不经过 cv2 解码/重编码，直接把 record 里的原始图像 bytes 写入 LMDB，
速度显著提升；LMDB 数据集使用 PIL 读取，可接受 JPEG/PNG 等原始编码。
"""

import argparse
import pickle
import sys
from pathlib import Path

# 让脚本从项目根目录找到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lmdb
from tqdm import tqdm

from src.data.recordio import MXRecordIO, unpack_record


def rec_to_lmdb_fast(
    rec_root: str,
    output_path: str,
    map_size: int | None = None,
):
    rec_root = Path(rec_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imgrec = MXRecordIO(rec_root / "train.idx", rec_root / "train.rec")
    imgidx = list(imgrec.keys)

    # 估算 map_size：rec 大小 * 1.2 + 元数据 overhead
    if map_size is None:
        rec_size = (rec_root / "train.rec").stat().st_size
        map_size = int(rec_size * 1.2) + 1 * 1024 * 1024 * 1024

    env = lmdb.open(str(output_path), map_size=map_size)
    labels = []
    skipped = 0

    with env.begin(write=True) as txn:
        for idx in tqdm(imgidx, desc="Converting .rec to LMDB (fast)"):
            s = imgrec.read_idx(idx)
            header, img_bytes = unpack_record(s)
            lbl = header.label
            if hasattr(lbl, "__len__") and not isinstance(lbl, (str, bytes)):
                label = int(lbl.flat[0])
            else:
                label = int(lbl)

            if not img_bytes:
                skipped += 1
                continue

            value = pickle.dumps((label, img_bytes))
            key = f"{len(labels):08d}".encode("ascii")
            txn.put(key, value)
            labels.append(label)

        txn.put(b"__len__", str(len(labels)).encode("ascii"))
        txn.put(b"__labels__", pickle.dumps(labels))

    env.close()
    print(f"Wrote {len(labels)} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} empty samples")
    print(f"Map size: {map_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_root", type=str, required=True, help="Directory containing train.rec and train.idx")
    parser.add_argument("--output", type=str, required=True, help="LMDB output path")
    parser.add_argument("--map_size", type=int, default=None, help="LMDB map size in bytes")
    args = parser.parse_args()
    rec_to_lmdb_fast(args.rec_root, args.output, map_size=args.map_size)


if __name__ == "__main__":
    main()

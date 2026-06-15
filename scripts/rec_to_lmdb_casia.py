"""
将 CASIA-WebFace 的 MXNet `.rec` / `.idx` + `train.lst` 转换为 LMDB。

该 Kaggle 版 .rec 的格式略有不同：
  - key 0 是 header 记录（非图像），需要跳过；
  - record header 的 label 字段实际存储的是 image payload 长度；
  - payload 前 24 字节是二次头（含 record index），真正 JPEG 从 SOI (\xff\xd8) 开始；
  - 身份标签需要从 train.lst 的第三列读取。

因此本脚本按 .lst 顺序读取，为每个 record key（key = line_index + 1）写入正确的身份标签。
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lmdb
from tqdm import tqdm

from src.data.recordio import MXRecordIO, unpack_record


def _extract_jpeg(img_bytes: bytes) -> bytes | None:
    """从含 24 字节二次头的 payload 中提取干净 JPEG bytes。"""
    soi = img_bytes.find(b"\xff\xd8\xff")
    if soi < 0:
        return None
    # EOI 之后的 padding 也去掉
    eoi = img_bytes.rfind(b"\xff\xd9")
    if eoi < 0:
        return img_bytes[soi:]
    return img_bytes[soi : eoi + 2]


def rec_to_lmdb_casia(
    rec_root: str,
    output_path: str,
    map_size: int | None = None,
):
    rec_root = Path(rec_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 lst 身份标签
    lst_path = rec_root / "train.lst"
    if not lst_path.exists():
        raise FileNotFoundError(f"train.lst not found at {lst_path}")

    identities = []
    with open(lst_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            identities.append(int(parts[2]))

    print(f"Read {len(identities)} identities from {lst_path}")

    imgrec = MXRecordIO(rec_root / "train.idx", rec_root / "train.rec")

    if map_size is None:
        rec_size = (rec_root / "train.rec").stat().st_size
        map_size = int(rec_size * 1.2) + 1 * 1024 * 1024 * 1024

    env = lmdb.open(str(output_path), map_size=map_size)
    labels = []
    skipped = 0

    with env.begin(write=True) as txn:
        # key 0 是 header，跳过；图像 record 从 key 1 开始
        for line_idx in tqdm(range(len(identities)), desc="Converting CASIA .rec to LMDB"):
            record_key = line_idx + 1
            s = imgrec.read_idx(record_key)
            header, img_bytes = unpack_record(s)
            jpeg = _extract_jpeg(img_bytes)
            if jpeg is None:
                skipped += 1
                continue

            label = identities[line_idx]
            value = pickle.dumps((label, jpeg))
            key = f"{len(labels):08d}".encode("ascii")
            txn.put(key, value)
            labels.append(label)

        txn.put(b"__len__", str(len(labels)).encode("ascii"))
        txn.put(b"__labels__", pickle.dumps(labels))

    env.close()
    print(f"Wrote {len(labels)} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} records without valid JPEG")
    print(f"Map size: {map_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--map_size", type=int, default=None)
    args = parser.parse_args()
    rec_to_lmdb_casia(args.rec_root, args.output, map_size=args.map_size)


if __name__ == "__main__":
    main()

"""
数据集下载辅助脚本。

由于 MS1MV2 / CASIA-WebFace / LFW 等数据集分散在不同平台且常需要账号，
本脚本提供统一的命令入口和说明，而不是在代码里硬编码私有下载链接。
"""

import argparse
import os
import subprocess
from pathlib import Path


DATASET_INFO = {
    "lfw": {
        "name": "LFW (Labeled Faces in the Wild)",
        "size": "~180 MB",
        "urls": [
            "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
            "http://vis-www.cs.umass.edu/lfw/pairs.txt",
        ],
        "note": "官网提供 original / funneled / deep funneled 多个版本；pairs.txt 需要单独下载。",
    },
    "casia-webface": {
        "name": "CASIA-WebFace 112×112",
        "size": "~2.7 GB",
        "kaggle": "debarghamitraroy/casia-webface",
        "note": "Kaggle 版为 MXNet .rec/.idx 格式，已 112×112 对齐。",
    },
    "ms1mv2": {
        "name": "MS1MV2 112×112",
        "size": "~16.5 GB",
        "kaggle": "yakhyokhuja/ms1m-arcface-dataset",
        "note": "Kaggle 版包含 train.rec / train.idx；InsightFace DataZoo 也提供百度网盘/Google Drive 镜像。",
    },
    "vggface2": {
        "name": "VGGFace2",
        "size": "~36 GB compressed",
        "url": "http://zeus.robots.ox.ac.uk/vgg_face2/",
        "note": "需要官网注册；本机磁盘足够但训练/预处理成本高，优先级低于 MS1MV2。",
    },
    "youtube-faces": {
        "name": "YouTube Faces DB",
        "size": "~10–25 GB",
        "url": "http://www.cs.tau.ac.il/~wolf/ytfaces/",
        "note": "论文次评测集；有余力再下载。",
    },
}


def ensure_kaggle():
    """检查 kaggle.json 是否存在。"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "ERROR: 未找到 Kaggle API 凭证 ~/.kaggle/kaggle.json。\n"
            "请先在 Kaggle 账号页面创建 API Token 并放到 ~/.kaggle/kaggle.json，"
            "权限设为 chmod 600。"
        )
        return False
    return True


def download_kaggle(dataset: str, output_dir: Path):
    if not ensure_kaggle():
        return
    info = DATASET_INFO[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        info["kaggle"],
        "-p",
        str(output_dir),
        "--unzip",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Downloaded {info['name']} to {output_dir}")


def download_lfw(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    img_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    pairs_url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
    print(f"Downloading LFW images from {img_url} ...")
    subprocess.run(["curl", "-L", "-o", str(output_dir / "lfw.tgz"), img_url], check=True)
    print(f"Downloading LFW pairs from {pairs_url} ...")
    subprocess.run(["curl", "-L", "-o", str(output_dir / "pairs.txt"), pairs_url], check=True)
    print(f"Extracting {output_dir / 'lfw.tgz'} ...")
    subprocess.run(["tar", "-xzf", str(output_dir / "lfw.tgz"), "-C", str(output_dir)], check=True)
    print(f"LFW ready in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download FaceNet datasets")
    parser.add_argument("dataset", type=str, choices=list(DATASET_INFO.keys()))
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    info = DATASET_INFO[args.dataset]
    print(f"Dataset: {info['name']}")
    print(f"Size: {info['size']}")
    print(f"Note: {info['note']}")

    if args.dataset == "lfw":
        download_lfw(output_dir / "lfw")
    elif "kaggle" in info:
        download_kaggle(args.dataset, output_dir / args.dataset)
    else:
        print(f"请手动从 {info.get('url', info.get('urls'))} 下载并解压到 {output_dir / args.dataset}")


if __name__ == "__main__":
    main()

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import FaceDataset, MXFaceDataset
from src.data.lfw import LFWDataset, load_lfw_pairs
from src.data.lmdb_dataset import LMDBFaceDataset
from src.data.sampler import PKBatchSampler
from src.losses.triplet import TripletLoss
from src.mining.mining import hard_mining, semi_hard_mining
from src.models.facenet import NN2, NN3, NN4, NNS1, NNS2


MODEL_REGISTRY = {
    "nn2": NN2,
    "nn3": NN3,
    "nn4": NN4,
    "nns1": NNS1,
    "nns2": NNS2,
}


def setup_distributed(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(input_size: int, is_train: bool = True):
    ops = [transforms.Resize((input_size, input_size))]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(
            transforms.ColorJitter(
                brightness=0.125, contrast=0.125, saturation=0.125, hue=0.05
            )
        )
    ops.append(transforms.ToTensor())
    # ImageNet 统计量近似；人脸数据集可再调整
    ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(ops)


def build_dataset(args):
    train_transform = build_transform(args.input_size, is_train=True)
    if args.dataset_type == "imagefolder":
        dataset = FaceDataset(args.data_root, transform=train_transform)
    elif args.dataset_type == "mxrec":
        dataset = MXFaceDataset(args.data_root, transform=train_transform)
    elif args.dataset_type == "lmdb":
        dataset = LMDBFaceDataset(
            args.data_root, transform=train_transform, preload=args.preload_lmdb
        )
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    return dataset


def build_model(args):
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(embedding_dim=args.embedding_dim, dropout=args.dropout)
    return model


def _build_warmup_scheduler(optimizer, warmup_batches: int, base_lr: float):
    """Linear warmup from 0 to base_lr over warmup_batches."""
    if warmup_batches <= 0:
        return None

    def lr_lambda(step):
        # step starts at 0
        return min(1.0, (step + 1) / warmup_batches)

    return LambdaLR(optimizer, lr_lambda)


def evaluate_bin(model, bin_path: str, device, input_size: int, eval_batch_size: int = 64):
    """Evaluate on InsightFace .bin file (batched)."""
    import io
    import pickle
    from PIL import Image

    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    issame = np.array(issame_list, dtype=bool)
    nrof_pairs = len(issame)

    transform = build_transform(input_size, is_train=False)
    images = []
    for img_bytes in bins:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(transform(img))
    images = torch.stack(images, dim=0)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), eval_batch_size):
            batch = images[i : i + eval_batch_size].to(device)
            emb = F.normalize(model(batch), p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    assert len(embeddings) == 2 * nrof_pairs

    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    scores = np.sum(emb1 * emb2, axis=1)

    # 10-fold cross validation
    fold_size = nrof_pairs // 10
    indices = np.arange(nrof_pairs)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    accs = []
    for fold in range(10):
        test_mask = np.zeros(nrof_pairs, dtype=bool)
        test_mask[fold * fold_size : (fold + 1) * fold_size] = True
        train_mask = ~test_mask

        train_scores = scores[indices[train_mask]]
        train_labels = issame[indices[train_mask]]
        test_scores = scores[indices[test_mask]]
        test_labels = issame[indices[test_mask]]

        best_acc = 0.0
        best_thresh = 0.0
        for thresh in np.linspace(-1, 1, 200):
            acc = ((train_scores > thresh) == train_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        test_acc = ((test_scores > best_thresh) == test_labels).mean()
        accs.append(test_acc)

    return float(np.mean(accs)), float(np.std(accs))


def _build_scheduler(optimizer, args, total_steps: int):
    """Build main scheduler optionally wrapped with linear warmup."""
    if args.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - args.warmup_batches), eta_min=args.min_lr
        )
    elif args.scheduler == "step":
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=[int(total_steps * 0.5), int(total_steps * 0.8)],
            gamma=0.1,
        )
    else:
        main_scheduler = None

    warmup_scheduler = _build_warmup_scheduler(optimizer, args.warmup_batches, args.lr)
    if warmup_scheduler is None or main_scheduler is None:
        return warmup_scheduler or main_scheduler

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_batches],
    )


def evaluate_lfw(model, lfw_dataset, pairs_folds, device, eval_batch_size: int = 64):
    model.eval()
    all_correct = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for fold_pairs in pairs_folds:
            # Collect unique paths for batched inference
            path_to_idx = {}
            tensors = []
            for path1, path2, _ in fold_pairs:
                for p in (path1, path2):
                    if p not in path_to_idx:
                        path_to_idx[p] = len(tensors)
                        tensors.append(lfw_dataset.get(p))
            stacked = torch.stack(tensors, dim=0).to(device)

            embeddings = []
            for i in range(0, len(stacked), eval_batch_size):
                batch = stacked[i : i + eval_batch_size]
                emb = F.normalize(model(batch), p=2, dim=1)
                embeddings.append(emb)
            embeddings = torch.cat(embeddings, dim=0)

            scores = []
            labels = []
            for path1, path2, is_same in fold_pairs:
                emb1 = embeddings[path_to_idx[path1]]
                emb2 = embeddings[path_to_idx[path2]]
                sim = (emb1 * emb2).sum(dim=0).cpu().item()
                scores.append(sim)
                labels.append(is_same)

            scores = np.array(scores)
            labels = np.array(labels)
            best_acc = 0.0
            for thresh in np.linspace(-1, 1, 100):
                preds = scores > thresh
                acc = (preds == labels).mean()
                if acc > best_acc:
                    best_acc = acc
            all_correct.append(best_acc)
            all_scores.extend(scores)
            all_labels.extend(labels)

    mean_acc = np.mean(all_correct)
    std_acc = np.std(all_correct)
    return mean_acc, std_acc


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    epoch: int,
    args,
    writer: SummaryWriter | None,
    global_step: int,
):
    model.train()
    epoch_loss = 0.0
    epoch_valid = 0.0
    epoch_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if args.rank == 0 else dataloader
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=args.amp):
            embeddings = model(images)
            if args.mining == "semi-hard":
                triplets = semi_hard_mining(
                    embeddings, labels, margin=args.margin, max_triplets=args.max_triplets
                )
            elif args.mining == "hard":
                triplets = hard_mining(
                    embeddings, labels, max_triplets=args.max_triplets
                )
            else:
                raise ValueError(f"Unknown mining: {args.mining}")

            loss, stats = criterion(embeddings, labels, triplets)
            loss = loss / args.accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accum_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
                if writer is not None and args.rank == 0:
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        batch_loss = loss.item() * args.accum_steps
        epoch_loss += batch_loss
        epoch_valid += stats["num_valid"]
        epoch_total += stats["num_triplets"]
        global_step += 1

        if args.rank == 0:
            if writer is not None:
                writer.add_scalar("train/loss", batch_loss, global_step)
                writer.add_scalar("train/valid_triplet_frac", stats["frac_valid"], global_step)
                writer.add_scalar("train/d_ap", stats["d_ap_mean"], global_step)
                writer.add_scalar("train/d_an", stats["d_an_mean"], global_step)

            if isinstance(pbar, tqdm):
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    valid=f"{stats['frac_valid']:.2%}",
                    d_ap=f"{stats['d_ap_mean']:.3f}",
                    d_an=f"{stats['d_an_mean']:.3f}",
                )

    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_valid_frac = epoch_valid / max(epoch_total, 1)
    return avg_loss, avg_valid_frac, global_step


def main_worker(rank: int, world_size: int, args):
    args.rank = rank
    args.world_size = world_size

    if world_size > 1:
        setup_distributed(rank, world_size)

    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    # Dataset
    dataset = build_dataset(args)
    num_batches = args.num_batches_per_epoch
    sampler = PKBatchSampler(
        dataset.labels,
        p=args.p,
        k=args.k,
        num_batches=num_batches,
        shuffle=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = build_model(args).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Loss / Optimizer / Scheduler
    criterion = TripletLoss(margin=args.margin)
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    total_steps = max(1, args.epochs * num_batches // args.accum_steps)
    scheduler = _build_scheduler(optimizer, args, total_steps)
    scaler = GradScaler("cuda", enabled=args.amp)

    # Resume
    start_epoch = 1
    global_step = 0
    best_acc = 0.0
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(args.resume, map_location=map_location, weights_only=False)
        model_state = checkpoint["model_state_dict"]
        if world_size > 1 and not any(k.startswith("module.") for k in model_state.keys()):
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("accuracy", 0.0)
        global_step = checkpoint.get("global_step", (start_epoch - 1) * num_batches)

    # LFW eval setup
    lfw_dataset = None
    pairs_folds = None
    if args.lfw_root and args.lfw_pairs and rank == 0:
        lfw_transform = build_transform(args.input_size, is_train=False)
        lfw_dataset = LFWDataset(args.lfw_root, transform=lfw_transform)
        pairs_folds = load_lfw_pairs(args.lfw_pairs)

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss, avg_valid_frac, global_step = train_one_epoch(
            model,
            dataloader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            device,
            epoch,
            args,
            writer,
            global_step,
        )

        if rank == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                f"valid_triplet_frac={avg_valid_frac:.2%} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # LFW eval (image folder)
            if lfw_dataset is not None and pairs_folds is not None:
                mean_acc, std_acc = evaluate_lfw(
                    model, lfw_dataset, pairs_folds, device, eval_batch_size=args.eval_batch_size
                )
                print(f"LFW: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
                if writer is not None:
                    writer.add_scalar("lfw/mean_accuracy", mean_acc, epoch)
                    writer.add_scalar("lfw/std_accuracy", std_acc, epoch)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    ckpt_path = os.path.join(args.output_dir, "best.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "accuracy": mean_acc,
                            "global_step": global_step,
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint -> {ckpt_path}")

            # LFW eval (InsightFace .bin)
            if args.lfw_bin:
                mean_acc, std_acc = evaluate_bin(
                    model, args.lfw_bin, device, args.input_size, args.eval_batch_size
                )
                print(f"LFW(bin): {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
                if writer is not None:
                    writer.add_scalar("lfw_bin/mean_accuracy", mean_acc, epoch)
                    writer.add_scalar("lfw_bin/std_accuracy", std_acc, epoch)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    ckpt_path = os.path.join(args.output_dir, "best.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "accuracy": mean_acc,
                            "global_step": global_step,
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint -> {ckpt_path}")

            if args.cfp_fp_bin:
                mean_acc, std_acc = evaluate_bin(
                    model, args.cfp_fp_bin, device, args.input_size, args.eval_batch_size
                )
                print(f"CFP-FP: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
                if writer is not None:
                    writer.add_scalar("cfp_fp/mean_accuracy", mean_acc, epoch)
                    writer.add_scalar("cfp_fp/std_accuracy", std_acc, epoch)

            if args.agedb_30_bin:
                mean_acc, std_acc = evaluate_bin(
                    model, args.agedb_30_bin, device, args.input_size, args.eval_batch_size
                )
                print(f"AgeDB-30: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
                if writer is not None:
                    writer.add_scalar("agedb_30/mean_accuracy", mean_acc, epoch)
                    writer.add_scalar("agedb_30/std_accuracy", std_acc, epoch)

            if args.save_every > 0 and epoch % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": best_acc,
                        "global_step": global_step,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved periodic checkpoint -> {ckpt_path}")

    if rank == 0:
        final_path = os.path.join(args.output_dir, "final.pth")
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
                "global_step": global_step,
                "args": vars(args),
            },
            final_path,
        )
        print(f"Saved final checkpoint -> {final_path}")
        writer.close()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train FaceNet with Triplet Loss")

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="imagefolder",
        choices=["imagefolder", "mxrec", "lmdb"],
    )
    parser.add_argument("--preload_lmdb", action="store_true", help="Preload entire LMDB into RAM")
    parser.add_argument("--lfw_root", type=str, default=None)
    parser.add_argument("--lfw_pairs", type=str, default=None)
    parser.add_argument("--lfw_bin", type=str, default=None, help="InsightFace LFW .bin for per-epoch eval")
    parser.add_argument("--cfp_fp_bin", type=str, default=None)
    parser.add_argument("--agedb_30_bin", type=str, default=None)

    # Model
    parser.add_argument("--model", type=str, default="nn4", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input_size", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.6)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_batches_per_epoch", type=int, default=1000)
    parser.add_argument("--p", type=int, default=32, help="identities per batch")
    parser.add_argument("--k", type=int, default=8, help="images per identity")
    parser.add_argument("--batch_size", type=int, default=None, help="deprecated, use --p and --k")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    # Loss / Mining
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--mining", type=str, default="semi-hard", choices=["semi-hard", "hard"])
    parser.add_argument("--max_triplets", type=int, default=None)

    # Optimizer / Scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Cosine annealing minimum LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup_batches", type=int, default=0, help="Linear warmup steps at start")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)")

    # System
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/facenet")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0=disabled)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for LFW inference")

    args = parser.parse_args()
    if args.no_amp:
        args.amp = False

    # 支持 torchrun 启动，也支持直接 python train.py 多卡 spawn
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        main_worker(rank, world_size, args)
    else:
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
        else:
            main_worker(0, 1, args)


if __name__ == "__main__":
    main()

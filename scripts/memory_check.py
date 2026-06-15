"""Quick VRAM check for NN2 224x224 with a given per-GPU batch size."""
import argparse
import torch
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler

from src.models.facenet import NN2
from src.mining.mining import semi_hard_mining
from src.losses.triplet import TripletLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_gpu_batch", type=int, default=128)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model = NN2(embedding_dim=args.embedding_dim).to(device)
    criterion = TripletLoss(margin=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda", enabled=True)

    # Simulate one training step
    images = torch.randn(args.per_gpu_batch, 3, args.input_size, args.input_size, device=device)
    labels = torch.arange(args.per_gpu_batch // 8, device=device).repeat_interleave(8)

    with autocast("cuda", enabled=True):
        embeddings = model(images)
        triplets = semi_hard_mining(embeddings, labels, margin=0.2)
        loss, _ = criterion(embeddings, labels, triplets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"Peak memory on cuda:{args.device}: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()

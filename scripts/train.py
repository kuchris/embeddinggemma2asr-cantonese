"""第 3 步：用預先抽好嘅 feature train reader（CTC）。

每個 feature 資料夾最後一個 shard 留嚟做驗證（CER）。

用法：python scripts/train.py data/feats/wordshk data/feats/mixed --vocab data/vocab.pt --out runs/frozen
      [--head frozen|tuned|learned] [--epochs 30] [--max-frames 40000]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from g2asr.data import batches, shard_paths
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units

parser = argparse.ArgumentParser()
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--vocab", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--head", default="frozen", choices=["frozen", "tuned", "learned"])
parser.add_argument("--d-model", type=int, default=384)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--time-masks", type=int, default=2, help="每 5 秒遮幾段時間（SpecAugment 式，0 = 唔遮）")
parser.add_argument("--time-mask-width", type=int, default=6, help="每段最多遮幾格（每格 40ms）")
parser.add_argument("--feat-masks", type=int, default=2, help="每句遮幾段 feature 維度")
parser.add_argument("--feat-mask-width", type=int, default=48, help="每段最多遮幾多維（共 512 維）")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--max-frames", type=int, default=40000, help="每個 batch 最多幾多格（連 padding）")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup", type=int, default=500)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
device = "cuda"
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

vocab = torch.load(args.vocab)
unit_to_id = {u: i + 1 for i, u in enumerate(vocab["units"])}  # 0 留畀 blank

train_paths, val_paths = [], []
for d in args.feat_dirs:
    paths = shard_paths([d])
    if len(paths) < 2:
        raise SystemExit(f"{d} 最少要 2 個 shard（一個做驗證）")
    train_paths += paths[:-1]
    val_paths.append(paths[-1])

model = Reader(vocab["emb"], d_model=args.d_model, layers=args.layers, dropout=args.dropout, head=args.head).to(device)
print(f"reader 參數：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M，"
      f"字典 {len(vocab['units'])}，head={args.head}，train shard {len(train_paths)}，val shard {len(val_paths)}")

# 數晒所有 shard 嘅 batch 數，畀 cosine schedule 用（只用第一個 shard 推算會估少，lr 太早跌到 0）
steps_per_epoch = sum(1 for _ in batches(train_paths, unit_to_id, args.max_frames, shuffle=False))
total_steps = max(steps_per_epoch * args.epochs, args.warmup + 1)
print(f"每個 epoch {steps_per_epoch} 步，總共 {total_steps} 步")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (
    1 + math.cos(math.pi * min(1.0, s / total_steps))))
scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

step, epoch, best_cer = 0, 0, float("inf")
ckpt_path = out / "last.pt"
if args.resume and ckpt_path.exists():
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    step, epoch, best_cer = ck["step"], ck["epoch"], ck["best_cer"]
    print(f"由 step {step}（epoch {epoch}）續 train")


def augment(feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """喺 feature 上面隨機遮時間段同維度段（SpecAugment 式），對付 overfitting。encoder 凍結都用得。"""
    feats = feats.clone()
    for i, n in enumerate(lengths.tolist()):
        for _ in range(max(1, round(args.time_masks * n / 125))):  # 125 格 = 5 秒
            w = int(torch.randint(0, args.time_mask_width + 1, ()))
            if w and n > w:
                t0 = int(torch.randint(0, n - w, ()))
                feats[i, t0:t0 + w] = 0
        for _ in range(args.feat_masks):
            w = int(torch.randint(0, args.feat_mask_width + 1, ()))
            if w:
                f0 = int(torch.randint(0, feats.shape[-1] - w, ()))
                feats[i, :, f0:f0 + w] = 0
    return feats


def save(path: Path):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "epoch": epoch, "best_cer": best_cer, "config": model.config,
                "units": vocab["units"], "vocab_emb": vocab["emb"]}, path)


@torch.no_grad()
def evaluate(show: int = 5) -> float:
    model.eval()
    errors = total = 0
    examples = []
    for feats, lengths, targets, target_lengths, texts in batches(val_paths, unit_to_id, args.max_frames, False):
        with torch.autocast("cuda", dtype=amp_dtype):
            log_probs = model(feats.to(device), lengths.to(device))
        refs = targets.split(target_lengths.tolist())
        for hyp, ref in zip(Reader.greedy(log_probs, lengths), refs):
            ref = ref.tolist()
            ref = [r - 1 for r in ref]
            errors += edit_distance(hyp, ref)
            total += len(ref)
            if len(examples) < show:
                examples.append((join_units([vocab["units"][i] for i in ref]),
                                 join_units([vocab["units"][i] for i in hyp])))
    model.train()
    for ref, hyp in examples:
        print(f"    參考：{ref}\n    輸出：{hyp}")
    return errors / max(total, 1)


log = open(out / "log.jsonl", "a", encoding="utf-8")
model.train()
t0, loss_sum, loss_n = time.time(), 0.0, 0
while epoch < args.epochs:
    for feats, lengths, targets, target_lengths, _ in batches(train_paths, unit_to_id, args.max_frames, True):
        feats, lengths = feats.to(device), lengths.to(device)
        if args.time_masks or args.feat_masks:
            feats = augment(feats, lengths)
        with torch.autocast("cuda", dtype=amp_dtype):
            log_probs = model(feats, lengths)
        loss = F.ctc_loss(log_probs.float().transpose(0, 1), targets.to(device), lengths, target_lengths.to(device),
                          blank=0, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        loss_sum, loss_n = loss_sum + loss.item(), loss_n + 1

        if step % 100 == 0:
            print(f"step {step}/{total_steps}  epoch {epoch}  loss {loss_sum / loss_n:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  scale {model.log_scale.exp().item():.1f}  {time.time() - t0:.0f}s")
            loss_sum, loss_n = 0.0, 0
        if step % args.eval_every == 0:
            cer = evaluate()
            print(f"  ▶ step {step} 驗證 CER {cer:.2%}")
            log.write(json.dumps({"step": step, "epoch": epoch, "cer": cer}) + "\n")
            log.flush()
            if cer < best_cer:
                best_cer = cer
                save(out / "best.pt")
            save(ckpt_path)
    epoch += 1

cer = evaluate()
print(f"完成：最後 CER {cer:.2%}，最好 {min(best_cer, cer):.2%}")
log.write(json.dumps({"step": step, "epoch": epoch, "cer": cer}) + "\n")
if cer < best_cer:
    best_cer = cer
    save(out / "best.pt")
save(ckpt_path)

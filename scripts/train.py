"""第 3 步：用預先抽好嘅 feature train reader（CTC）。

每個 feature 資料夾最後一個 shard 留嚟做驗證（CER）。

用法：python scripts/train.py data/feats/wordshk data/feats/mixed --vocab data/vocab.pt --out runs/frozen
      [--head frozen|tuned|learned] [--epochs 30] [--max-frames 40000]
      [--use-layers 0,4,8] [--jy-weight 0.3 --jy-layer 3]

--use-layers：shard 有多層 feature 嗰陣揀用邊幾層（預設全部；見 Encoder.frames）。
--jy-weight：加粵拼輔助 CTC（loss = (1−w)×字 + w×粵拼），推理唔使用；要裝 ToJyutping。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # 寫去檔案都即時睇到 log

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from g2asr.data import batches, shard_paths
from g2asr.fastdata import Preloaded, augment_gpu, prefetch
from g2asr.reader import Reader
from g2asr.text import edit_distance, join_units

parser = argparse.ArgumentParser()
parser.add_argument("feat_dirs", nargs="+")
parser.add_argument("--vocab", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--head", default="frozen", choices=["frozen", "tuned", "learned"])
parser.add_argument("--d-model", type=int, default=384)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--heads", type=int, default=6, help="attention head 數，要整除 d-model")
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
parser.add_argument("--stream", action="store_true",
                    help="唔預先載入 RAM，逐個 shard 由硬碟讀（RAM 唔夠先用；慢好多）")
parser.add_argument("--max-steps", type=int, default=0, help="行咗幾多步就停（測試用，0 = 唔限）")
parser.add_argument("--use-layers", help="用 shard 入面邊幾層 feature，例如 0 或者 4,8,12,0（預設全部）")
parser.add_argument("--jy-weight", type=float, default=0.0, help="粵拼輔助 CTC 嘅比重（0 = 唔用）")
parser.add_argument("--jy-layer", type=int, default=3, help="粵拼 head 駁喺 reader 第幾層後面")
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

use_layers = [int(x) for x in args.use_layers.split(",")] if args.use_layers else None
meta = torch.load(train_paths[0], mmap=True)
feat_layers = use_layers or meta.get("layers", [0])
pca = torch.load(meta["pca"])["pca"] if meta.get("pca") else {}  # 抽 feature 時用嘅 PCA，要跟 checkpoint 一齊存
pca = {l: v for l, v in pca.items() if l in feat_layers}
del meta
jy_vocab = {} if args.jy_weight > 0 else None
if args.stream and (args.use_layers or jy_vocab is not None):
    raise SystemExit("--stream 唔支援 --use-layers／--jy-weight")

if args.stream:
    # 數晒所有 shard 嘅 batch 數，畀 cosine schedule 用（只用第一個 shard 推算會估少，lr 太早跌到 0）
    steps_per_epoch = sum(1 for _ in batches(train_paths, unit_to_id, args.max_frames, shuffle=False))
else:
    t_load = time.time()
    data = Preloaded(train_paths, unit_to_id, use_layers, jy_vocab)
    val_data = Preloaded(val_paths, unit_to_id, use_layers)
    steps_per_epoch = len(data.batches(args.max_frames, shuffle=False))
    print(f"預先載入 {len(data.items)} 句、{data.frames / 25 / 3600:.0f} 小時，用咗 {time.time() - t_load:.0f}s"
          + (f"；粵拼 {len(jy_vocab)} 個音節" if jy_vocab is not None else ""))

model = Reader(vocab["emb"], d_model=args.d_model, layers=args.layers, heads=args.heads, dropout=args.dropout,
               head=args.head, feat_layers=feat_layers, jy_classes=len(jy_vocab) if jy_vocab else 0,
               jy_layer=args.jy_layer, pca_dims={l: c.shape[1] for l, (_, c) in pca.items()}).to(device)
for l, (mean, comps) in pca.items():
    getattr(model, f"pca_mean_{l}").copy_(mean)
    getattr(model, f"pca_comps_{l}").copy_(comps)
print(f"reader 參數：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M，"
      f"字典 {len(vocab['units'])}，head={args.head}，feature 層 {list(feat_layers)}，"
      f"粵拼比重 {args.jy_weight}（第 {args.jy_layer} 層），train shard {len(train_paths)}，val shard {len(val_paths)}")
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


def save(path: Path):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "epoch": epoch, "best_cer": best_cer, "config": model.config,
                "units": vocab["units"], "vocab_emb": vocab["emb"], "jy_vocab": jy_vocab}, path)


@torch.no_grad()
def evaluate(show: int = 5) -> float:
    model.eval()
    errors = total = 0
    examples = []
    val = (batches(val_paths, unit_to_id, args.max_frames, False) if args.stream
           else (val_data.collate(g) for g in val_data.batches(args.max_frames, shuffle=False)))
    for feats, lengths, targets, target_lengths, *_ in val:
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
    if args.stream:
        source = ((f, l, t, tl) for f, l, t, tl, _ in batches(train_paths, unit_to_id, args.max_frames, True))
    else:
        source = prefetch(data, data.batches(args.max_frames, shuffle=True, seed=epoch))
    for feats, lengths, targets, target_lengths, *jy in source:
        feats, lengths = feats.to(device, non_blocking=True), lengths.to(device)
        if args.time_masks or args.feat_masks:
            feats = augment_gpu(feats, lengths, args.time_masks, args.time_mask_width,
                                args.feat_masks, args.feat_mask_width)
        with torch.autocast("cuda", dtype=amp_dtype):
            res = model(feats, lengths, return_jy=bool(jy))
        log_probs, jy_lp = res if jy else (res, None)
        loss = F.ctc_loss(log_probs.float().transpose(0, 1), targets.to(device), lengths, target_lengths.to(device),
                          blank=0, zero_infinity=True)
        if jy:
            jy_loss = F.ctc_loss(jy_lp.transpose(0, 1), jy[0].to(device), lengths, jy[1].to(device),
                                 blank=0, zero_infinity=True)
            loss = (1 - args.jy_weight) * loss + args.jy_weight * jy_loss
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
                  f"lr {sched.get_last_lr()[0]:.2e}  scale {model.log_scale.exp().item():.1f}  {time.time() - t0:.0f}s"
                  + (f"  層權重 {' '.join(f'{w:.2f}' for w in model.mix_weights())}" if len(feat_layers) > 1 else ""))
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
        if args.max_steps and step >= args.max_steps:
            break
    epoch += 1
    if args.max_steps and step >= args.max_steps:
        break

cer = evaluate()
print(f"完成：最後 CER {cer:.2%}，最好 {min(best_cer, cer):.2%}")
log.write(json.dumps({"step": step, "epoch": epoch, "cer": cer}) + "\n")
if cer < best_cer:
    best_cer = cer
    save(out / "best.pt")
save(ckpt_path)

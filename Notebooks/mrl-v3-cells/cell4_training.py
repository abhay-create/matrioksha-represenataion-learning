# ═══════════════════════════════════════════════════
# CELL 4 — TRAINING ENGINE
# Model-pair parallelism: Standard on GPU 0, MRL on GPU 1
# Early stopping, checkpoint resume, on-the-fly negative sampling
# ═══════════════════════════════════════════════════
# v3.2 fixes:
#   ▸ Per-batch random sampling (torch.randint) instead of
#     torch.randperm which needs ~1.5-3GB for 189M indices
#   ▸ Zero extra GPU memory beyond pairs + model + optimizer
#   ▸ Batch size 32768 to fit T4 15GB
# ═══════════════════════════════════════════════════

import os, time, threading, pickle, gc
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class NegativeSampler:
    """On-the-fly negative sampling entirely on GPU."""
    def __init__(self, vocab, device):
        self.probs  = torch.tensor(vocab.neg_sample_probs()).to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size):
        return torch.multinomial(
            self.probs, num_samples=batch_size * NEG_SAMPLES, replacement=True
        ).view(batch_size, NEG_SAMPLES)


# ── Checkpoint helpers ───────────────────────────────────────────────────────
def _save_ckpt(model, optimizer, scheduler, path, mtype, vsz, ep, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_type":      mtype,
        "vocab_size":      vsz,
        "epochs_trained":  ep,
        "best_loss":       best,
    }, path)

def _load_ckpt(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        return 0, float("inf")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epochs_trained", 0), ckpt.get("best_loss", float("inf"))


# ── Skip-gram training loop (Word2Vec / FastText) ────────────────────────────
def _train_skipgram(rank, model, pairs_np, vocab, save_path, model_type,
                    epochs=EPOCHS):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model = model.to(device)

    neg_sampler = NegativeSampler(vocab, device)

    # Load pairs to GPU — keep as the only copy, no duplication
    pairs_gpu = torch.tensor(np.array(pairs_np), dtype=torch.long).to(device)
    N         = len(pairs_gpu)
    bs        = BATCH_SIZE
    steps_ep  = N // bs
    total_steps = epochs * steps_ep

    print(f"  [{model_type}@GPU{rank}] Pairs: {N:,} | "
          f"Batch: {bs} | Steps/ep: {steps_ep} | "
          f"GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR*0.01)

    ep_done, best_loss = _load_ckpt(model, optimizer, scheduler, save_path, device)
    if ep_done > 0:
        print(f"  [{model_type}@GPU{rank}] Resumed from epoch {ep_done}")
    else:
        print(f"  [{model_type}@GPU{rank}] Training from scratch")

    patience_ctr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        # Per-batch random sampling — O(batch) GPU memory, not O(N)
        # Equivalent to SGD with replacement; standard for word embeddings
        pbar = tqdm(range(steps_ep),
                    desc=f"[{model_type}] Ep {ep_done+epoch}/{ep_done+epochs}",
                    dynamic_ncols=True, position=rank)
        for i in pbar:
            idx       = torch.randint(0, N, (bs,), device=device)
            batch     = pairs_gpu[idx]
            centers   = batch[:, 0]
            contexts  = batch[:, 1]
            negatives = neg_sampler.sample(len(centers))

            optimizer.zero_grad(set_to_none=True)
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if i % 100 == 0:
                pbar.set_postfix(loss=f"{epoch_loss/(i+1):.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = epoch_loss / steps_ep
        elapsed = time.time() - t0
        print(f"  [{model_type}@GPU{rank}] Ep {ep_done+epoch} | "
              f"loss {avg:.4f} | {elapsed:.0f}s | "
              f"{steps_ep*bs/elapsed/1e6:.2f}M pairs/s")

        # Early stopping
        if best_loss - avg > MIN_DELTA:
            best_loss = avg
            patience_ctr = 0
        else:
            patience_ctr += 1

        _save_ckpt(model, optimizer, scheduler, save_path,
                   model_type, vocab.size, ep_done + epoch, best_loss)

        if patience_ctr >= PATIENCE:
            print(f"  [{model_type}@GPU{rank}] Early stop at epoch {ep_done+epoch}")
            break

    return model


# ── GloVe training loop ─────────────────────────────────────────────────────
def _train_glove(rank, model, cooc_row, cooc_col, cooc_data, vocab,
                 save_path, model_type, epochs=EPOCHS):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model = model.to(device)

    row_gpu  = torch.tensor(np.array(cooc_row),  dtype=torch.long).to(device)
    col_gpu  = torch.tensor(np.array(cooc_col),  dtype=torch.long).to(device)
    data_gpu = torch.tensor(np.array(cooc_data), dtype=torch.float32).to(device)
    N        = len(row_gpu)
    bs       = BATCH_SIZE
    steps_ep = N // bs
    total_steps = epochs * steps_ep

    print(f"  [{model_type}@GPU{rank}] Co-oc entries: {N:,} | "
          f"GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR*0.01)

    ep_done, best_loss = _load_ckpt(model, optimizer, scheduler, save_path, device)
    if ep_done > 0:
        print(f"  [{model_type}@GPU{rank}] Resumed from epoch {ep_done}")
    else:
        print(f"  [{model_type}@GPU{rank}] Training from scratch")

    patience_ctr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        pbar = tqdm(range(steps_ep),
                    desc=f"[{model_type}] Ep {ep_done+epoch}/{ep_done+epochs}",
                    dynamic_ncols=True, position=rank)
        for i in pbar:
            idx = torch.randint(0, N, (bs,), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(row_gpu[idx], col_gpu[idx], data_gpu[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if i % 100 == 0:
                pbar.set_postfix(loss=f"{epoch_loss/(i+1):.4f}")

        avg = epoch_loss / steps_ep
        elapsed = time.time() - t0
        print(f"  [{model_type}@GPU{rank}] Ep {ep_done+epoch} | "
              f"loss {avg:.4f} | {elapsed:.0f}s")

        if best_loss - avg > MIN_DELTA:
            best_loss = avg
            patience_ctr = 0
        else:
            patience_ctr += 1

        _save_ckpt(model, optimizer, scheduler, save_path,
                   model_type, vocab.size, ep_done + epoch, best_loss)

        if patience_ctr >= PATIENCE:
            print(f"  [{model_type}@GPU{rank}] Early stop at epoch {ep_done+epoch}")
            break

    return model


# ── Pair launcher: Standard on GPU 0, MRL on GPU 1 ──────────────────────────
def train_pair(std_fn, mrl_fn):
    """
    Run two training functions concurrently — one per GPU.
    GIL is released during CUDA ops so both GPUs run at full speed.
    """
    results = [None, None]
    errors  = [None, None]

    def wrapper(idx, fn):
        try:
            results[idx] = fn()
        except Exception as ex:
            errors[idx] = ex
            import traceback; traceback.print_exc()

    t0 = threading.Thread(target=wrapper, args=(0, std_fn))
    t1 = threading.Thread(target=wrapper, args=(1, mrl_fn))
    t0.start(); t1.start()
    t0.join();  t1.join()

    for i, err in enumerate(errors):
        if err:
            raise RuntimeError(f"GPU {i} training failed: {err}")
    return results


print("Training engine loaded.")

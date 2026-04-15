# ═══════════════════════════════════════════════════
# CELL 2 — DATA
# Pile streaming, Vocabulary, SkipGram/Cooccurrence/FastText datasets
# ═══════════════════════════════════════════════════

import os, pickle, time, threading
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix, csr_matrix

PAIRS_CACHE = f"{DATA_DIR}/skipgram_pairs.npy"
COOC_CACHE  = f"{DATA_DIR}/cooc.npz"


# ── Corpus download ──────────────────────────────────────────────────────────
def download_corpus(target_tokens=TARGET_TOKENS):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CORPUS_CACHE):
        print(f"Loading cached tokens from {CORPUS_CACHE}")
        with open(CORPUS_CACHE, "rb") as f:
            tokens = pickle.load(f)
        print(f"  {len(tokens):,} tokens")
        return tokens

    print(f"Streaming {target_tokens/1e6:.0f}M tokens from The Pile …")
    from datasets import load_dataset
    ds = load_dataset(PILE_DATASET, split="train", streaming=True)

    tokens = []
    for doc in ds:
        words = doc["text"].lower().split()
        tokens.extend(words)
        if len(tokens) % 5_000_000 < len(words):
            print(f"  {len(tokens)/1e6:.1f}M …")
        if len(tokens) >= target_tokens:
            break
    tokens = tokens[:target_tokens]

    with open(CORPUS_CACHE, "wb") as f:
        pickle.dump(tokens, f)
    print(f"  Cached {len(tokens):,} tokens")
    return tokens


# ── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, tokens):
        counts = Counter(tokens)
        vocab  = [w for w, c in counts.most_common(VOCAB_SIZE) if c >= MIN_COUNT]
        self.w2i    = {w: i for i, w in enumerate(vocab)}
        self.i2w    = vocab
        self.size   = len(vocab)
        self.counts = np.array([counts[w] for w in vocab], dtype=np.float32)

        freq = self.counts / self.counts.sum()
        self.keep_prob = np.minimum(
            1.0,
            np.sqrt(SUBSAMPLE_T / np.maximum(freq, 1e-12))
            + SUBSAMPLE_T / np.maximum(freq, 1e-12),
        ).astype(np.float32)

    def encode(self, tokens):
        return np.array([self.w2i[t] for t in tokens if t in self.w2i], dtype=np.int32)

    def neg_sample_probs(self):
        p = self.counts ** 0.75
        return (p / p.sum()).astype(np.float32)


# ── FastText Vocabulary (extends Vocabulary with n-gram indices) ─────────────
class FastTextVocabulary(Vocabulary):
    def __init__(self, tokens):
        super().__init__(tokens)
        # Pre-compute padded n-gram hash matrix for all vocab words
        all_hashes = []
        for word in self.i2w:
            bounded = f"<{word}>"
            hashes = []
            for n in range(FT_NGRAM_LO, FT_NGRAM_HI + 1):
                for i in range(len(bounded) - n + 1):
                    hashes.append(hash(bounded[i:i+n]) % FT_BUCKETS)
            all_hashes.append(hashes)

        self.max_ngrams = max(len(h) for h in all_hashes)
        self.ngram_matrix  = np.zeros((self.size, self.max_ngrams), dtype=np.int32)
        self.ngram_lengths = np.zeros(self.size, dtype=np.int32)
        for i, hashes in enumerate(all_hashes):
            self.ngram_matrix[i, :len(hashes)] = hashes
            self.ngram_lengths[i] = len(hashes)
        print(f"  FastText n-grams: max {self.max_ngrams} per word, {FT_BUCKETS} buckets")


# ── Skip-gram pair dataset ──────────────────────────────────────────────────
def build_skipgram_pairs(vocab, tokens):
    """Vectorised skip-gram pair construction with subsampling. Cached to disk."""
    if os.path.exists(PAIRS_CACHE):
        print(f"Loading cached pairs from {PAIRS_CACHE} …")
        pairs = np.load(PAIRS_CACHE, mmap_mode="r")
        print(f"  {len(pairs):,} pairs  ({pairs.nbytes/1e6:.0f} MB)")
        return pairs

    print("Building skip-gram pairs (vectorised) …")
    ids  = vocab.encode(tokens)
    mask = np.random.rand(len(ids)).astype(np.float32) < vocab.keep_prob[ids]
    ids  = ids[mask]
    print(f"  After subsampling: {len(ids):,} tokens")

    chunks = []
    for offset in range(1, WINDOW_SIZE + 1):
        chunks.append(np.stack([ids[:-offset], ids[offset:]], axis=1))
        chunks.append(np.stack([ids[offset:],  ids[:-offset]], axis=1))
    pairs = np.concatenate(chunks, axis=0).astype(np.int32)

    np.save(PAIRS_CACHE, pairs)
    print(f"  {len(pairs):,} pairs  ({pairs.nbytes/1e6:.0f} MB)")
    return pairs


# ── Co-occurrence matrix for GloVe ──────────────────────────────────────────
def build_cooccurrence(vocab, tokens):
    """Build GloVe co-occurrence matrix with 1/distance weighting. No subsampling."""
    if os.path.exists(COOC_CACHE):
        print(f"Loading cached co-occurrence from {COOC_CACHE} …")
        data = np.load(COOC_CACHE)
        print(f"  {len(data['row']):,} non-zero entries")
        return data["row"], data["col"], data["data"]

    print("Building co-occurrence matrix …")
    ids  = vocab.encode(tokens)
    V    = vocab.size
    t0   = time.time()

    accumulated = csr_matrix((V, V), dtype=np.float64)
    for offset in range(1, WINDOW_SIZE + 1):
        weight = 1.0 / offset
        n = len(ids) - offset
        rows = ids[:-offset]
        cols = ids[offset:]
        vals = np.full(n, weight, dtype=np.float64)
        # Forward
        accumulated += coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
        # Backward
        accumulated += coo_matrix((vals, (cols, rows)), shape=(V, V)).tocsr()
        print(f"  offset {offset}/{WINDOW_SIZE} done ({time.time()-t0:.0f}s)")

    cooc = accumulated.tocoo()
    row  = cooc.row.astype(np.int32)
    col  = cooc.col.astype(np.int32)
    data = cooc.data.astype(np.float32)

    np.savez(COOC_CACHE, row=row, col=col, data=data)
    print(f"  {len(row):,} non-zero entries ({time.time()-t0:.0f}s total)")
    return row, col, data


print("Data module loaded.")

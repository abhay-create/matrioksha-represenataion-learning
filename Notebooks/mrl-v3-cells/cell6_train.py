# ═══════════════════════════════════════════════════
# CELL 6 — TRAINING CALLS
# 6 individual train functions + 3 pair functions.
# Each resumes from checkpoint. Rerun for more epochs.
# Self-contained: loads data from cache if globals are missing.
# ═══════════════════════════════════════════════════

def _ensure_data():
    """Load vocab, pairs, cooccurrence from cache if not already in globals."""
    global vocab, ft_vocab, pairs, cooc_row, cooc_col, cooc_data

    if "vocab" not in globals() or vocab is None:
        print("Loading vocab from cache …")
        with open(f"{SAVE_DIR}/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        print(f"  Vocabulary: {vocab.size:,} words")

    if "ft_vocab" not in globals() or ft_vocab is None:
        print("Loading FastText vocab from cache …")
        with open(f"{SAVE_DIR}/ft_vocab.pkl", "rb") as f:
            ft_vocab = pickle.load(f)

    if "pairs" not in globals() or pairs is None:
        print("Loading skip-gram pairs from cache …")
        pairs = np.load(PAIRS_CACHE, mmap_mode="r")
        print(f"  Pairs: {len(pairs):,}")

    if "cooc_row" not in globals() or cooc_row is None:
        print("Loading co-occurrence from cache …")
        data = np.load(COOC_CACHE)
        cooc_row  = data["row"]
        cooc_col  = data["col"]
        cooc_data = data["data"]
        print(f"  Co-oc entries: {len(cooc_row):,}")


# ── Word2Vec ─────────────────────────────────────────────────────────────────
def train_standard_w2v(epochs=EPOCHS):
    _ensure_data()
    model = StandardWord2Vec(vocab.size)
    return _train_skipgram(0, model, pairs, vocab, STD_W2V_CKPT,
                           "standard_w2v", epochs)

def train_mrl_w2v(epochs=EPOCHS):
    _ensure_data()
    model = MRLWord2Vec(vocab.size)
    return _train_skipgram(1, model, pairs, vocab, MRL_W2V_CKPT,
                           "mrl_w2v", epochs)

def train_w2v_pair(epochs=EPOCHS):
    """Standard W2V on GPU 0 + MRL W2V on GPU 1 simultaneously."""
    _ensure_data()
    print("="*60)
    print("  Word2Vec pair training (Standard ↔ MRL)")
    print("="*60)
    return train_pair(
        lambda: train_standard_w2v(epochs),
        lambda: train_mrl_w2v(epochs),
    )


# ── GloVe ────────────────────────────────────────────────────────────────────
def train_standard_glove(epochs=EPOCHS):
    _ensure_data()
    model = StandardGloVe(vocab.size)
    return _train_glove(0, model, cooc_row, cooc_col, cooc_data, vocab,
                        STD_GLOVE_CKPT, "standard_glove", epochs)

def train_mrl_glove(epochs=EPOCHS):
    _ensure_data()
    model = MRLGloVe(vocab.size)
    return _train_glove(1, model, cooc_row, cooc_col, cooc_data, vocab,
                        MRL_GLOVE_CKPT, "mrl_glove", epochs)

def train_glove_pair(epochs=EPOCHS):
    """Standard GloVe on GPU 0 + MRL GloVe on GPU 1 simultaneously."""
    _ensure_data()
    print("="*60)
    print("  GloVe pair training (Standard ↔ MRL)")
    print("="*60)
    return train_pair(
        lambda: train_standard_glove(epochs),
        lambda: train_mrl_glove(epochs),
    )


# ── FastText ─────────────────────────────────────────────────────────────────
def train_standard_fasttext(epochs=EPOCHS):
    _ensure_data()
    model = StandardFastText(ft_vocab.size, ft_vocab.ngram_matrix,
                             ft_vocab.ngram_lengths)
    return _train_skipgram(0, model, pairs, ft_vocab, STD_FT_CKPT,
                           "standard_fasttext", epochs)

def train_mrl_fasttext(epochs=EPOCHS):
    _ensure_data()
    model = MRLFastText(ft_vocab.size, ft_vocab.ngram_matrix,
                        ft_vocab.ngram_lengths)
    return _train_skipgram(1, model, pairs, ft_vocab, MRL_FT_CKPT,
                           "mrl_fasttext", epochs)

def train_fasttext_pair(epochs=EPOCHS):
    """Standard FastText on GPU 0 + MRL FastText on GPU 1 simultaneously."""
    _ensure_data()
    print("="*60)
    print("  FastText pair training (Standard ↔ MRL)")
    print("="*60)
    return train_pair(
        lambda: train_standard_fasttext(epochs),
        lambda: train_mrl_fasttext(epochs),
    )


print("Training functions ready.")
print("  Individual: train_standard_w2v(), train_mrl_w2v(), etc.")
print("  Pairs:      train_w2v_pair(), train_glove_pair(), train_fasttext_pair()")

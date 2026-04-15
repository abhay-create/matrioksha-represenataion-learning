# ═══════════════════════════════════════════════════
# CELL 5 — PREPARE DATA
# Download corpus, build vocab, build and cache all datasets.
# Run once; subsequent runs load from cache.
# ═══════════════════════════════════════════════════

tokens = download_corpus()
vocab  = Vocabulary(tokens)
print(f"Vocabulary: {vocab.size:,} words")

ft_vocab = FastTextVocabulary(tokens)

os.makedirs(SAVE_DIR, exist_ok=True)
with open(f"{SAVE_DIR}/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open(f"{SAVE_DIR}/ft_vocab.pkl", "wb") as f:
    pickle.dump(ft_vocab, f)

# Build skip-gram pairs (used by Word2Vec + FastText)
pairs = build_skipgram_pairs(vocab, tokens)

# Build co-occurrence matrix (used by GloVe)
cooc_row, cooc_col, cooc_data = build_cooccurrence(vocab, tokens)

print(f"\n✓ Data ready. Pairs: {len(pairs):,} | Co-oc entries: {len(cooc_row):,}")

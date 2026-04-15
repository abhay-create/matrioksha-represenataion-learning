# ═══════════════════════════════════════════════════
# CELL 1 — CONFIG
# ═══════════════════════════════════════════════════

# ── Corpus ────────────────────────────────────────
PILE_DATASET  = "monology/pile-uncopyrighted"
TARGET_TOKENS = 50_000_000          # adjustable 10M–100M
CORPUS_CACHE  = "data/pile_tokens.pkl"

# ── Vocabulary ────────────────────────────────────
VOCAB_SIZE = None                   # keep all above MIN_COUNT
MIN_COUNT  = 10

# ── Embedding ─────────────────────────────────────
EMBED_DIM   = 300
MRL_NESTING = [50, 100, 150, 200, 250, 300]

# ── Training ──────────────────────────────────────
WINDOW_SIZE = 5
NEG_SAMPLES = 10
EPOCHS      = 5                     # per call; rerun to continue
BATCH_SIZE  = 32768
LR          = 0.001
SUBSAMPLE_T = 1e-5

# ── Early Stopping ────────────────────────────────
PATIENCE  = 2
MIN_DELTA = 0.01

# ── GloVe ─────────────────────────────────────────
GLOVE_X_MAX = 100
GLOVE_ALPHA = 0.75

# ── FastText ──────────────────────────────────────
FT_NGRAM_LO = 3
FT_NGRAM_HI = 6
FT_BUCKETS  = 200_000

# ── Paths ─────────────────────────────────────────
SAVE_DIR       = "checkpoints"
DATA_DIR       = "data"
STD_W2V_CKPT   = f"{SAVE_DIR}/standard_w2v.pt"
MRL_W2V_CKPT   = f"{SAVE_DIR}/mrl_w2v.pt"
STD_GLOVE_CKPT = f"{SAVE_DIR}/standard_glove.pt"
MRL_GLOVE_CKPT = f"{SAVE_DIR}/mrl_glove.pt"
STD_FT_CKPT    = f"{SAVE_DIR}/standard_fasttext.pt"
MRL_FT_CKPT    = f"{SAVE_DIR}/mrl_fasttext.pt"

print("Config loaded.")

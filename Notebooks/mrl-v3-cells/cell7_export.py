# ═══════════════════════════════════════════════════
# CELL 7 — EXPORT
# Save all 6 embedding matrices + vocab to zip.
# ═══════════════════════════════════════════════════

import zipfile
from IPython.display import FileLink, display


def load_model(save_path):
    """Load model from checkpoint, auto-detecting type and vocab size."""
    ckpt       = torch.load(save_path, map_location="cpu", weights_only=False)
    vocab_size = ckpt["vocab_size"]
    mtype      = ckpt["model_type"]

    # Reconstruct model by type
    if "fasttext" in mtype:
        with open(f"{SAVE_DIR}/ft_vocab.pkl", "rb") as f:
            ftv = pickle.load(f)
        if "mrl" in mtype:
            model = MRLFastText(vocab_size, ftv.ngram_matrix, ftv.ngram_lengths)
        else:
            model = StandardFastText(vocab_size, ftv.ngram_matrix, ftv.ngram_lengths)
    elif "glove" in mtype:
        if "mrl" in mtype:
            model = MRLGloVe(vocab_size)
        else:
            model = StandardGloVe(vocab_size)
    else:  # w2v
        if "mrl" in mtype:
            model = MRLWord2Vec(vocab_size)
        else:
            model = StandardWord2Vec(vocab_size)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ep = ckpt.get("epochs_trained", "?")
    bl = ckpt.get("best_loss", "?")
    print(f"  Loaded {mtype} | vocab={vocab_size} | epochs={ep} | best_loss={bl}")
    return model


# ── Export all available models ──────────────────────────────────────────────
CKPT_MAP = {
    "standard_w2v":      STD_W2V_CKPT,
    "mrl_w2v":           MRL_W2V_CKPT,
    "standard_glove":    STD_GLOVE_CKPT,
    "mrl_glove":         MRL_GLOVE_CKPT,
    "standard_fasttext": STD_FT_CKPT,
    "mrl_fasttext":      MRL_FT_CKPT,
}

os.makedirs("exports", exist_ok=True)
exported = {}

for name, ckpt_path in CKPT_MAP.items():
    if os.path.exists(ckpt_path):
        model = load_model(ckpt_path)
        emb   = model.get_embeddings()
        np.save(f"exports/{name}_embeddings.npy", emb)
        exported[name] = emb
        print(f"    {name}: {emb.shape}  ({emb.nbytes/1e6:.1f} MB)")
    else:
        print(f"    {name}: NOT TRAINED (skipped)")

# Save vocab
with open(f"{SAVE_DIR}/vocab.pkl", "rb") as f:
    v = pickle.load(f)
np.save("exports/vocab_words.npy", np.array(v.i2w))
print(f"\nVocab: {len(v.i2w)} words")

# ── Zip everything ───────────────────────────────────────────────────────────
zip_name = "mrl_bias_v3_embeddings.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir("exports"):
        zf.write(f"exports/{fn}", fn)
    zf.write(f"{SAVE_DIR}/vocab.pkl", "vocab.pkl")
    if os.path.exists(f"{SAVE_DIR}/ft_vocab.pkl"):
        zf.write(f"{SAVE_DIR}/ft_vocab.pkl", "ft_vocab.pkl")
print(f"\nEmbeddings zip: {os.path.getsize(zip_name)/1e6:.1f} MB")

# Checkpoints zip
ckpt_zip = "mrl_bias_v3_checkpoints.zip"
with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(SAVE_DIR):
        fp = os.path.join(SAVE_DIR, fn)
        zf.write(fp, fn)
        print(f"  Added: {fn} ({os.path.getsize(fp)/1e6:.1f} MB)")
print(f"Checkpoints zip: {os.path.getsize(ckpt_zip)/1e6:.1f} MB")

display(FileLink(zip_name))
display(FileLink(ckpt_zip))

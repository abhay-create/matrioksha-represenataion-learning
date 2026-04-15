import numpy as np
import os
import urllib.request
from scipy.stats import spearmanr
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
emb_dir = os.path.join(base_dir, "Data_and_Embeddings", "_v3_embeddings_W2V", "Embeddings", "mrl_bias_v4_embeddings")
vocab_path_v4 = os.path.join(emb_dir, "vocab.pkl")
plot_dir = os.path.join(base_dir, "Results_and_Reports", "geometry_plots")
os.makedirs(plot_dir, exist_ok=True)

print("Loading vocab...")
try:
    with open(vocab_path_v4, "rb") as f:
        vocab = pickle.load(f)
    w2i = {w: i for i, w in enumerate(vocab.i2w)}
except Exception as e:
    print(f"Failed to load vocab via pickle. Loading numpy array... {e}")
    vocab_words = np.load(os.path.join(emb_dir, "vocab_words.npy"))
    w2i = {w: i for i, w in enumerate(vocab_words)}

print("Loading embeddings...")
std_emb = np.load(os.path.join(emb_dir, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(emb_dir, "mrl_v4_w2v_embeddings.npy"))

# Download SimLex-999
simlex_url = "https://fh295.github.io/SimLex-999.zip"
simlex_zip = os.path.join(base_dir, "Data_and_Embeddings", "SimLex-999.zip")
simlex_txt = os.path.join(base_dir, "Data_and_Embeddings", "SimLex-999", "SimLex-999.txt")

if not os.path.exists(simlex_txt):
    import zipfile
    print("Downloading SimLex-999...")
    try:
        urllib.request.urlretrieve(simlex_url, simlex_zip)
        with zipfile.ZipFile(simlex_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(base_dir, "Data_and_Embeddings"))
    except Exception as e:
        print(f"Error downloading SimLex-999: {e}")
        exit(1)

print("Parsing SimLex-999...")
pairs = []
with open(simlex_txt, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[1:]: # skip header
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            w1, w2, score = parts[0], parts[1], float(parts[3])
            if w1 in w2i and w2 in w2i:
                pairs.append((w2i[w1], w2i[w2], score))

print(f"Found {len(pairs)} in-vocabulary pairs out of 999.")

def evaluate_sim(emb_matrix):
    sims = []
    humans = []
    for i1, i2, human_score in pairs:
        v1 = emb_matrix[i1]
        v2 = emb_matrix[i2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            sim = 0
        else:
            sim = np.dot(v1, v2) / (n1 * n2)
        sims.append(sim)
        humans.append(human_score)
    corr, _ = spearmanr(sims, humans)
    return corr

levels = [50, 100, 150, 200, 250, 300]
print("\n=== Intrinsic Word Similarity (Spearman rho) on SimLex-999 ===")
print("Dimension | Standard | MRL-v4 | Diff")
print("--------------------------------------")

res_std = []
res_mrl = []

for d in levels:
    std_corr = evaluate_sim(std_emb[:, :d])
    mrl_corr = evaluate_sim(mrl_emb[:, :d])
    res_std.append(std_corr)
    res_mrl.append(mrl_corr)
    diff = mrl_corr - std_corr
    print(f"  {d:<7} |  {std_corr:.4f}  | {mrl_corr:.4f} | {diff:+.4f}")

# Plotting
plt.figure(figsize=(8,5))
plt.plot(levels, res_std, marker='o', label="Standard")
plt.plot(levels, res_mrl, marker='s', label="MRL-v4")
plt.title(f"Intrinsic Word Similarity on SimLex-999 (N={len(pairs)})")
plt.xlabel("Prefix Dimension")
plt.ylabel("Spearman Correlation (rho)")
plt.legend()
plt.grid(True)
out_plot = os.path.join(plot_dir, "exp_intrinsic_simlex999_v4.png")
plt.savefig(out_plot)
plt.close()
print(f"\nPlot saved to {out_plot}")

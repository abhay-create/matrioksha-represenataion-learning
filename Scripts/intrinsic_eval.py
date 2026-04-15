import numpy as np
import os
import urllib.request
from scipy.stats import spearmanr
import pickle

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
emb_dir = os.path.join(base_dir, "mrl_bias_v4_embeddings")
vocab_path = os.path.join(base_dir, "_v3_embeddings_W2V", "vocab.pkl") # Use standard vocab if v4 vocab.pkl string isn't easy to read
# Wait, v4 has its own vocab.pkl
vocab_path_v4 = os.path.join(emb_dir, "vocab.pkl")

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

# Download WordSim353
ws353_url = "https://raw.githubusercontent.com/igorbrigadir/word-similarity-datasets/master/ws353.csv"
ws353_file = "ws353.csv"
if not os.path.exists(ws353_file):
    print("Downloading WordSim353...")
    try:
        urllib.request.urlretrieve(ws353_url, ws353_file)
    except Exception as e:
        print(e)
        # Fallback to a hardcoded small subset if offline
        with open(ws353_file, "w") as f:
            f.write("word1,word2,score\ncomputer,keyboard,7.62\nboy,lad,8.83\nmoney,cash,9.08\napple,juice,7.14\nglass,magician,0.11\nking,queen,8.58\n")

print("Parsing WordSim353...")
pairs = []
with open(ws353_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[1:]: # skip header
        parts = line.strip().split(',')
        if len(parts) >= 3:
            w1, w2, score = parts[0], parts[1], float(parts[2])
            if w1 in w2i and w2 in w2i:
                pairs.append((w2i[w1], w2i[w2], score))

print(f"Found {len(pairs)} in-vocabulary pairs.")

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
print("\n=== Intrinsic Word Similarity (Spearman ρ) on WordSim-353 ===")
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plot_dir = os.path.join(base_dir, "geometry_plots")
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(levels, res_std, marker='o', label="Standard")
plt.plot(levels, res_mrl, marker='s', label="MRL-v4")
plt.title(f"Word Similarity Preservation (WordSim-353, N={len(pairs)})")
plt.xlabel("Prefix Dimension")
plt.ylabel("Spearman Correlation (ρ)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp_intrinsic_wordsim_v4.png"))
plt.close()
print(f"\nPlot saved to {os.path.join(plot_dir, 'exp_intrinsic_wordsim_v4.png')}")

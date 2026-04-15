import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import pickle
from sklearn.metrics import pairwise_distances

matplotlib.use('Agg')

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS\mrl_bias_v4_embeddings"
std_path = os.path.join(base_dir, "standard_w2v_embeddings.npy")
mrl_path = os.path.join(base_dir, "mrl_v4_w2v_embeddings.npy")
plot_dir = r"c:\Users\abhay\Desktop\MRL TESTS\geometry_plots"
os.makedirs(plot_dir, exist_ok=True)

# Load embeddings
print("Loading embeddings...")
std_emb = np.load(std_path)
mrl_emb = np.load(mrl_path)

vocab_size, full_dim = std_emb.shape
print(f"Loaded Standard shape: {std_emb.shape}")
print(f"Loaded MRL-v4 shape: {mrl_emb.shape}")

# Top N most frequent words (assuming ordered by frequency)
N_WORDS = 5000
K = 10
nesting_levels = [50, 100, 150, 200, 250]

std_sub = std_emb[:N_WORDS]
mrl_sub = mrl_emb[:N_WORDS]

print(f"\\n--- Experiment 9: Jaccard Neighborhood Preservation ---")
print(f"Computing Top-{K} neighbors for the most frequent {N_WORDS} words...")

def get_top_k_indices(emb, k=10):
    # Normalize for fast cosine similarity via dot product
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    emb_norm = emb / norms
    sim_matrix = emb_norm @ emb_norm.T
    # Fill diagonal with -inf to ignore self
    np.fill_diagonal(sim_matrix, -np.inf)
    # Get top K indices
    return np.argsort(-sim_matrix, axis=1)[:, :k]

def jaccard_similarity(set1, set2):
    intersection_len = len(set(set1).intersection(set(set2)))
    union_len = len(set(set1).union(set(set2)))
    return intersection_len / union_len if union_len > 0 else 0

# Baseline Reference: Top-10 neighbors in the full 300d embeddings
std_top_k_300 = get_top_k_indices(std_sub)
mrl_top_k_300 = get_top_k_indices(mrl_sub)

std_jaccards = []
mrl_jaccards = []

for d in nesting_levels:
    # Top-10 neighbors at truncated dimension d
    std_top_k_d = get_top_k_indices(std_sub[:, :d])
    mrl_top_k_d = get_top_k_indices(mrl_sub[:, :d])
    
    # Compute mean Jaccard overlap vs 300d reference for the 5000 words
    j_std = [jaccard_similarity(std_top_k_300[i], std_top_k_d[i]) for i in range(N_WORDS)]
    j_mrl = [jaccard_similarity(mrl_top_k_300[i], mrl_top_k_d[i]) for i in range(N_WORDS)]
    
    mean_j_std = np.mean(j_std)
    mean_j_mrl = np.mean(j_mrl)
    
    std_jaccards.append(mean_j_std)
    mrl_jaccards.append(mean_j_mrl)
    
    print(f"Dim {d:3d} | Standard Jaccard: {mean_j_std:.4f} | MRL-v4 Jaccard: {mean_j_mrl:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(nesting_levels, std_jaccards, marker='o', label="Standard (Truncated)", color='red', linestyle='dashed')
plt.plot(nesting_levels, mrl_jaccards, marker='s', label="MRL-v4 (RD)", color='blue', linewidth=2)

# Also plot the old MRL v3 results for context if we know them approximately:
# Old MRL v3 at 50d was around 0.165, at 100d around 0.40, etc. (we can just focus on v4 vs standard)

plt.axhline(1.0, color='black', linestyle=':', label='Perfect Preservation')

plt.title(f"MRL-v4: Neighborhood Preservation vs Truncation Dimension\\n(Jaccard Similarity of Top-{K} Neighbors, N={N_WORDS})")
plt.xlabel("Prefix Dimension (d)")
plt.ylabel("Mean Jaccard Overlap with 300d Neighborhood")
plt.legend()
plt.grid(True, alpha=0.3)

out_plot = os.path.join(plot_dir, "exp9_jaccard_preservation_v4.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\\nSaved plot to {out_plot}")
print("Experiment complete.")

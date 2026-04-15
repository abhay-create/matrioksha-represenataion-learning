import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import svds
import matplotlib

matplotlib.use('Agg')

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
emb_dir = os.path.join(base_dir, "_v3_embeddings_W2V")
plot_dir = os.path.join(base_dir, "geometry_plots")
os.makedirs(plot_dir, exist_ok=True)

print("Loading embeddings...")
std_emb = np.load(os.path.join(emb_dir, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(emb_dir, "mrl_w2v_embeddings.npy"))
V = std_emb.shape[0]

levels = [50, 100, 150, 200, 250, 300]
buckets = {
    "Top_100": (0, 100), 
    "Top_101_500": (100, 500), 
    "Top_501_5000": (500, 5000),
    "Full_Sample": (0, min(10000, V)) # Use first 10k as a stable proxy for full
}

# --- Exp 7b: Detailed L2 Norm Logging ---
print("\n--- Exp 7b: L2 Norm Logging for All Buckets ---")
for b_name, (start, end) in buckets.items():
    print(f"\nBucket: {b_name}")
    for d in levels:
        std_d = std_emb[start:end, :d]
        mrl_d = mrl_emb[start:end, :d]
        avg_norm_std = np.mean(np.linalg.norm(std_d, axis=1))
        avg_norm_mrl = np.mean(np.linalg.norm(mrl_d, axis=1))
        print(f"  Level {d:<3} | Std L2 = {avg_norm_std:.4f}, MRL L2 = {avg_norm_mrl:.4f}")

# --- Exp 8: WCSS Distance (K-Means Inertia) ---
print("\n--- Exp 8: WCSS (K-Means Inertia) ---")
# K=10 for smaller buckets, K=50 for large buckets
fig, axs = plt.subplots(2, 2, figsize=(16,10))
axs = axs.flatten()

for i, (b_name, (start, end)) in enumerate(buckets.items()):
    wcss_std = []
    wcss_mrl = []
    n_clus = 10 if (end - start) < 500 else 50
    
    for d in levels:
        std_d = std_emb[start:end, :d]
        mrl_d = mrl_emb[start:end, :d]
        
        # We normalize vectors for spherical clustering (cosine distance proxy)
        # Using unnormalized vectors for WCSS strongly couples to L2 norm, altering the semantic clustering interpretation.
        std_d_norm = std_d / (np.linalg.norm(std_d, axis=1, keepdims=True) + 1e-10)
        mrl_d_norm = mrl_d / (np.linalg.norm(mrl_d, axis=1, keepdims=True) + 1e-10)
        
        km_std = KMeans(n_clusters=n_clus, n_init=5, random_state=42).fit(std_d_norm)
        km_mrl = KMeans(n_clusters=n_clus, n_init=5, random_state=42).fit(mrl_d_norm)
        
        # We divide WCSS by the number of samples so it's comparable if buckets differ in size, 
        # but here buckets are fixed per iteration, so absolute inertia is fine as a metric of compactness.
        wcss_std.append(km_std.inertia_ / (end - start))
        wcss_mrl.append(km_mrl.inertia_ / (end - start))
        
    print(f"WCSS ({b_name}) | Std d=50 -> d=300: {wcss_std[0]:.4f} -> {wcss_std[-1]:.4f}")
    print(f"WCSS ({b_name}) | MRL d=50 -> d=300: {wcss_mrl[0]:.4f} -> {wcss_mrl[-1]:.4f}")
    
    axs[i].plot(levels, wcss_std, marker='o', label="Standard")
    axs[i].plot(levels, wcss_mrl, marker='s', label="MRL")
    axs[i].set_title(f"WCSS per Sample - {b_name} (K={n_clus})")
    axs[i].set_xlabel("Prefix Dimension")
    axs[i].set_ylabel("Inertia / N")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "exp8_wcss_inertia.png"))
plt.close()

# --- Exp 9: Neighborhood Preservation (Jaccard Sim of Top K) ---
print("\n--- Exp 9: Neighborhood Preservation (Top 10 Jaccard) ---")
np.random.seed(42)
# Sample 1000 words to act as queries.
# We will compare their 10 nearest neighbors in full space vs truncated space
K_neighbors = 10
sample_queries = np.random.choice(V, 1000, replace=False)

def get_top_k_neighbors(embeddings, queries_idx, k=10):
    # calculate dot product of queries against all V
    # Normalize heavily to avoid massive memory usage. We'll only search within top 50k vocab to save RAM if needed, 
    # but V is 175k. A 1000 x 175k matrix is 175 million floats (~700MB), very safe.
    q_vecs = embeddings[queries_idx]
    q_norm = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-10)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    sims = np.dot(q_norm, emb_norm.T) # shape: (1000, 175523)
    # Get top K+1 because itself is always top 1
    top_indices = np.argsort(sims, axis=1)[:, -(k+1):]
    # Reverse so closest is first, and drop the self index (which is at index 0 after reverse)
    # Actually self isn't strictly guaranteed to be at 0 index if precision issues occur, 
    # but practically we can just strip the last element before reverse or filter.
    # We will just collect sets of indices for Jaccard.
    neighbor_sets = []
    for i, row in enumerate(top_indices):
        s = set(row)
        s.discard(queries_idx[i]) # remove self
        neighbor_sets.append(s)
    return neighbor_sets

print("Calculating baseline neighborhoods at d=300...")
std_base_neighbors = get_top_k_neighbors(std_emb, sample_queries, k=K_neighbors)
mrl_base_neighbors = get_top_k_neighbors(mrl_emb, sample_queries, k=K_neighbors)

jaccard_std = []
jaccard_mrl = []

for d in levels:
    print(f"Calculating neighborhoods at d={d}...")
    std_d_neighbors = get_top_k_neighbors(std_emb[:, :d], sample_queries, k=K_neighbors)
    mrl_d_neighbors = get_top_k_neighbors(mrl_emb[:, :d], sample_queries, k=K_neighbors)
    
    std_jaccard_scores = []
    mrl_jaccard_scores = []
    
    for i in range(1000):
        # Std
        inter_std = std_base_neighbors[i].intersection(std_d_neighbors[i])
        union_std = std_base_neighbors[i].union(std_d_neighbors[i])
        std_jaccard_scores.append(len(inter_std) / len(union_std))
        
        # MRL
        inter_mrl = mrl_base_neighbors[i].intersection(mrl_d_neighbors[i])
        union_mrl = mrl_base_neighbors[i].union(mrl_d_neighbors[i])
        mrl_jaccard_scores.append(len(inter_mrl) / len(union_mrl))
        
    avg_jaccard_std = np.mean(std_jaccard_scores)
    avg_jaccard_mrl = np.mean(mrl_jaccard_scores)
    
    jaccard_std.append(avg_jaccard_std)
    jaccard_mrl.append(avg_jaccard_mrl)
    
    print(f"Jaccard at d={d:<3} | Std: {avg_jaccard_std:.4f}, MRL: {avg_jaccard_mrl:.4f}")

plt.figure(figsize=(8,5))
plt.plot(levels, jaccard_std, marker='o', label="Standard Model")
plt.plot(levels, jaccard_mrl, marker='s', label="MRL Model")
plt.title("Neighborhood Preservation (Jaccard Score vs d=300 baseline)")
plt.xlabel("Prefix Dimension")
plt.ylabel("Avg Jaccard Similarity of Top 10 Neighbors")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp9_neighborhood_preservation.png"))
plt.close()

# --- Exp 10: Singular Value Spectrum ---
print("\n--- Exp 10: Singular Value Spectrum (SVD Decay) ---")
# Use a 10k random sample (or first 10k) to compute SVD quickly and stably
sample_std = std_emb[:10000]
sample_mrl = mrl_emb[:10000]

u_s, s_std, vh_s = np.linalg.svd(sample_std - np.mean(sample_std, axis=0), full_matrices=False)
u_m, s_mrl, vh_m = np.linalg.svd(sample_mrl - np.mean(sample_mrl, axis=0), full_matrices=False)

# Normalize singular values so they sum to 1 to compare the density profile
s_std_norm = s_std / np.sum(s_std)
s_mrl_norm = s_mrl / np.sum(s_mrl)

plt.figure(figsize=(10,6))
plt.plot(np.arange(1, 301), s_std_norm, label="Standard", alpha=0.8)
plt.plot(np.arange(1, 301), s_mrl_norm, label="MRL", alpha=0.8)
plt.title("Singular Value Spectrum Decay (Normalized)")
plt.xlabel("Singular Value Rank (1-300)")
plt.ylabel("Relative Singular Value Magnitude")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp10_singular_value_spectrum.png"))
plt.close()

print("\n--- Summary Snippets ---")
print(f"Top 10 SVs % of Total | Std: {np.sum(s_std_norm[:10])*100:.2f}%, MRL: {np.sum(s_mrl_norm[:10])*100:.2f}%")

print("\nAdvanced Analysis Complete! Plots saved to:", plot_dir)

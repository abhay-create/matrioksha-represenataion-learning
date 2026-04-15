import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
import matplotlib

matplotlib.use('Agg')

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
emb_dir = os.path.join(base_dir, "mrl_bias_v4_embeddings")
plot_dir = os.path.join(base_dir, "geometry_plots")
os.makedirs(plot_dir, exist_ok=True)

print("Loading V4 embeddings...")
std_emb = np.load(os.path.join(emb_dir, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(emb_dir, "mrl_v4_w2v_embeddings.npy"))
V = std_emb.shape[0]

levels = [50, 100, 150, 200, 250, 300]
buckets = {
    "Top_100": (0, 100), 
    "Top_101_500": (100, 500), 
    "Top_501_5000": (500, 5000),
    "Full_Sample": (0, min(10000, V))
}

with open(os.path.join(plot_dir, "geometry_results_v4.txt"), "w") as f:
    f.write("=== MRL-v4 Geometric Analysis ===\n\n")

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
fig, axs = plt.subplots(2, 2, figsize=(16,10))
axs = axs.flatten()

for i, (b_name, (start, end)) in enumerate(buckets.items()):
    wcss_std = []
    wcss_mrl = []
    n_clus = 10 if (end - start) < 500 else 50
    
    for d in levels:
        std_d = std_emb[start:end, :d]
        mrl_d = mrl_emb[start:end, :d]
        
        std_d_norm = std_d / (np.linalg.norm(std_d, axis=1, keepdims=True) + 1e-10)
        mrl_d_norm = mrl_d / (np.linalg.norm(mrl_d, axis=1, keepdims=True) + 1e-10)
        
        km_std = KMeans(n_clusters=n_clus, n_init=5, random_state=42).fit(std_d_norm)
        km_mrl = KMeans(n_clusters=n_clus, n_init=5, random_state=42).fit(mrl_d_norm)
        
        wcss_std.append(km_std.inertia_ / (end - start))
        wcss_mrl.append(km_mrl.inertia_ / (end - start))
        
    axs[i].plot(levels, wcss_std, marker='o', label="Standard")
    axs[i].plot(levels, wcss_mrl, marker='s', label="MRL-v4")
    axs[i].set_title(f"WCSS per Sample - {b_name} (K={n_clus})")
    axs[i].set_xlabel("Prefix Dimension")
    axs[i].set_ylabel("Inertia / N")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "exp8_wcss_inertia_v4.png"))
plt.close()


# --- Exp 10: Singular Value Spectrum ---
print("\n--- Exp 10: Singular Value Spectrum (SVD Decay) ---")
sample_std = std_emb[:10000]
sample_mrl = mrl_emb[:10000]

u_s, s_std, vh_s = np.linalg.svd(sample_std - np.mean(sample_std, axis=0), full_matrices=False)
u_m, s_mrl, vh_m = np.linalg.svd(sample_mrl - np.mean(sample_mrl, axis=0), full_matrices=False)

s_std_norm = s_std / np.sum(s_std)
s_mrl_norm = s_mrl / np.sum(s_mrl)

plt.figure(figsize=(10,6))
plt.plot(np.arange(1, 301), s_std_norm, label="Standard", alpha=0.8)
plt.plot(np.arange(1, 301), s_mrl_norm, label="MRL-v4", alpha=0.8)
plt.title("MRL-v4 Singular Value Spectrum Decay (Normalized)")
plt.xlabel("Singular Value Rank (1-300)")
plt.ylabel("Relative Singular Value Magnitude")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp10_singular_value_spectrum_v4.png"))
plt.close()

print("\n--- Summary Snippets ---")
print(f"Top 10 SVs % of Total | Std: {np.sum(s_std_norm[:10])*100:.2f}%, MRL-v4: {np.sum(s_mrl_norm[:10])*100:.2f}%")

print("\nV4 Geometric Analysis Complete! Plots saved to:", plot_dir)
with open(os.path.join(plot_dir, "geometry_results_v4.txt"), "a") as f:
    f.write(f"Top 10 SVs % of Total | Std: {np.sum(s_std_norm[:10])*100:.2f}%, MRL-v4: {np.sum(s_mrl_norm[:10])*100:.2f}%\n")

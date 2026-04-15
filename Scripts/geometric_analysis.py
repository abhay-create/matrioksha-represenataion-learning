import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
import matplotlib
import random

matplotlib.use('Agg') # for non-interactive plotting

# Paths
base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
emb_dir = os.path.join(base_dir, "Data_and_Embeddings", "_v3_embeddings_W2V")
plot_dir = os.path.join(base_dir, "Results_and_Reports", "geometry_plots")
os.makedirs(plot_dir, exist_ok=True)

print("Loading embeddings and vocabulary...")
std_emb = np.load(os.path.join(emb_dir, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(emb_dir, "mrl_w2v_embeddings.npy"))

print(f"Standard shape: {std_emb.shape}, MRL shape: {mrl_emb.shape}")
V = std_emb.shape[0]

# --- Exp 1: Mean Embedding & Nested Level Scaled Dot product ---
print("\n--- Exp 1: Mean Embedding Dot Product ---")
levels = [50, 100, 150, 200, 250, 300]
mean_std_full = np.mean(std_emb, axis=0)  # shape (300,)
mean_mrl_full = np.mean(mrl_emb, axis=0)  # shape (300,)

exp1_sims = []
for d in levels:
    vec_s = mean_std_full[:d]
    vec_m = mean_mrl_full[:d]
    num = np.dot(vec_s, vec_m)
    den = (np.linalg.norm(vec_s) * np.linalg.norm(vec_m)) + 1e-10
    sim = num / den
    exp1_sims.append(sim)
    print(f"Level {d:<3}: Cosine Similarity = {sim:.4f}")

plt.figure(figsize=(8,5))
plt.plot(levels, exp1_sims, marker='o', color='purple')
plt.title("Cosine Sim between Mean Embeddings (Standard vs MRL) across Prefixes")
plt.xlabel("Prefix Dimension")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp1_mean_sim.png"))
plt.close()

# --- Exp 2: Cross Dot Product ---
print("\n--- Exp 2: Cross Dot Product scaled by Vocab Size ---")
std_on_mrl_mean = []
mrl_on_std_mean = []

def cossim_matrix(A, B):
    nums = np.dot(A, B)
    dens = np.linalg.norm(A, axis=1) * np.linalg.norm(B) + 1e-10
    return nums / dens

for d in levels:
    std_d = std_emb[:, :d]
    mrl_d = mrl_emb[:, :d]
    mean_std_d = mean_std_full[:d]
    mean_mrl_d = mean_mrl_full[:d]
    
    cs1 = cossim_matrix(std_d, mean_mrl_d)
    cs2 = cossim_matrix(mrl_d, mean_std_d)
    std_on_mrl_mean.append(np.mean(cs1))
    mrl_on_std_mean.append(np.mean(cs2))
    print(f"Level {d:<3}: Avg Cos Sim (Std_w, Mrl_mean) = {np.mean(cs1):.4f}, (Mrl_w, Std_mean) = {np.mean(cs2):.4f}")

plt.figure(figsize=(8,5))
plt.plot(levels, std_on_mrl_mean, marker='o', label="Std words with MRL mean")
plt.plot(levels, mrl_on_std_mean, marker='s', label="MRL words with Std mean")
plt.title("Avg Cross Cosine Similarity across Prefixes")
plt.xlabel("Prefix Dimension")
plt.ylabel("Mean Cosine Similarity")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp2_cross_sim.png"))
plt.close()

# --- Exp 3: PCA Explained Variance for Full Vocabulary ---
print("\n--- Exp 3: PCA on Random Sample of Full Vocabulary ---")
np.random.seed(42)
sample_idx = np.random.choice(V, min(10000, V), replace=False)
std_sample = std_emb[sample_idx]
mrl_sample = mrl_emb[sample_idx]

pca_std = PCA(n_components=50).fit(std_sample)
pca_mrl = PCA(n_components=50).fit(mrl_sample)

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_std.explained_variance_ratio_), label="Standard 300d", color='blue')
plt.plot(np.cumsum(pca_mrl.explained_variance_ratio_), label="MRL 300d", color='orange')
plt.title("Cumulative Explained Variance (Top 50 PCs) - Full Vocab")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp3_pca_full.png"))
plt.close()
print(f"Top 50 PCs EVR -> Std: {np.sum(pca_std.explained_variance_ratio_):.4f}, MRL: {np.sum(pca_mrl.explained_variance_ratio_):.4f}")

# --- Exp 4: PCA for specific frequency buckets ---
print("\n--- Exp 4: PCA by Frequency Bucket ---")
buckets = {"Top 100": (0, 100), "Top 101-500": (100, 500), "Top 501-5000": (500, 5000)}

fig, axs = plt.subplots(1, 3, figsize=(18,5))
for i, (b_name, (start, end)) in enumerate(buckets.items()):
    std_b = std_emb[start:end]
    mrl_b = mrl_emb[start:end]
    
    n_comp = min(50, end-start)
    pca_std = PCA(n_components=n_comp).fit(std_b)
    pca_mrl = PCA(n_components=n_comp).fit(mrl_b)
    
    axs[i].plot(np.cumsum(pca_std.explained_variance_ratio_), label="Standard 300d", color='blue')
    axs[i].plot(np.cumsum(pca_mrl.explained_variance_ratio_), label="MRL 300d", color='orange')
    axs[i].set_title(b_name)
    axs[i].set_xlabel("Principal Component")
    axs[i].set_ylabel("Cum. Explained Variance EVR")
    axs[i].legend()
    axs[i].grid(True)
    
    print(f"Bucket {b_name} Top {n_comp} PCs EVR -> Std: {np.sum(pca_std.explained_variance_ratio_):.4f}, MRL: {np.sum(pca_mrl.explained_variance_ratio_):.4f}")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "exp4_pca_buckets.png"))
plt.close()

# --- Exp 5: Angle between consecutive 50 dimensions of mean vectors ---
print("\n--- Exp 5: Angle between Consecutive 50 Dimensions (Mean Vector) ---")
pairs = [(0,50), (50,100), (100,150), (150,200), (200,250), (250,300)]
std_angles = []
mrl_angles = []

for i in range(len(pairs)-1):
    s1, e1 = pairs[i]
    s2, e2 = pairs[i+1]
    
    v1_std = mean_std_full[s1:e1]
    v2_std = mean_std_full[s2:e2]
    num_std = np.dot(v1_std, v2_std)
    sim_std = num_std / (np.linalg.norm(v1_std)*np.linalg.norm(v2_std) + 1e-10)
    std_angles.append(sim_std)
    
    v1_mrl = mean_mrl_full[s1:e1]
    v2_mrl = mean_mrl_full[s2:e2]
    num_mrl = np.dot(v1_mrl, v2_mrl)
    sim_mrl = num_mrl / (np.linalg.norm(v1_mrl)*np.linalg.norm(v2_mrl) + 1e-10)
    mrl_angles.append(sim_mrl)
    
    print(f"Pair {s1:03}-{e1:03} vs {s2:03}-{e2:03} -> Cos Sim | Std: {sim_std:6.4f}, MRL: {sim_mrl:6.4f}")

labels = [f"{pairs[i][0]}-{pairs[i][1]} vs {pairs[i+1][0]}-{pairs[i+1][1]}" for i in range(len(pairs)-1)]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10,5))
rects1 = ax.bar(x - width/2, std_angles, width, label='Standard', color='blue')
rects2 = ax.bar(x + width/2, mrl_angles, width, label='MRL', color='orange')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Cosine Similarity between Consecutive 50d Chunks of Mean Vector')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "exp5_consecutive_angles.png"))
plt.close()

# --- Exp 6: Mimno & Thompson 2017 - Cone Geometry & Non-negativity ---
print("\n--- Exp 6: Mimno & Thompson (2017) Experiments ---")
std_sims_300 = cossim_matrix(std_emb, mean_std_full)
mrl_sims_300 = cossim_matrix(mrl_emb, mean_mrl_full)
mrl_sims_50 = cossim_matrix(mrl_emb[:,:50], mean_mrl_full[:50])
std_sims_50 = cossim_matrix(std_emb[:,:50], mean_std_full[:50])

print(f"Avg Cos Sim to Mean | Std300: {np.mean(std_sims_300):.4f}, Std50: {np.mean(std_sims_50):.4f}, MRL300: {np.mean(mrl_sims_300):.4f}, MRL50: {np.mean(mrl_sims_50):.4f}")

plt.figure(figsize=(10,6))
plt.hist(std_sims_300, bins=50, alpha=0.5, label='Std 300d', density=True)
plt.hist(mrl_sims_300, bins=50, alpha=0.5, label='MRL 300d', density=True)
plt.hist(mrl_sims_50, bins=50, alpha=0.5, label='MRL 50d', density=True)
plt.title("Distribution of Cosine Similarity to Mean Vector (The Narrow Cone)")
plt.xlabel("Cosine Similarity to Mean")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp6a_cone_alignment.png"))
plt.close()

std_pos_pct = np.mean(std_emb > 0, axis=0)
mrl_pos_pct = np.mean(mrl_emb > 0, axis=0)

print(f"Avg % Positive Components overall | Std: {np.mean(std_pos_pct)*100:.2f}%, MRL: {np.mean(mrl_pos_pct)*100:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(np.arange(1, 301), std_pos_pct, label="Standard", alpha=0.7)
plt.plot(np.arange(1, 301), mrl_pos_pct, label="MRL", alpha=0.7)
plt.axhline(y=0.5, color='r', linestyle='--', label="Expected (0.5 for mean-0)")
plt.title("Percentage of Positive Values per Dimension")
plt.xlabel("Dimension Index (1-300)")
plt.ylabel("Fraction > 0")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "exp6b_non_negativity.png"))
plt.close()

# --- Exp 7: Average L2 Norm by Frequency across dimensions ---
print("\n--- Exp 7: L2 Norm Tracking by Frequency Bucket ---")
norm_std_res = {k: [] for k in buckets.keys()}
norm_mrl_res = {k: [] for k in buckets.keys()}

for d in levels:
    std_d = std_emb[:, :d]
    mrl_d = mrl_emb[:, :d]
    
    for b_name, (start, end) in buckets.items():
        avg_norm_std = np.mean(np.linalg.norm(std_d[start:end], axis=1))
        avg_norm_mrl = np.mean(np.linalg.norm(mrl_d[start:end], axis=1))
        
        norm_std_res[b_name].append(avg_norm_std)
        norm_mrl_res[b_name].append(avg_norm_mrl)
    
    print(f"Level {d:<3} Avg L2 | Top 100 Std: {norm_std_res['Top 100'][-1]:.2f}, MRL: {norm_mrl_res['Top 100'][-1]:.2f}")

fig, axs = plt.subplots(1, 4, figsize=(20,5))

for i, b_name in enumerate(buckets.keys()):
    axs[i].plot(levels, norm_std_res[b_name], marker='o', label="Standard")
    axs[i].plot(levels, norm_mrl_res[b_name], marker='s', label="MRL")
    axs[i].set_title(b_name)
    axs[i].set_xlabel("Prefix Dimension")
    axs[i].set_ylabel("Average L2 Norm")
    axs[i].legend()
    axs[i].grid(True)

avg_norm_std_full = []
avg_norm_mrl_full = []
for d in levels:
    std_d = std_emb[:, :d]
    mrl_d = mrl_emb[:, :d]
    avg_norm_std_full.append(np.mean(np.linalg.norm(std_d, axis=1)))
    avg_norm_mrl_full.append(np.mean(np.linalg.norm(mrl_d, axis=1)))

axs[3].plot(levels, avg_norm_std_full, marker='o', label="Standard")
axs[3].plot(levels, avg_norm_mrl_full, marker='s', label="MRL")
axs[3].set_title("Full Vocabulary")
axs[3].set_xlabel("Prefix Dimension")
axs[3].set_ylabel("Average L2 Norm")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "exp7_l2_norm_freq.png"))
plt.close()

print("\nAnalysis Complete! Plots saved to:", plot_dir)

"""
Full Geometry + Bias Evaluation for 768d MRL-v4 vs Standard Word2Vec
====================================================================
Nesting levels: 64, 128, 256, 384, 512, 768
"""
import numpy as np
import os, json, pickle, sys
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE     = r"c:\Users\abhay\Desktop\MRL TESTS"
EMB_DIR  = os.path.join(BASE, "latest")
PLOT_DIR = os.path.join(EMB_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print("Loading vocab...")
try:
    with open(os.path.join(EMB_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    w2i = {w: i for i, w in enumerate(vocab.i2w)}
except Exception:
    vw = np.load(os.path.join(EMB_DIR, "vocab_words.npy"))
    w2i = {str(w): i for i, w in enumerate(vw)}

print("Loading embeddings...")
std_emb = np.load(os.path.join(EMB_DIR, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(EMB_DIR, "mrl_v4_w2v_embeddings.npy"))
V, D = std_emb.shape
print(f"Loaded {V} words, {D} dims")

LEVELS = [64, 128, 256, 384, 512, 768]

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────
def cossim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def cossim_matrix(A, b):
    nums = np.dot(A, b)
    dens = np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-10
    return nums / dens

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)

# ═══════════════════════════════════════════════════════════
#  PART A: GEOMETRY TESTS
# ═══════════════════════════════════════════════════════════
log("=" * 70)
log("GEOMETRY EVALUATION - 768d Embeddings")
log("=" * 70)

mean_std = np.mean(std_emb, axis=0)
mean_mrl = np.mean(mrl_emb, axis=0)

# --- Exp 1: Mean Embedding Cosine Sim ---
log("\n--- Exp 1: Mean Embedding Cosine Similarity ---")
exp1 = []
for d in LEVELS:
    s = cossim(mean_std[:d], mean_mrl[:d])
    exp1.append(s)
    log(f"  d={d:>3}: cos(mean_std, mean_mrl) = {s:.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, exp1, marker='o', color='purple')
plt.title("Cosine Sim between Mean Embeddings (Standard vs MRL-v4)")
plt.xlabel("Prefix Dimension"); plt.ylabel("Cosine Similarity"); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp1_mean_sim.png")); plt.close()

# --- Exp 2: Cross Cosine Similarity ---
log("\n--- Exp 2: Cross Cosine Similarity ---")
exp2_s, exp2_m = [], []
for d in LEVELS:
    cs1 = np.mean(cossim_matrix(std_emb[:, :d], mean_mrl[:d]))
    cs2 = np.mean(cossim_matrix(mrl_emb[:, :d], mean_std[:d]))
    exp2_s.append(cs1); exp2_m.append(cs2)
    log(f"  d={d:>3}: Avg(Std_w, Mrl_mean)={cs1:.4f}, Avg(Mrl_w, Std_mean)={cs2:.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, exp2_s, marker='o', label="Std words w/ MRL mean")
plt.plot(LEVELS, exp2_m, marker='s', label="MRL words w/ Std mean")
plt.title("Cross Cosine Similarity"); plt.xlabel("Prefix Dim"); plt.ylabel("Mean Cosine Sim")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp2_cross_sim.png")); plt.close()

# --- Exp 3: PCA Full Vocab ---
log("\n--- Exp 3: PCA (Full Vocab Sample) ---")
np.random.seed(42)
sidx = np.random.choice(V, min(10000, V), replace=False)
pca_s = PCA(n_components=50).fit(std_emb[sidx])
pca_m = PCA(n_components=50).fit(mrl_emb[sidx])
log(f"  Top 50 PCs EVR -> Std: {np.sum(pca_s.explained_variance_ratio_):.4f}, MRL: {np.sum(pca_m.explained_variance_ratio_):.4f}")

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_s.explained_variance_ratio_), label="Standard 768d")
plt.plot(np.cumsum(pca_m.explained_variance_ratio_), label="MRL-v4 768d")
plt.title("Cumulative Explained Variance (Top 50 PCs)"); plt.xlabel("PC"); plt.ylabel("Cum. EVR")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp3_pca_full.png")); plt.close()

# --- Exp 4: PCA by Frequency Bucket ---
log("\n--- Exp 4: PCA by Frequency Bucket ---")
buckets = {"Top_100": (0,100), "Top_101-500": (100,500), "Top_501-5000": (500,5000)}
fig, axs = plt.subplots(1, 3, figsize=(18,5))
for i, (bn, (s, e)) in enumerate(buckets.items()):
    nc = min(50, e-s)
    ps = PCA(n_components=nc).fit(std_emb[s:e])
    pm = PCA(n_components=nc).fit(mrl_emb[s:e])
    axs[i].plot(np.cumsum(ps.explained_variance_ratio_), label="Standard")
    axs[i].plot(np.cumsum(pm.explained_variance_ratio_), label="MRL-v4")
    axs[i].set_title(bn); axs[i].legend(); axs[i].grid(True)
    log(f"  {bn} Top {nc} PCs EVR -> Std: {np.sum(ps.explained_variance_ratio_):.4f}, MRL: {np.sum(pm.explained_variance_ratio_):.4f}")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "exp4_pca_buckets.png")); plt.close()

# --- Exp 5: Consecutive 64d chunk angles ---
log("\n--- Exp 5: Consecutive Chunk Angles ---")
chunk_size = 64
n_chunks = D // chunk_size
chunks = [(i*chunk_size, (i+1)*chunk_size) for i in range(n_chunks)]
std_ang, mrl_ang = [], []
for i in range(len(chunks)-1):
    s1, e1 = chunks[i]; s2, e2 = chunks[i+1]
    sa = cossim(mean_std[s1:e1], mean_std[s2:e2])
    ma = cossim(mean_mrl[s1:e1], mean_mrl[s2:e2])
    std_ang.append(sa); mrl_ang.append(ma)
    log(f"  Chunk {s1}-{e1} vs {s2}-{e2}: Std={sa:.4f}, MRL={ma:.4f}")

labels = [f"{chunks[i][0]}-{chunks[i][1]}" for i in range(len(chunks)-1)]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(x - 0.18, std_ang, 0.35, label='Standard')
ax.bar(x + 0.18, mrl_ang, 0.35, label='MRL-v4')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_ylabel('Cosine Similarity'); ax.set_title('Consecutive Chunk Angles'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "exp5_consecutive_angles.png")); plt.close()

# --- Exp 6a: Narrow Cone ---
log("\n--- Exp 6a: Narrow Cone (Mimno & Thompson) ---")
for d in LEVELS:
    ss = np.mean(cossim_matrix(std_emb[:,:d], mean_std[:d]))
    ms = np.mean(cossim_matrix(mrl_emb[:,:d], mean_mrl[:d]))
    log(f"  d={d:>3}: Std cone={ss:.4f}, MRL cone={ms:.4f}")

plt.figure(figsize=(10,6))
plt.hist(cossim_matrix(std_emb, mean_std), bins=50, alpha=0.5, label='Std 768d', density=True)
plt.hist(cossim_matrix(mrl_emb, mean_mrl), bins=50, alpha=0.5, label='MRL 768d', density=True)
plt.hist(cossim_matrix(mrl_emb[:,:64], mean_mrl[:64]), bins=50, alpha=0.5, label='MRL 64d', density=True)
plt.title("Narrow Cone Distribution"); plt.xlabel("Cosine Sim to Mean"); plt.ylabel("Density")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp6a_cone.png")); plt.close()

# --- Exp 6b: Non-negativity ---
log("\n--- Exp 6b: Non-Negativity ---")
sp = np.mean(std_emb > 0, axis=0)
mp = np.mean(mrl_emb > 0, axis=0)
log(f"  Avg % positive: Std={np.mean(sp)*100:.2f}%, MRL={np.mean(mp)*100:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(np.arange(1, D+1), sp, label="Standard", alpha=0.7)
plt.plot(np.arange(1, D+1), mp, label="MRL-v4", alpha=0.7)
plt.axhline(y=0.5, color='r', linestyle='--', label="Expected 0.5")
plt.title("Non-Negativity per Dimension"); plt.xlabel("Dim Index"); plt.ylabel("Frac > 0")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp6b_nonneg.png")); plt.close()

# --- Exp 7: L2 Norm ---
log("\n--- Exp 7: L2 Norm by Frequency Bucket ---")
all_buckets = {**buckets, "Full_10k": (0, min(10000, V))}
fig, axs = plt.subplots(1, 4, figsize=(20,5))
for i, (bn, (s, e)) in enumerate(all_buckets.items()):
    ns, nm = [], []
    for d in LEVELS:
        ns.append(np.mean(np.linalg.norm(std_emb[s:e, :d], axis=1)))
        nm.append(np.mean(np.linalg.norm(mrl_emb[s:e, :d], axis=1)))
    axs[i].plot(LEVELS, ns, marker='o', label="Standard")
    axs[i].plot(LEVELS, nm, marker='s', label="MRL-v4")
    axs[i].set_title(bn); axs[i].set_xlabel("Prefix Dim"); axs[i].set_ylabel("Avg L2 Norm")
    axs[i].legend(); axs[i].grid(True)
    log(f"  {bn} d=64: Std={ns[0]:.4f} MRL={nm[0]:.4f} | d=768: Std={ns[-1]:.4f} MRL={nm[-1]:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "exp7_l2_norm.png")); plt.close()

# --- Exp 8: WCSS ---
log("\n--- Exp 8: WCSS (K-Means Inertia) ---")
fig, axs = plt.subplots(2, 2, figsize=(16,10))
axs = axs.flatten()
for i, (bn, (s, e)) in enumerate(all_buckets.items()):
    ws, wm = [], []
    nc = 10 if (e-s) < 500 else 50
    for d in LEVELS:
        sd = std_emb[s:e, :d]; md = mrl_emb[s:e, :d]
        sn = sd / (np.linalg.norm(sd, axis=1, keepdims=True)+1e-10)
        mn = md / (np.linalg.norm(md, axis=1, keepdims=True)+1e-10)
        ws.append(KMeans(n_clusters=nc, n_init=5, random_state=42).fit(sn).inertia_ / (e-s))
        wm.append(KMeans(n_clusters=nc, n_init=5, random_state=42).fit(mn).inertia_ / (e-s))
    axs[i].plot(LEVELS, ws, marker='o', label="Standard")
    axs[i].plot(LEVELS, wm, marker='s', label="MRL-v4")
    axs[i].set_title(f"WCSS - {bn} (K={nc})"); axs[i].set_xlabel("Prefix Dim"); axs[i].set_ylabel("Inertia/N")
    axs[i].legend(); axs[i].grid(True)
    log(f"  WCSS {bn} d=64: Std={ws[0]:.4f} MRL={wm[0]:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "exp8_wcss.png")); plt.close()

# --- Exp 9: Jaccard Neighborhood Preservation ---
log("\n--- Exp 9: Jaccard Neighborhood Preservation ---")
K_nb = 10
np.random.seed(42)
queries = np.random.choice(V, 1000, replace=False)

def get_topk(emb_slice, qidx, k=10):
    q = emb_slice[qidx]
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True)+1e-10)
    en = emb_slice / (np.linalg.norm(emb_slice, axis=1, keepdims=True)+1e-10)
    sims = qn @ en.T
    topk = np.argsort(sims, axis=1)[:, -(k+1):]
    sets = []
    for i, row in enumerate(topk):
        s = set(row); s.discard(qidx[i]); sets.append(s)
    return sets

std_base = get_topk(std_emb, queries, K_nb)
mrl_base = get_topk(mrl_emb, queries, K_nb)
j_std, j_mrl = [], []
for d in LEVELS:
    sn = get_topk(std_emb[:,:d], queries, K_nb)
    mn = get_topk(mrl_emb[:,:d], queries, K_nb)
    js = np.mean([len(std_base[i]&sn[i])/len(std_base[i]|sn[i]) for i in range(1000)])
    jm = np.mean([len(mrl_base[i]&mn[i])/len(mrl_base[i]|mn[i]) for i in range(1000)])
    j_std.append(js); j_mrl.append(jm)
    log(f"  d={d:>3}: Std Jaccard={js:.4f}, MRL Jaccard={jm:.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, j_std, marker='o', label="Standard")
plt.plot(LEVELS, j_mrl, marker='s', label="MRL-v4")
plt.title("Neighborhood Preservation (Jaccard vs d=768)"); plt.xlabel("Prefix Dim"); plt.ylabel("Avg Jaccard")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp9_jaccard.png")); plt.close()

# --- Exp 10: SVD Spectrum ---
log("\n--- Exp 10: SVD Spectrum ---")
ss = std_emb[:10000]; ms = mrl_emb[:10000]
_, sv_s, _ = np.linalg.svd(ss - np.mean(ss, axis=0), full_matrices=False)
_, sv_m, _ = np.linalg.svd(ms - np.mean(ms, axis=0), full_matrices=False)
sv_sn = sv_s / np.sum(sv_s); sv_mn = sv_m / np.sum(sv_m)
log(f"  Top 10 SVs: Std={np.sum(sv_sn[:10])*100:.2f}%, MRL={np.sum(sv_mn[:10])*100:.2f}%")

plt.figure(figsize=(10,6))
plt.plot(np.arange(1, D+1), sv_sn, label="Standard", alpha=0.8)
plt.plot(np.arange(1, D+1), sv_mn, label="MRL-v4", alpha=0.8)
plt.title("SVD Spectrum Decay"); plt.xlabel("SV Rank"); plt.ylabel("Relative SV Magnitude")
plt.xscale('log'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "exp10_svd.png")); plt.close()


# ═══════════════════════════════════════════════════════════
#  PART B: BIAS TESTS
# ═══════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("BIAS EVALUATION - 768d Embeddings")
log("=" * 70)

DEFINITIONAL_PAIRS = [
    ("he","she"),("man","woman"),("boy","girl"),("father","mother"),
    ("son","daughter"),("husband","wife"),("brother","sister"),("king","queen"),
]
MALE_A = [w for w in ["john","paul","mike","kevin","steve","greg","jeff","bill"] if w in w2i]
FEMALE_A = [w for w in ["amy","joan","lisa","sarah","diana","kate","ann","donna"] if w in w2i]
CAREER = [w for w in ["executive","management","professional","corporation","salary","office","business","career"] if w in w2i]
FAMILY = [w for w in ["home","parents","children","family","cousins","marriage","wedding","relatives"] if w in w2i]
NEUTRAL = [w for w in ["doctor","nurse","engineer","teacher","programmer","librarian","soldier",
    "receptionist","housekeeper","carpenter","mechanic","pilot","accountant","plumber",
    "professor","chef","scientist","artist","manager","secretary"] if w in w2i]
ALL_T = [w for w in (NEUTRAL + CAREER + FAMILY) if w in w2i]

log(f"  In-vocab: Male={len(MALE_A)} Female={len(FEMALE_A)} Career={len(CAREER)} Family={len(FAMILY)} Neutral={len(NEUTRAL)}")

def get_vecs(wl, emb, d):
    return emb[[w2i[w] for w in wl], :d]

def gender_dir(emb, d):
    diffs = []
    for m, f in DEFINITIONAL_PAIRS:
        if m in w2i and f in w2i:
            diffs.append(emb[w2i[m], :d] - emb[w2i[f], :d])
    if len(diffs) < 2: return np.zeros(d)
    return PCA(n_components=1).fit(np.array(diffs)).components_[0]

def weat(emb, d):
    X = get_vecs(CAREER, emb, d); Y = get_vecs(FAMILY, emb, d)
    A = get_vecs(MALE_A, emb, d); B = get_vecs(FEMALE_A, emb, d)
    def s(w):
        return np.mean([cossim(w,a) for a in A]) - np.mean([cossim(w,b) for b in B])
    sx = [s(x) for x in X]; sy = [s(y) for y in Y]
    return (np.mean(sx)-np.mean(sy)) / (np.std(sx+sy)+1e-10)

def direct_bias(emb, d):
    g = gender_dir(emb, d)
    if np.linalg.norm(g) < 1e-10: return 0.0
    return float(np.mean([abs(cossim(emb[w2i[w],:d], g)) for w in NEUTRAL]))

def ripa(emb, d):
    g = gender_dir(emb, d)
    if np.linalg.norm(g) < 1e-10: return 0.0, 0.0
    scores = [abs(float(np.dot(emb[w2i[w],:d], g))) for w in NEUTRAL]
    return float(np.mean(scores)), float(np.max(scores))

def ect(emb, d):
    mu_m = np.mean(get_vecs(MALE_A, emb, d), axis=0)
    mu_f = np.mean(get_vecs(FEMALE_A, emb, d), axis=0)
    dm = [cossim(emb[w2i[w],:d], mu_m) for w in ALL_T]
    df = [cossim(emb[w2i[w],:d], mu_f) for w in ALL_T]
    if len(dm) < 3: return 0.0
    return float(spearmanr(dm, df)[0])

def nbm(emb, d, k=100):
    ed = emb[:,:d].copy()
    en = ed / (np.linalg.norm(ed, axis=1, keepdims=True)+1e-10)
    mi = set(w2i[w] for w in MALE_A if w in w2i)
    fi = set(w2i[w] for w in FEMALE_A if w in w2i)
    scores = []
    for w in NEUTRAL:
        if w not in w2i: continue
        sims = en @ en[w2i[w]]
        topk = np.argsort(sims)[-(k+1):-1]
        cm = sum(1 for n in topk if n in mi)
        cf = sum(1 for n in topk if n in fi)
        scores.append(abs(cm-cf)/(cm+cf) if (cm+cf)>0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def cluster_purity(emb, d):
    words = MALE_A + FEMALE_A
    labels = [0]*len(MALE_A) + [1]*len(FEMALE_A)
    vecs = get_vecs(words, emb, d)
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True)+1e-10)
    pred = KMeans(n_clusters=2, n_init=10, random_state=42).fit(vn).labels_
    m0 = sum(1 for p,t in zip(pred,labels) if p==t)
    m1 = sum(1 for p,t in zip(pred,labels) if (1-p)==t)
    return max(m0,m1)/len(labels)

def bias_analogy(emb, d):
    if "he" not in w2i or "she" not in w2i: return []
    offset = emb[w2i["she"],:d] - emb[w2i["he"],:d]
    i2w = {i:w for w,i in w2i.items()}
    ed = emb[:,:d]; en = ed/(np.linalg.norm(ed,axis=1,keepdims=True)+1e-10)
    results = []
    for w in NEUTRAL:
        if w not in w2i: continue
        q = emb[w2i[w],:d] + offset
        qn = q/(np.linalg.norm(q)+1e-10)
        sims = en @ qn
        for ex in [w2i["he"], w2i["she"], w2i[w]]: sims[ex] = -999
        bi = int(np.argmax(sims))
        results.append((w, i2w.get(bi,"?"), float(sims[bi])))
    return results

bias_results = {"standard": {}, "mrl_v4": {}}
for mname, emb in [("standard", std_emb), ("mrl_v4", mrl_emb)]:
    log(f"\n{'='*50}\nMODEL: {mname.upper()}\n{'='*50}")
    for d in LEVELS:
        log(f"\n--- Dimension {d} ---")
        r = {}
        r["weat"] = float(weat(emb, d)); log(f"  WEAT:         {r['weat']:.4f}")
        r["direct_bias"] = float(direct_bias(emb, d)); log(f"  DirectBias:   {r['direct_bias']:.4f}")
        rm, rx = ripa(emb, d); r["ripa_mean"]=rm; r["ripa_max"]=rx; log(f"  RIPA m|x:     {rm:.4f} | {rx:.4f}")
        r["ect"] = float(ect(emb, d)); log(f"  ECT:          {r['ect']:.4f}")
        r["nbm"] = float(nbm(emb, d)); log(f"  NBM:          {r['nbm']:.4f}")
        r["cluster_purity"] = float(cluster_purity(emb, d)); log(f"  ClusterPurity:{r['cluster_purity']:.4f}")
        if d == D:
            ar = bias_analogy(emb, d)
            r["analogies"] = [(w,r2,round(s,4)) for w,r2,s in ar]
            log(f"  Analogies (he:she :: word:?):")
            for w,r2,s in ar: log(f"    {w:15} -> {r2:15} (sim={s:.4f})")
        else:
            r["analogies"] = []
        bias_results[mname][str(d)] = r

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
with open(os.path.join(EMB_DIR, "bias_results.json"), "w") as f:
    json.dump(bias_results, f, indent=2, default=str)
with open(os.path.join(EMB_DIR, "full_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))

# ─────────────────────────────────────────────
# BIAS PLOTS
# ─────────────────────────────────────────────
metrics = [
    ("weat", "WEAT Effect Size", "Effect Size (d)"),
    ("direct_bias", "DirectBias", "DirectBias Score"),
    ("ripa_mean", "RIPA Mean", "Mean |RIPA|"),
    ("ripa_max", "RIPA Max", "Max |RIPA|"),
    ("ect", "ECT Spearman rho", "Spearman rho"),
    ("nbm", "NBM (k=100)", "Mean |NBM|"),
    ("cluster_purity", "Cluster Purity (K=2)", "Purity"),
]

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()
for i, (key, title, yl) in enumerate(metrics):
    sv = [bias_results["standard"][str(d)][key] for d in LEVELS]
    mv = [bias_results["mrl_v4"][str(d)][key] for d in LEVELS]
    axes[i].plot(LEVELS, sv, marker='o', label="Standard", linewidth=2)
    axes[i].plot(LEVELS, mv, marker='s', label="MRL-v4", linewidth=2)
    axes[i].set_title(title, fontweight='bold'); axes[i].set_xlabel("Prefix Dim"); axes[i].set_ylabel(yl)
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
for j in range(len(metrics), len(axes)): axes[j].set_visible(False)
plt.suptitle("Bias Metrics: MRL-v4 vs Standard (768d)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(PLOT_DIR, "bias_dashboard.png"), dpi=150); plt.close()

for key, title, yl in metrics:
    plt.figure(figsize=(8,5))
    sv = [bias_results["standard"][str(d)][key] for d in LEVELS]
    mv = [bias_results["mrl_v4"][str(d)][key] for d in LEVELS]
    plt.plot(LEVELS, sv, marker='o', label="Standard", linewidth=2)
    plt.plot(LEVELS, mv, marker='s', label="MRL-v4", linewidth=2)
    plt.title(title); plt.xlabel("Prefix Dim"); plt.ylabel(yl)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, f"bias_{key}.png")); plt.close()

log("\n=== ALL EVALUATIONS COMPLETE ===")
log(f"Plots saved to: {PLOT_DIR}")
log(f"JSON saved to: {os.path.join(EMB_DIR, 'bias_results.json')}")
log(f"Text saved to: {os.path.join(EMB_DIR, 'full_results.txt')}")

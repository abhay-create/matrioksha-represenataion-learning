"""
Deep-Dive Bias Investigation: MRL-v4 vs Standard Word2Vec (300d)
================================================================
Runs ~15 experiments probing how the MRL training objective
affects social bias encoding. Saves all plots and a detailed
text log for analysis.
"""
import numpy as np
import os, json, pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = r"c:\Users\abhay\Desktop\MRL TESTS"
EMB_DIR = os.path.join(BASE, "Data_and_Embeddings", "_v3_embeddings_W2V",
                       "Embeddings", "mrl_bias_v4_embeddings")
PLOT_DIR = os.path.join(EMB_DIR, "deep_bias_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading...")
try:
    with open(os.path.join(EMB_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    w2i = {w: i for i, w in enumerate(vocab.i2w)}
    i2w = {i: w for w, i in w2i.items()}
except:
    vw = np.load(os.path.join(EMB_DIR, "vocab_words.npy"))
    w2i = {str(w): i for i, w in enumerate(vw)}
    i2w = {i: str(w) for i, w in enumerate(vw)}

std_emb = np.load(os.path.join(EMB_DIR, "standard_w2v_embeddings.npy"))
mrl_emb = np.load(os.path.join(EMB_DIR, "mrl_v4_w2v_embeddings.npy"))
V, D = std_emb.shape
print(f"Loaded {V} words, {D} dims")
LEVELS = [50, 100, 150, 200, 250, 300]

# ─────────────────────────────────────────
# WORD LISTS
# ─────────────────────────────────────────
DEFP = [("he","she"),("man","woman"),("boy","girl"),("father","mother"),
        ("son","daughter"),("husband","wife"),("brother","sister"),("king","queen")]

MALE_NAMES = [w for w in ["john","paul","mike","kevin","steve","greg","jeff","bill"] if w in w2i]
FEMALE_NAMES = [w for w in ["amy","joan","lisa","sarah","diana","kate","ann","donna"] if w in w2i]
MALE_PRONOUNS = [w for w in ["he","him","his","himself","man","boy","father","son","brother","husband","king"] if w in w2i]
FEMALE_PRONOUNS = [w for w in ["she","her","hers","herself","woman","girl","mother","daughter","sister","wife","queen"] if w in w2i]

CAREER = [w for w in ["executive","management","professional","corporation","salary","office","business","career"] if w in w2i]
FAMILY = [w for w in ["home","parents","children","family","cousins","marriage","wedding","relatives"] if w in w2i]
MATH = [w for w in ["math","algebra","geometry","calculus","equations","computation","numbers","addition"] if w in w2i]
ARTS = [w for w in ["poetry","art","dance","literature","novel","symphony","drama","sculpture"] if w in w2i]
SCIENCE = [w for w in ["science","technology","physics","chemistry","experiment","astronomy"] if w in w2i]

OCCUPATIONS = [w for w in [
    "doctor","nurse","engineer","teacher","programmer","librarian","soldier",
    "receptionist","housekeeper","carpenter","mechanic","pilot","accountant",
    "plumber","professor","chef","scientist","artist","manager","secretary",
    "surgeon","therapist","architect","electrician","janitor","pharmacist",
    "dentist","lawyer","judge","firefighter","paramedic","technician",
    "analyst","consultant","designer","writer","editor","journalist",
    "photographer","musician","painter","dancer","actor","singer"
] if w in w2i]

STEREOTYP_MALE = [w for w in ["engineer","programmer","mechanic","carpenter","plumber","electrician","surgeon","pilot","soldier","architect"] if w in w2i]
STEREOTYP_FEMALE = [w for w in ["nurse","receptionist","housekeeper","secretary","librarian","teacher","therapist","dancer","singer","designer"] if w in w2i]

ALL_GENDERED = set(MALE_NAMES + FEMALE_NAMES + MALE_PRONOUNS + FEMALE_PRONOUNS)
print(f"Occupations: {len(OCCUPATIONS)}, StereotypM: {len(STEREOTYP_MALE)}, StereotypF: {len(STEREOTYP_FEMALE)}")

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)

def cossim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def gdir(emb, d):
    diffs = [emb[w2i[m],:d]-emb[w2i[f],:d] for m,f in DEFP if m in w2i and f in w2i]
    if len(diffs) < 2: return np.zeros(d)
    return PCA(n_components=1).fit(np.array(diffs)).components_[0]

# ═══════════════════════════════════════════════════
# EXP A: MULTI-CATEGORY WEAT
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP A: Multi-Category WEAT")
log("="*70)

def weat_score(emb, d, X_w, Y_w, A_w, B_w):
    X=emb[[w2i[w] for w in X_w],:d]; Y=emb[[w2i[w] for w in Y_w],:d]
    A=emb[[w2i[w] for w in A_w],:d]; B=emb[[w2i[w] for w in B_w],:d]
    def s(w): return np.mean([cossim(w,a) for a in A])-np.mean([cossim(w,b) for b in B])
    sx=[s(x) for x in X]; sy=[s(y) for y in Y]
    return (np.mean(sx)-np.mean(sy))/(np.std(sx+sy)+1e-10)

weat_tests = {
    "Career/Family": (CAREER, FAMILY, MALE_NAMES, FEMALE_NAMES),
    "Math/Arts": (MATH, ARTS, MALE_NAMES, FEMALE_NAMES),
    "Science/Arts": (SCIENCE, ARTS, MALE_NAMES, FEMALE_NAMES),
}

fig, axes = plt.subplots(1, len(weat_tests), figsize=(6*len(weat_tests), 5))
if len(weat_tests) == 1: axes = [axes]
for idx, (tname, (X,Y,A,B)) in enumerate(weat_tests.items()):
    ws, wm = [], []
    for d in LEVELS:
        ws.append(weat_score(std_emb, d, X, Y, A, B))
        wm.append(weat_score(mrl_emb, d, X, Y, A, B))
    axes[idx].plot(LEVELS, ws, marker='o', label="Standard"); axes[idx].plot(LEVELS, wm, marker='s', label="MRL-v4")
    axes[idx].set_title(f"WEAT: {tname}"); axes[idx].set_xlabel("Dim"); axes[idx].set_ylabel("Effect Size")
    axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
    log(f"\n  {tname}:")
    for i,d in enumerate(LEVELS):
        log(f"    d={d}: Std={ws[i]:.4f}, MRL={wm[i]:.4f}, Diff={wm[i]-ws[i]:+.4f}")
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "expA_multi_weat.png"), dpi=150); plt.close()

# ═══════════════════════════════════════════════════
# EXP B: GENDER DIRECTION STABILITY
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP B: Gender Direction Stability Across Dimensions")
log("="*70)

# At each dim d, compute gender direction, then measure cos sim between
# the gender dir at d and the gender dir at d=300 (projected to d dims)
g_std_full = gdir(std_emb, 300)
g_mrl_full = gdir(mrl_emb, 300)

stab_s, stab_m = [], []
for d in LEVELS:
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    ss = abs(cossim(gs, g_std_full[:d])); sm = abs(cossim(gm, g_mrl_full[:d]))
    stab_s.append(ss); stab_m.append(sm)
    log(f"  d={d}: Std stability={ss:.4f}, MRL stability={sm:.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, stab_s, marker='o', label="Standard"); plt.plot(LEVELS, stab_m, marker='s', label="MRL-v4")
plt.title("Gender Direction Stability (cos sim to d=300 gender dir)")
plt.xlabel("Prefix Dim"); plt.ylabel("|cos(g_d, g_300[:d])|"); plt.legend(); plt.grid(True)
plt.ylim(0,1.05)
plt.savefig(os.path.join(PLOT_DIR, "expB_gender_stability.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP C: PER-WORD BIAS PROFILE (Projection onto gender dir)
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP C: Per-Word Gender Projection at d=300")
log("="*70)

proj_std = {w: float(np.dot(std_emb[w2i[w],:300], g_std_full)) for w in OCCUPATIONS}
proj_mrl = {w: float(np.dot(mrl_emb[w2i[w],:300], g_mrl_full)) for w in OCCUPATIONS}

# Sort by std projection
sorted_occ = sorted(OCCUPATIONS, key=lambda w: proj_std[w])
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(sorted_occ))
ax.barh(y - 0.2, [proj_std[w] for w in sorted_occ], 0.4, label="Standard", alpha=0.7)
ax.barh(y + 0.2, [proj_mrl[w] for w in sorted_occ], 0.4, label="MRL-v4", alpha=0.7)
ax.set_yticks(y); ax.set_yticklabels(sorted_occ, fontsize=7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel("Projection onto Gender Direction (+ = Male, - = Female)")
ax.set_title("Per-Occupation Gender Direction Projection (d=300)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "expC_perword_projection.png"), dpi=150); plt.close()

# Correlation between models
corr, pval = pearsonr([proj_std[w] for w in OCCUPATIONS], [proj_mrl[w] for w in OCCUPATIONS])
log(f"  Pearson correlation of per-word projections (Std vs MRL): r={corr:.4f}, p={pval:.6f}")
log(f"  Top 5 male-biased (Std): {sorted(OCCUPATIONS, key=lambda w: proj_std[w], reverse=True)[:5]}")
log(f"  Top 5 female-biased (Std): {sorted(OCCUPATIONS, key=lambda w: proj_std[w])[:5]}")
log(f"  Top 5 male-biased (MRL): {sorted(OCCUPATIONS, key=lambda w: proj_mrl[w], reverse=True)[:5]}")
log(f"  Top 5 female-biased (MRL): {sorted(OCCUPATIONS, key=lambda w: proj_mrl[w])[:5]}")

# ═══════════════════════════════════════════════════
# EXP D: BIAS PROJECTION HEATMAP ACROSS DIMENSIONS
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP D: Bias Projection Heatmap")
log("="*70)

heat_std = np.zeros((len(OCCUPATIONS), len(LEVELS)))
heat_mrl = np.zeros((len(OCCUPATIONS), len(LEVELS)))
for j, d in enumerate(LEVELS):
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    for i, w in enumerate(OCCUPATIONS):
        heat_std[i,j] = float(np.dot(std_emb[w2i[w],:d], gs))
        heat_mrl[i,j] = float(np.dot(mrl_emb[w2i[w],:d], gm))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
vmax = max(np.abs(heat_std).max(), np.abs(heat_mrl).max())
im1 = ax1.imshow(heat_std, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax1.set_xticks(range(len(LEVELS))); ax1.set_xticklabels(LEVELS)
ax1.set_yticks(range(len(OCCUPATIONS))); ax1.set_yticklabels(OCCUPATIONS, fontsize=6)
ax1.set_title("Standard: Gender Projection"); ax1.set_xlabel("Dim")
plt.colorbar(im1, ax=ax1, shrink=0.5)

im2 = ax2.imshow(heat_mrl, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax2.set_xticks(range(len(LEVELS))); ax2.set_xticklabels(LEVELS)
ax2.set_yticks(range(len(OCCUPATIONS))); ax2.set_yticklabels(OCCUPATIONS, fontsize=6)
ax2.set_title("MRL-v4: Gender Projection"); ax2.set_xlabel("Dim")
plt.colorbar(im2, ax=ax2, shrink=0.5)
plt.suptitle("Gender Direction Projection Heatmap (Blue=Female, Red=Male)")
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "expD_bias_heatmap.png"), dpi=150); plt.close()

# ═══════════════════════════════════════════════════
# EXP E: BIAS VARIANCE - Is bias spread or concentrated?
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP E: Bias Variance (Spread of projections)")
log("="*70)

var_s, var_m = [], []
for d in LEVELS:
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    ps = [float(np.dot(std_emb[w2i[w],:d], gs)) for w in OCCUPATIONS]
    pm = [float(np.dot(mrl_emb[w2i[w],:d], gm)) for w in OCCUPATIONS]
    var_s.append(np.std(ps)); var_m.append(np.std(pm))
    log(f"  d={d}: Std projection std={np.std(ps):.4f}, MRL projection std={np.std(pm):.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, var_s, marker='o', label="Standard"); plt.plot(LEVELS, var_m, marker='s', label="MRL-v4")
plt.title("Spread of Gender Projections (Std Dev across Occupations)")
plt.xlabel("Dim"); plt.ylabel("Std Dev of Projections"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "expE_bias_variance.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP F: GENDER SUBSPACE DIMENSIONALITY
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP F: Gender Subspace Dimensionality")
log("="*70)

diffs_s = [std_emb[w2i[m]]-std_emb[w2i[f]] for m,f in DEFP if m in w2i and f in w2i]
diffs_m = [mrl_emb[w2i[m]]-mrl_emb[w2i[f]] for m,f in DEFP if m in w2i and f in w2i]
nc = min(len(diffs_s), 8)
pca_gs = PCA(n_components=nc).fit(np.array(diffs_s))
pca_gm = PCA(n_components=nc).fit(np.array(diffs_m))

log(f"  Explained variance ratios (gender PCA):")
for i in range(nc):
    log(f"    PC{i+1}: Std={pca_gs.explained_variance_ratio_[i]:.4f}, MRL={pca_gm.explained_variance_ratio_[i]:.4f}")

plt.figure(figsize=(8,5))
plt.bar(np.arange(nc)-0.2, pca_gs.explained_variance_ratio_, 0.4, label="Standard")
plt.bar(np.arange(nc)+0.2, pca_gm.explained_variance_ratio_, 0.4, label="MRL-v4")
plt.title("Gender Subspace PCA Explained Variance"); plt.xlabel("PC"); plt.ylabel("Explained Variance Ratio")
plt.xticks(range(nc), [f"PC{i+1}" for i in range(nc)]); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "expF_gender_subspace.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP G: k-NN GENDER ASYMMETRY PER OCCUPATION
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP G: k-NN Gender Asymmetry (k=50)")
log("="*70)

k = 50
all_male_idx = set(w2i[w] for w in MALE_NAMES + MALE_PRONOUNS if w in w2i)
all_female_idx = set(w2i[w] for w in FEMALE_NAMES + FEMALE_PRONOUNS if w in w2i)

def gender_asymmetry(emb, d, word, k=50):
    ed = emb[:,:d]; en = ed / (np.linalg.norm(ed, axis=1, keepdims=True)+1e-10)
    q = en[w2i[word]]; sims = en @ q
    topk = np.argsort(sims)[-(k+1):-1]
    cm = sum(1 for n in topk if n in all_male_idx)
    cf = sum(1 for n in topk if n in all_female_idx)
    return cm, cf

d_test = 300
asym_std, asym_mrl = {}, {}
for w in OCCUPATIONS:
    cm_s, cf_s = gender_asymmetry(std_emb, d_test, w, k)
    cm_m, cf_m = gender_asymmetry(mrl_emb, d_test, w, k)
    asym_std[w] = (cm_s - cf_s) / (cm_s + cf_s + 1e-10)
    asym_mrl[w] = (cm_m - cf_m) / (cm_m + cf_m + 1e-10)

sorted_occ2 = sorted(OCCUPATIONS, key=lambda w: asym_std[w])
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(sorted_occ2))
ax.barh(y - 0.2, [asym_std[w] for w in sorted_occ2], 0.4, label="Standard", alpha=0.7, color='blue')
ax.barh(y + 0.2, [asym_mrl[w] for w in sorted_occ2], 0.4, label="MRL-v4", alpha=0.7, color='orange')
ax.set_yticks(y); ax.set_yticklabels(sorted_occ2, fontsize=7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel("Gender Asymmetry (+ = more male neighbors, - = more female)")
ax.set_title(f"k-NN Gender Asymmetry per Occupation (k={k}, d={d_test})")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "expG_knn_asymmetry.png"), dpi=150); plt.close()

log(f"  Mean |asymmetry| Std: {np.mean([abs(v) for v in asym_std.values()]):.4f}")
log(f"  Mean |asymmetry| MRL: {np.mean([abs(v) for v in asym_mrl.values()]):.4f}")
corr_asym, _ = pearsonr([asym_std[w] for w in OCCUPATIONS], [asym_mrl[w] for w in OCCUPATIONS])
log(f"  Pearson correlation of asymmetry (Std vs MRL): r={corr_asym:.4f}")

# ═══════════════════════════════════════════════════
# EXP H: DEFINITIONAL PAIR DISTANCES
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP H: Definitional Pair Cosine Distances")
log("="*70)

fig, ax = plt.subplots(figsize=(8,5))
for mname, emb in [("Standard", std_emb), ("MRL-v4", mrl_emb)]:
    dists = []
    for d in LEVELS:
        cd = [cossim(emb[w2i[m],:d], emb[w2i[f],:d]) for m,f in DEFP if m in w2i and f in w2i]
        dists.append(np.mean(cd))
        log(f"  d={d} {mname}: avg cos(male,female) = {np.mean(cd):.4f}")
    ax.plot(LEVELS, dists, marker='o' if 'Std' in mname else 's', label=mname)
ax.set_title("Avg Cosine Sim of Definitional Gender Pairs"); ax.set_xlabel("Dim"); ax.set_ylabel("Avg Cosine Sim")
ax.legend(); ax.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "expH_pair_distances.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP I: STEREOTYPE ALIGNMENT SCORE
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP I: Stereotype Alignment Score")
log("="*70)
# For stereotypically male occupations, measure avg projection onto male direction
# For stereotypically female occupations, measure avg projection onto female direction
# A "perfectly stereotyped" space would have high positive for male-stereo and
# high negative for female-stereo

stm_s, stm_m, stf_s, stf_m = [], [], [], []
for d in LEVELS:
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    # Male stereo occupations -> expect positive projection
    ms_s = np.mean([float(np.dot(std_emb[w2i[w],:d], gs)) for w in STEREOTYP_MALE])
    ms_m = np.mean([float(np.dot(mrl_emb[w2i[w],:d], gm)) for w in STEREOTYP_MALE])
    # Female stereo occupations -> expect negative projection
    fs_s = np.mean([float(np.dot(std_emb[w2i[w],:d], gs)) for w in STEREOTYP_FEMALE])
    fs_m = np.mean([float(np.dot(mrl_emb[w2i[w],:d], gm)) for w in STEREOTYP_FEMALE])
    stm_s.append(ms_s); stm_m.append(ms_m)
    stf_s.append(fs_s); stf_m.append(fs_m)
    log(f"  d={d}: Std male_stereo={ms_s:+.4f} female_stereo={fs_s:+.4f} | MRL male_stereo={ms_m:+.4f} female_stereo={fs_m:+.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(LEVELS, stm_s, marker='o', label="Std Male-Stereo", color='blue')
ax1.plot(LEVELS, stm_m, marker='s', label="MRL Male-Stereo", color='orange')
ax1.plot(LEVELS, stf_s, marker='o', label="Std Female-Stereo", color='blue', linestyle='--')
ax1.plot(LEVELS, stf_m, marker='s', label="MRL Female-Stereo", color='orange', linestyle='--')
ax1.set_title("Stereotype Group Avg Projection"); ax1.set_xlabel("Dim"); ax1.set_ylabel("Avg Projection")
ax1.legend(fontsize=8); ax1.grid(True); ax1.axhline(0, color='black', linewidth=0.5)

# Stereotype Gap = male_stereo_proj - female_stereo_proj
gap_s = [m-f for m,f in zip(stm_s, stf_s)]
gap_m = [m-f for m,f in zip(stm_m, stf_m)]
ax2.plot(LEVELS, gap_s, marker='o', label="Standard"); ax2.plot(LEVELS, gap_m, marker='s', label="MRL-v4")
ax2.set_title("Stereotype Gap (Male - Female Avg Projection)"); ax2.set_xlabel("Dim"); ax2.set_ylabel("Gap")
ax2.legend(); ax2.grid(True); ax2.axhline(0, color='black', linewidth=0.5)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "expI_stereotype_alignment.png"), dpi=150); plt.close()

# ═══════════════════════════════════════════════════
# EXP J: BIAS PRESERVATION CORRELATION ACROSS DIMS
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP J: Bias Preservation Correlation (per-word projection correlation across dims)")
log("="*70)

# For each model, how correlated are per-word projections at d=50 vs d=300?
for d in LEVELS[:-1]:
    gs_d = gdir(std_emb, d); gm_d = gdir(mrl_emb, d)
    gs_300 = gdir(std_emb, 300); gm_300 = gdir(mrl_emb, 300)
    
    ps_d = [float(np.dot(std_emb[w2i[w],:d], gs_d)) for w in OCCUPATIONS]
    ps_300 = [float(np.dot(std_emb[w2i[w],:300], gs_300)) for w in OCCUPATIONS]
    pm_d = [float(np.dot(mrl_emb[w2i[w],:d], gm_d)) for w in OCCUPATIONS]
    pm_300 = [float(np.dot(mrl_emb[w2i[w],:300], gm_300)) for w in OCCUPATIONS]
    
    cr_s, _ = pearsonr(ps_d, ps_300)
    cr_m, _ = pearsonr(pm_d, pm_300)
    log(f"  d={d} vs d=300: Std r={cr_s:.4f}, MRL r={cr_m:.4f}")

# ═══════════════════════════════════════════════════
# EXP K: GENDER INFORMATION CONTENT (% variance on gender dir)
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP K: Gender Information Content")
log("="*70)

gic_s, gic_m = [], []
for d in LEVELS:
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    # Project all vocab onto gender dir, compute variance
    projs_s = std_emb[:10000,:d] @ gs
    projs_m = mrl_emb[:10000,:d] @ gm
    total_var_s = np.sum(np.var(std_emb[:10000,:d], axis=0))
    total_var_m = np.sum(np.var(mrl_emb[:10000,:d], axis=0))
    gvar_s = np.var(projs_s) / total_var_s * 100
    gvar_m = np.var(projs_m) / total_var_m * 100
    gic_s.append(gvar_s); gic_m.append(gvar_m)
    log(f"  d={d}: Std gender_var={gvar_s:.4f}%, MRL gender_var={gvar_m:.4f}%")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, gic_s, marker='o', label="Standard"); plt.plot(LEVELS, gic_m, marker='s', label="MRL-v4")
plt.title("Gender Direction Variance as % of Total Variance")
plt.xlabel("Dim"); plt.ylabel("% of Total Variance"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "expK_gender_info_content.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP L: INTER-MODEL BIAS CORRELATION (per occupation)
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP L: Inter-Model Bias Correlation")
log("="*70)

corrs = []
for d in LEVELS:
    gs = gdir(std_emb, d); gm = gdir(mrl_emb, d)
    ps = [float(np.dot(std_emb[w2i[w],:d], gs)) for w in OCCUPATIONS]
    pm = [float(np.dot(mrl_emb[w2i[w],:d], gm)) for w in OCCUPATIONS]
    cr, _ = pearsonr(ps, pm)
    corrs.append(cr)
    log(f"  d={d}: Pearson r(Std proj, MRL proj) = {cr:.4f}")

plt.figure(figsize=(8,5))
plt.plot(LEVELS, corrs, marker='o', color='purple')
plt.title("Cross-Model Bias Correlation (per-word RIPA)")
plt.xlabel("Dim"); plt.ylabel("Pearson r"); plt.grid(True)
plt.ylim(-1, 1); plt.axhline(0, color='black', linewidth=0.5)
plt.savefig(os.path.join(PLOT_DIR, "expL_intermodel_corr.png")); plt.close()

# ═══════════════════════════════════════════════════
# EXP M: OCCUPATION CLUSTERING BY GENDER
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP M: Occupation Clustering")
log("="*70)

for d in [50, 300]:
    for mname, emb in [("Standard", std_emb), ("MRL-v4", mrl_emb)]:
        vecs = emb[[w2i[w] for w in OCCUPATIONS], :d]
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True)+1e-10)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(vn)
        c0 = [OCCUPATIONS[i] for i in range(len(OCCUPATIONS)) if km.labels_[i]==0]
        c1 = [OCCUPATIONS[i] for i in range(len(OCCUPATIONS)) if km.labels_[i]==1]
        
        # Check which cluster has more stereotypically male/female
        m0 = len([w for w in c0 if w in STEREOTYP_MALE])
        m1 = len([w for w in c1 if w in STEREOTYP_MALE])
        f0 = len([w for w in c0 if w in STEREOTYP_FEMALE])
        f1 = len([w for w in c1 if w in STEREOTYP_FEMALE])
        log(f"  {mname} d={d}: C0({len(c0)} words, {m0}M/{f0}F stereo) C1({len(c1)} words, {m1}M/{f1}F stereo)")

# ═══════════════════════════════════════════════════
# EXP N: INDIVIDUAL WEAT s(w) DISTRIBUTIONS
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP N: Individual WEAT s(w) Value Distributions at d=300")
log("="*70)

def weat_s_values(emb, d, targets, A_w, B_w):
    A=emb[[w2i[w] for w in A_w],:d]; B=emb[[w2i[w] for w in B_w],:d]
    vals = {}
    for w in targets:
        v = emb[w2i[w],:d]
        vals[w] = np.mean([cossim(v,a) for a in A]) - np.mean([cossim(v,b) for b in B])
    return vals

sv_std = weat_s_values(std_emb, 300, OCCUPATIONS, MALE_NAMES, FEMALE_NAMES)
sv_mrl = weat_s_values(mrl_emb, 300, OCCUPATIONS, MALE_NAMES, FEMALE_NAMES)

fig, ax = plt.subplots(figsize=(8,5))
ax.hist([sv_std[w] for w in OCCUPATIONS], bins=15, alpha=0.5, label="Standard", density=True)
ax.hist([sv_mrl[w] for w in OCCUPATIONS], bins=15, alpha=0.5, label="MRL-v4", density=True)
ax.set_title("Distribution of Individual WEAT s(w) Values (d=300)")
ax.set_xlabel("s(w) = mean_cos(w,male) - mean_cos(w,female)"); ax.set_ylabel("Density")
ax.axvline(0, color='black', linewidth=0.5); ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "expN_weat_distribution.png")); plt.close()

log(f"  Std: mean s(w)={np.mean(list(sv_std.values())):.4f}, std={np.std(list(sv_std.values())):.4f}")
log(f"  MRL: mean s(w)={np.mean(list(sv_mrl.values())):.4f}, std={np.std(list(sv_mrl.values())):.4f}")

# ═══════════════════════════════════════════════════
# EXP O: COSINE DISTANCE TO MALE vs FEMALE CENTROIDS
# ═══════════════════════════════════════════════════
log("\n" + "="*70)
log("EXP O: Male vs Female Centroid Distance for Stereotype Words")
log("="*70)

def centroid_distances(emb, d, targets, male_w, female_w):
    mu_m = np.mean(emb[[w2i[w] for w in male_w],:d], axis=0)
    mu_f = np.mean(emb[[w2i[w] for w in female_w],:d], axis=0)
    results = {}
    for w in targets:
        dm = cossim(emb[w2i[w],:d], mu_m)
        df = cossim(emb[w2i[w],:d], mu_f)
        results[w] = dm - df  # positive = closer to male
    return results

cd_std = centroid_distances(std_emb, 300, OCCUPATIONS, MALE_PRONOUNS, FEMALE_PRONOUNS)
cd_mrl = centroid_distances(mrl_emb, 300, OCCUPATIONS, MALE_PRONOUNS, FEMALE_PRONOUNS)

sorted_occ3 = sorted(OCCUPATIONS, key=lambda w: cd_std[w])
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(sorted_occ3))
ax.barh(y - 0.2, [cd_std[w] for w in sorted_occ3], 0.4, label="Standard", alpha=0.7)
ax.barh(y + 0.2, [cd_mrl[w] for w in sorted_occ3], 0.4, label="MRL-v4", alpha=0.7)
ax.set_yticks(y); ax.set_yticklabels(sorted_occ3, fontsize=7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel("cos(w, male_centroid) - cos(w, female_centroid)")
ax.set_title("Centroid Distance Differential per Occupation (d=300)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "expO_centroid_diff.png"), dpi=150); plt.close()

corr_cd, _ = pearsonr([cd_std[w] for w in OCCUPATIONS], [cd_mrl[w] for w in OCCUPATIONS])
log(f"  Pearson r(centroid_diff Std vs MRL) = {corr_cd:.4f}")

# ═══════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════
with open(os.path.join(EMB_DIR, "deep_bias_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))

log(f"\nAll done. Plots in: {PLOT_DIR}")
log(f"Text log in: {os.path.join(EMB_DIR, 'deep_bias_results.txt')}")

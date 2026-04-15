"""
Comprehensive Bias Evaluation Suite for MRL-v4 vs Standard Word2Vec
===================================================================
Implements 7 bias tests across all nesting dimensions.
"""
import numpy as np
import os
import json
import pickle
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE    = r"c:\Users\abhay\Desktop\MRL TESTS"
EMB_DIR = os.path.join(BASE, "Data_and_Embeddings", "_v3_embeddings_W2V",
                       "Embeddings", "mrl_bias_v4_embeddings")
PLOT_DIR = os.path.join(BASE, "Results_and_Reports", "bias_plots_v4")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD EMBEDDINGS + VOCAB
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

LEVELS = [50, 100, 150, 200, 250, 300]

# ─────────────────────────────────────────────
# WORD LISTS
# ─────────────────────────────────────────────
DEFINITIONAL_PAIRS = [
    ("he", "she"), ("man", "woman"), ("boy", "girl"),
    ("father", "mother"), ("son", "daughter"),
    ("husband", "wife"), ("brother", "sister"), ("king", "queen"),
]

MALE_ATTRS = ["john", "paul", "mike", "kevin", "steve", "greg", "jeff", "bill"]
FEMALE_ATTRS = ["amy", "joan", "lisa", "sarah", "diana", "kate", "ann", "donna"]

CAREER_TARGETS = ["executive", "management", "professional", "corporation",
                  "salary", "office", "business", "career"]
FAMILY_TARGETS = ["home", "parents", "children", "family",
                  "cousins", "marriage", "wedding", "relatives"]

NEUTRAL_OCCUPATIONS = [
    "doctor", "nurse", "engineer", "teacher", "programmer",
    "librarian", "soldier", "receptionist", "housekeeper",
    "carpenter", "mechanic", "pilot", "accountant", "plumber",
    "professor", "chef", "scientist", "artist", "manager", "secretary"
]

def filter_words(word_list):
    """Return only words present in vocab."""
    return [w for w in word_list if w in w2i]

def get_vecs(word_list, emb, d):
    """Get embedding vectors for word list at dimension d."""
    idxs = [w2i[w] for w in word_list]
    return emb[idxs, :d]

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_gender_direction(emb, d):
    """Compute gender direction as 1st PC of definitional pair differences."""
    diffs = []
    for m, f in DEFINITIONAL_PAIRS:
        if m in w2i and f in w2i:
            vm = emb[w2i[m], :d]
            vf = emb[w2i[f], :d]
            diffs.append(vm - vf)
    if len(diffs) < 2:
        return np.zeros(d)
    diffs = np.array(diffs)
    pca = PCA(n_components=1)
    pca.fit(diffs)
    return pca.components_[0]


# ═══════════════════════════════════════════════
# TEST 1: WEAT
# ═══════════════════════════════════════════════
def calc_weat(emb, d, X_words, Y_words, A_words, B_words):
    """
    WEAT effect size.
    X, Y = target word sets (e.g., Career, Family)
    A, B = attribute word sets (e.g., Male names, Female names)
    """
    X = get_vecs(X_words, emb, d)
    Y = get_vecs(Y_words, emb, d)
    A = get_vecs(A_words, emb, d)
    B = get_vecs(B_words, emb, d)

    def s(w, A_vecs, B_vecs):
        mean_a = np.mean([cosine_sim(w, a) for a in A_vecs])
        mean_b = np.mean([cosine_sim(w, b) for b in B_vecs])
        return mean_a - mean_b

    sx = [s(x, A, B) for x in X]
    sy = [s(y, A, B) for y in Y]

    numerator = np.mean(sx) - np.mean(sy)
    all_s = sx + sy
    denominator = np.std(all_s) + 1e-10
    effect_size = numerator / denominator
    return float(effect_size)


# ═══════════════════════════════════════════════
# TEST 2: DirectBias
# ═══════════════════════════════════════════════
def calc_direct_bias(emb, d, neutral_words, c=1):
    """DirectBias: avg |cos(w, gender_dir)|^c for neutral words."""
    g = get_gender_direction(emb, d)
    if np.linalg.norm(g) < 1e-10:
        return 0.0
    scores = []
    for w in neutral_words:
        if w in w2i:
            v = emb[w2i[w], :d]
            scores.append(abs(cosine_sim(v, g)) ** c)
    return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════
# TEST 3: RIPA
# ═══════════════════════════════════════════════
def calc_ripa(emb, d, neutral_words):
    """RIPA: unnormalized inner product with gender direction."""
    g = get_gender_direction(emb, d)
    if np.linalg.norm(g) < 1e-10:
        return 0.0, 0.0
    scores = []
    for w in neutral_words:
        if w in w2i:
            v = emb[w2i[w], :d]
            scores.append(abs(float(np.dot(v, g))))
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.max(scores))


# ═══════════════════════════════════════════════
# TEST 4: ECT
# ═══════════════════════════════════════════════
def calc_ect(emb, d, target_words, male_attrs, female_attrs):
    """ECT: Spearman correlation of target distances to male vs female centroids."""
    m_vecs = get_vecs(male_attrs, emb, d)
    f_vecs = get_vecs(female_attrs, emb, d)
    mu_m = np.mean(m_vecs, axis=0)
    mu_f = np.mean(f_vecs, axis=0)

    d_m, d_f = [], []
    for w in target_words:
        if w in w2i:
            v = emb[w2i[w], :d]
            d_m.append(cosine_sim(v, mu_m))
            d_f.append(cosine_sim(v, mu_f))
    if len(d_m) < 3:
        return 0.0
    rho, _ = spearmanr(d_m, d_f)
    return float(rho)


# ═══════════════════════════════════════════════
# TEST 5: NBM
# ═══════════════════════════════════════════════
def calc_nbm(emb, d, target_words, male_attrs, female_attrs, k=100):
    """NBM: neighborhood bias metric."""
    male_set = set(male_attrs)
    female_set = set(female_attrs)
    # Precompute normalized embeddings for fast neighbor search
    emb_d = emb[:, :d].copy()
    norms = np.linalg.norm(emb_d, axis=1, keepdims=True) + 1e-10
    emb_norm = emb_d / norms

    nbm_scores = []
    for w in target_words:
        if w not in w2i:
            continue
        idx = w2i[w]
        q = emb_norm[idx]
        sims = emb_norm @ q  # (V,)
        top_k_idx = np.argsort(sims)[-(k+1):-1][::-1]  # exclude self

        # Build reverse index for speed
        c_m, c_f = 0, 0
        for ni in top_k_idx:
            # Find the word for this index
            pass
        # We need i2w mapping
        # Instead, pre-build idx sets
        male_idxs = set(w2i[w] for w in male_attrs if w in w2i)
        female_idxs = set(w2i[w] for w in female_attrs if w in w2i)
        for ni in top_k_idx:
            if ni in male_idxs:
                c_m += 1
            elif ni in female_idxs:
                c_f += 1
        if c_m + c_f == 0:
            nbm_scores.append(0.0)
        else:
            nbm_scores.append(abs(c_m - c_f) / (c_m + c_f))
    return float(np.mean(nbm_scores)) if nbm_scores else 0.0


# ═══════════════════════════════════════════════
# TEST 6: Cluster Purity
# ═══════════════════════════════════════════════
def calc_cluster_purity(emb, d, male_attrs, female_attrs):
    """K-Means cluster purity for gender separability."""
    words = male_attrs + female_attrs
    labels_true = [0]*len(male_attrs) + [1]*len(female_attrs)
    vecs = get_vecs(words, emb, d)
    # Normalize for angular clustering
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_n = vecs / norms
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(vecs_n)
    pred = km.labels_

    # Cluster purity: try both label assignments and pick the best
    match_0 = sum(1 for p, t in zip(pred, labels_true) if p == t)
    match_1 = sum(1 for p, t in zip(pred, labels_true) if (1-p) == t)
    purity = max(match_0, match_1) / len(labels_true)
    return float(purity)


# ═══════════════════════════════════════════════
# TEST 7: Bias Analogy
# ═══════════════════════════════════════════════
def calc_bias_analogy(emb, d, target_words, he_word="he", she_word="she"):
    """Analogy: she - he + target, check if result is stereotyped."""
    if he_word not in w2i or she_word not in w2i:
        return 0.0, []

    v_he = emb[w2i[he_word], :d]
    v_she = emb[w2i[she_word], :d]
    gender_offset = v_she - v_he

    # Build reverse index
    # We need i2w
    try:
        i2w = {i: w for w, i in w2i.items()}
    except:
        return 0.0, []

    emb_d = emb[:, :d]
    norms = np.linalg.norm(emb_d, axis=1, keepdims=True) + 1e-10
    emb_norm = emb_d / norms

    results = []
    for w in target_words:
        if w not in w2i:
            continue
        v_w = emb[w2i[w], :d]
        query = v_w + gender_offset
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        sims = emb_norm @ q_norm
        # Exclude input words
        exclude = {w2i[he_word], w2i[she_word], w2i[w]}
        for ex in exclude:
            sims[ex] = -999
        best_idx = np.argmax(sims)
        best_word = i2w.get(best_idx, "???")
        results.append((w, best_word, float(sims[best_idx])))

    return results


# ═══════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    # Filter word lists to in-vocab words
    male_a = filter_words(MALE_ATTRS)
    female_a = filter_words(FEMALE_ATTRS)
    career_t = filter_words(CAREER_TARGETS)
    family_t = filter_words(FAMILY_TARGETS)
    neutral_o = filter_words(NEUTRAL_OCCUPATIONS)
    all_targets = filter_words(NEUTRAL_OCCUPATIONS + CAREER_TARGETS + FAMILY_TARGETS)

    print(f"\nIn-vocab counts: Male attrs={len(male_a)}, Female attrs={len(female_a)}, "
          f"Career={len(career_t)}, Family={len(family_t)}, Neutral={len(neutral_o)}")

    results = {"standard": {}, "mrl_v4": {}}
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("BIAS EVALUATION SUITE - MRL-v4 vs Standard Word2Vec")
    log("=" * 70)

    for model_name, emb in [("standard", std_emb), ("mrl_v4", mrl_emb)]:
        log(f"\n{'='*50}")
        log(f"MODEL: {model_name.upper()}")
        log(f"{'='*50}")
        results[model_name] = {}

        for d in LEVELS:
            log(f"\n--- Dimension {d} ---")
            res = {}

            # Test 1: WEAT
            weat = calc_weat(emb, d, career_t, family_t, male_a, female_a)
            res["weat_effect_size"] = weat
            log(f"  WEAT (Career/Family x Male/Female): d = {weat:.4f}")

            # Test 2: DirectBias
            db = calc_direct_bias(emb, d, neutral_o)
            res["direct_bias"] = db
            log(f"  DirectBias (neutral occupations):    {db:.4f}")

            # Test 3: RIPA
            ripa_mean, ripa_max = calc_ripa(emb, d, neutral_o)
            res["ripa_mean"] = ripa_mean
            res["ripa_max"] = ripa_max
            log(f"  RIPA mean|max (neutral occupations): {ripa_mean:.4f} | {ripa_max:.4f}")

            # Test 4: ECT
            ect = calc_ect(emb, d, all_targets, male_a, female_a)
            res["ect_spearman"] = ect
            log(f"  ECT Spearman rho:                    {ect:.4f}")

            # Test 5: NBM
            nbm = calc_nbm(emb, d, neutral_o, male_a, female_a, k=100)
            res["nbm_mean"] = nbm
            log(f"  NBM mean |bias| (k=100):             {nbm:.4f}")

            # Test 6: Cluster Purity
            cp = calc_cluster_purity(emb, d, male_a, female_a)
            res["cluster_purity"] = cp
            log(f"  Cluster Purity (K=2):                {cp:.4f}")

            # Test 7: Bias Analogy (only at d=300 for readability)
            if d == 300:
                analogy_results = calc_bias_analogy(emb, d, neutral_o)
                res["analogy_results"] = [(w, r, round(s, 4)) for w, r, s in analogy_results]
                log(f"  Bias Analogies (he:she :: word:?):")
                for w, r, s in analogy_results:
                    log(f"    {w:15} -> {r:15} (sim={s:.4f})")
            else:
                res["analogy_results"] = []

            results[model_name][str(d)] = res

    # ─────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────
    results_dir = os.path.join(BASE, "Results_and_Reports")
    json_path = os.path.join(results_dir, "bias_results_v4.json")
    txt_path = os.path.join(results_dir, "bias_results_v4.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nJSON results saved to: {json_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Text results saved to: {txt_path}")

    # ─────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────
    def extract_metric(results, model, metric):
        return [results[model][str(d)][metric] for d in LEVELS]

    metrics_to_plot = [
        ("weat_effect_size", "WEAT Effect Size (Career/Family)", "WEAT Effect Size (d)"),
        ("direct_bias",      "DirectBias (Neutral Occupations)", "DirectBias Score"),
        ("ripa_mean",        "RIPA Mean (Neutral Occupations)",  "Mean |RIPA|"),
        ("ripa_max",         "RIPA Max (Neutral Occupations)",   "Max |RIPA|"),
        ("ect_spearman",     "ECT Spearman Rho",                 "Spearman rho"),
        ("nbm_mean",         "NBM Mean |Bias| (k=100)",          "Mean |NBM|"),
        ("cluster_purity",   "Cluster Purity (K=2)",             "Purity (Accuracy)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i, (metric, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[i]
        std_vals = extract_metric(results, "standard", metric)
        mrl_vals = extract_metric(results, "mrl_v4", metric)
        ax.plot(LEVELS, std_vals, marker='o', label="Standard", linewidth=2)
        ax.plot(LEVELS, mrl_vals, marker='s', label="MRL-v4", linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Prefix Dimension")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(metrics_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Bias Evaluation: MRL-v4 vs Standard Word2Vec", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(PLOT_DIR, "bias_all_metrics_v4.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    log(f"Combined plot saved to: {combined_path}")

    # Individual plots
    for metric, title, ylabel in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        std_vals = extract_metric(results, "standard", metric)
        mrl_vals = extract_metric(results, "mrl_v4", metric)
        plt.plot(LEVELS, std_vals, marker='o', label="Standard", linewidth=2)
        plt.plot(LEVELS, mrl_vals, marker='s', label="MRL-v4", linewidth=2)
        plt.title(title)
        plt.xlabel("Prefix Dimension")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = f"bias_{metric}_v4.png"
        plt.savefig(os.path.join(PLOT_DIR, fname))
        plt.close()

    log(f"\nIndividual plots saved to: {PLOT_DIR}")
    log("\n=== BIAS EVALUATION COMPLETE ===")

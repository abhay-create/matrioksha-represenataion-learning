import numpy as np, os
from scipy.stats import spearmanr
emb_dir = r"c:\Users\abhay\Desktop\MRL TESTS\Data_and_Embeddings\_v3_embeddings_W2V\Embeddings\mrl_bias_v4_embeddings"
vw = np.load(os.path.join(emb_dir, "vocab_words.npy"))
w2i = {w: i for i, w in enumerate(vw)}
std = np.load(os.path.join(emb_dir, "standard_w2v_embeddings.npy"))
mrl = np.load(os.path.join(emb_dir, "mrl_v4_w2v_embeddings.npy"))
simlex = r"c:\Users\abhay\Desktop\MRL TESTS\Data_and_Embeddings\SimLex-999\SimLex-999.txt"
lines = open(simlex, encoding='utf-8').readlines()[1:]
pairs = []
for l in lines:
    p = l.strip().split('\t')
    if p[0] in w2i and p[1] in w2i:
        pairs.append((w2i[p[0]], w2i[p[1]], float(p[3])))

print(f"Loaded {len(pairs)} pairs")
for d in [50, 100, 150, 200, 250, 300]:
    s1 = [np.dot(std[i1,:d], std[i2,:d])/(np.linalg.norm(std[i1,:d])*np.linalg.norm(std[i2,:d])+1e-9) for i1, i2, _ in pairs]
    m1 = [np.dot(mrl[i1,:d], mrl[i2,:d])/(np.linalg.norm(mrl[i1,:d])*np.linalg.norm(mrl[i2,:d])+1e-9) for i1, i2, _ in pairs]
    h = [s for _, _, s in pairs]
    print(f"Dim {d:3d}: Std={spearmanr(s1,h)[0]:.4f}, MRL-v4={spearmanr(m1,h)[0]:.4f}")

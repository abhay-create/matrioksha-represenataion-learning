# ═══════════════════════════════════════════════════
# MRL-v4: Relational Distillation MRL Word2Vec
# Drop-in replacement for MRLWord2Vec in mrl-v3.ipynb
# ═══════════════════════════════════════════════════
#
# Usage:
#   1. Run cells 0-2 of mrl-v3.ipynb (config, data)
#   2. Replace MRLWord2Vec with MRLWord2Vec_v4_RD
#   3. Use existing _train_skipgram — no changes needed
#
# Minimal diff from mrl-v3:
#   - model = MRLWord2Vec_v4_RD(vocab.size)  instead of MRLWord2Vec(vocab.size)
#   - Optionally increase PATIENCE to 5, reduce LR to 0.0005

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config (must match mrl-v3 cell 1) ─────────────
EMBED_DIM   = 300
MRL_NESTING = [50, 100, 150, 200, 250, 300]


# ── Reused from mrl-v3 cell 3 ─────────────────────
def neg_sampling_loss(center, context, negatives):
    """
    L = -log σ(v_c · v_+) - Σ log σ(-v_c · v_k)
    center: (B, d)  context: (B, d)  negatives: (B, K, d)
    """
    pos_loss = F.logsigmoid((center * context).sum(dim=1))
    neg_loss = F.logsigmoid(
        -torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)
    ).sum(dim=1)
    return -(pos_loss + neg_loss).mean()


class _BaseW2V(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.W_in  = nn.Embedding(vocab_size, EMBED_DIM)
        self.W_out = nn.Embedding(vocab_size, EMBED_DIM)
        nn.init.uniform_(self.W_in.weight,  -0.5/EMBED_DIM, 0.5/EMBED_DIM)
        nn.init.zeros_(self.W_out.weight)

    def get_embeddings(self):
        return self.W_in.weight.detach().cpu().float().numpy()

    def get_prefix(self, m):
        return self.W_in.weight[:, :m].detach().cpu().float().numpy()


class StandardWord2Vec(_BaseW2V):
    def forward(self, centers, contexts, negatives):
        return neg_sampling_loss(
            self.W_in(centers), self.W_out(contexts), self.W_out(negatives)
        )


# ── NEW: Projection Head ──────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── NEW: MRL-v4 with Relational Distillation ──────
class MRLWord2Vec_v4_RD(_BaseW2V):
    """
    Relational Distillation MRL.

    Loss = L_NS(300d)
         + (1/5) Σ_{m<300} [ α·KL(teacher_B×B, student_B×B) + β·MSE(cos) ]

    The projection heads are ONLY used during training.
    At inference, use get_embeddings() / get_prefix(m) as before.
    """

    def __init__(self, vocab_size, nesting=None,
                 proj_hidden=512, proj_out=64,
                 sub_batch=2048, tau=0.05,
                 alpha_kl=1.0, beta_cos=0.5):
        super().__init__(vocab_size)
        self.nesting  = nesting or MRL_NESTING
        assert self.nesting[-1] == EMBED_DIM
        self.sub_batch = sub_batch
        self.tau       = tau
        self.alpha_kl  = alpha_kl
        self.beta_cos  = beta_cos
        self.lam       = 1.0 / (len(self.nesting) - 1)  # average over prefix levels

        # Projection heads (discarded after training)
        self.teacher_proj = nn.ModuleDict()
        self.student_proj = nn.ModuleDict()
        for m in self.nesting[:-1]:  # 50, 100, 150, 200, 250
            self.teacher_proj[str(m)] = ProjectionHead(EMBED_DIM, proj_hidden, proj_out)
            self.student_proj[str(m)] = ProjectionHead(m, proj_hidden, proj_out)

    def forward(self, centers, contexts, negatives):
        vc = self.W_in(centers)        # (B, 300)
        vp = self.W_out(contexts)      # (B, 300)
        vn = self.W_out(negatives)     # (B, K, 300)

        # ── 1. Primary NS loss on full 300d ────────
        loss = neg_sampling_loss(vc, vp, vn)

        # ── 2. Relational distillation per prefix ──
        B = vc.shape[0]
        sB = min(self.sub_batch, B)
        idx = torch.randperm(B, device=vc.device)[:sB]
        vc_sub = vc[idx]                               # (sB, 300)

        for m in self.nesting[:-1]:
            # Teacher: project full 300d → common space
            t = self.teacher_proj[str(m)](vc_sub.detach())
            t = F.normalize(t, dim=1)                  # (sB, proj_out)
            t_sim = t @ t.T / self.tau                 # (sB, sB)
            t_sim.fill_diagonal_(-1e9)
            t_dist = F.log_softmax(t_sim, dim=1)

            # Student: project m-dim prefix → same common space
            s = self.student_proj[str(m)](vc_sub[:, :m])
            s = F.normalize(s, dim=1)
            s_sim = s @ s.T / self.tau
            s_sim.fill_diagonal_(-1e9)
            s_dist = F.log_softmax(s_sim, dim=1)

            kl = F.kl_div(s_dist, t_dist.detach(),
                          log_target=True, reduction='batchmean')

            # Cosine MSE anchor on original (center, context) pairs
            with torch.no_grad():
                teacher_cos = F.cosine_similarity(vc, vp, dim=1)
            student_cos = F.cosine_similarity(vc[:, :m], vp[:, :m], dim=1)
            cos_mse = F.mse_loss(student_cos, teacher_cos)

            loss += self.lam * (self.alpha_kl * kl + self.beta_cos * cos_mse)

        return loss

# ═══════════════════════════════════════════════════
# CELL 3 — MODELS
# 6 model classes: Standard/MRL × Word2Vec/GloVe/FastText
# ═══════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


def neg_sampling_loss(center, context, negatives):
    """
    L = -log σ(v_c · v_+) - Σ log σ(-v_c · v_k)
    center:    (B, d)    context:   (B, d)    negatives: (B, K, d)
    """
    pos_loss = F.logsigmoid((center * context).sum(dim=1))
    neg_loss = F.logsigmoid(
        -torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)
    ).sum(dim=1)
    return -(pos_loss + neg_loss).mean()


# ═══════════════════════════════════════════════════
# WORD2VEC
# ═══════════════════════════════════════════════════

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


class MRLWord2Vec(_BaseW2V):
    """L_MRL = (1/|M|) Σ_{m∈M} L_NS(f(w)[:m], f(c)[:m])"""
    def __init__(self, vocab_size, nesting=None):
        super().__init__(vocab_size)
        self.nesting = nesting or MRL_NESTING
        assert self.nesting[-1] == EMBED_DIM
        self.lam = 1.0 / len(self.nesting)

    def forward(self, centers, contexts, negatives):
        vc = self.W_in(centers)
        vp = self.W_out(contexts)
        vn = self.W_out(negatives)
        loss = vc.new_zeros(1).squeeze()
        for m in self.nesting:
            loss = loss + self.lam * neg_sampling_loss(vc[:,:m], vp[:,:m], vn[:,:,:m])
        return loss


# ═══════════════════════════════════════════════════
# GLOVE
# ═══════════════════════════════════════════════════

class _BaseGloVe(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.W       = nn.Embedding(vocab_size, EMBED_DIM)
        self.W_tilde = nn.Embedding(vocab_size, EMBED_DIM)
        self.b       = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        nn.init.uniform_(self.W.weight,       -0.5/EMBED_DIM, 0.5/EMBED_DIM)
        nn.init.uniform_(self.W_tilde.weight, -0.5/EMBED_DIM, 0.5/EMBED_DIM)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b_tilde.weight)

    def get_embeddings(self):
        return (self.W.weight + self.W_tilde.weight).detach().cpu().float().numpy()

    def get_prefix(self, m):
        return (self.W.weight[:, :m] + self.W_tilde.weight[:, :m]).detach().cpu().float().numpy()


class StandardGloVe(_BaseGloVe):
    """J = Σ f(X_ij) * (w_i · w̃_j + b_i + b̃_j - log X_ij)²"""
    def forward(self, i_idx, j_idx, x_ij):
        wi = self.W(i_idx)                # (B, D)
        wj = self.W_tilde(j_idx)          # (B, D)
        bi = self.b(i_idx).squeeze(1)     # (B,)
        bj = self.b_tilde(j_idx).squeeze(1)
        dot  = (wi * wj).sum(dim=1)
        diff = dot + bi + bj - torch.log(x_ij)
        weight = torch.clamp(x_ij / GLOVE_X_MAX, max=1.0) ** GLOVE_ALPHA
        return (weight * diff ** 2).mean()


class MRLGloVe(_BaseGloVe):
    def __init__(self, vocab_size, nesting=None):
        super().__init__(vocab_size)
        self.nesting = nesting or MRL_NESTING
        assert self.nesting[-1] == EMBED_DIM
        self.lam = 1.0 / len(self.nesting)

    def forward(self, i_idx, j_idx, x_ij):
        wi = self.W(i_idx)
        wj = self.W_tilde(j_idx)
        bi = self.b(i_idx).squeeze(1)
        bj = self.b_tilde(j_idx).squeeze(1)
        weight = torch.clamp(x_ij / GLOVE_X_MAX, max=1.0) ** GLOVE_ALPHA
        log_x  = torch.log(x_ij)
        loss = wi.new_zeros(1).squeeze()
        for m in self.nesting:
            dot  = (wi[:,:m] * wj[:,:m]).sum(dim=1)
            diff = dot + bi + bj - log_x
            loss = loss + self.lam * (weight * diff ** 2).mean()
        return loss


# ═══════════════════════════════════════════════════
# FASTTEXT
# ═══════════════════════════════════════════════════

class _BaseFT(nn.Module):
    def __init__(self, vocab_size, ngram_matrix, ngram_lengths):
        super().__init__()
        self.W_in  = nn.Embedding(vocab_size, EMBED_DIM)
        self.W_out = nn.Embedding(vocab_size, EMBED_DIM)
        self.W_ng  = nn.Embedding(FT_BUCKETS, EMBED_DIM)
        nn.init.uniform_(self.W_in.weight,  -0.5/EMBED_DIM, 0.5/EMBED_DIM)
        nn.init.zeros_(self.W_out.weight)
        nn.init.uniform_(self.W_ng.weight, -0.5/EMBED_DIM, 0.5/EMBED_DIM)
        # Buffers move to GPU with model.to(device)
        self.register_buffer("ng_matrix",  torch.tensor(ngram_matrix,  dtype=torch.long))
        self.register_buffer("ng_lengths", torch.tensor(ngram_lengths, dtype=torch.long))
        self._max_ng = ngram_matrix.shape[1]

    def _enrich(self, word_ids):
        """word embedding + mean(n-gram embeddings)"""
        word_emb = self.W_in(word_ids)                             # (B, D)
        ng_ids   = self.ng_matrix[word_ids]                        # (B, max_ng)
        ng_lens  = self.ng_lengths[word_ids].unsqueeze(1).float()  # (B, 1)
        ng_emb   = self.W_ng(ng_ids)                               # (B, max_ng, D)
        mask     = torch.arange(self._max_ng, device=word_ids.device) < ng_lens
        ng_emb   = ng_emb * mask.unsqueeze(2)
        return (word_emb + ng_emb.sum(dim=1)) / (1.0 + ng_lens)

    def get_embeddings(self):
        with torch.no_grad():
            ids = torch.arange(self.W_in.weight.shape[0], device=self.W_in.weight.device)
            # Process in chunks to avoid OOM
            embs = []
            for start in range(0, len(ids), 4096):
                embs.append(self._enrich(ids[start:start+4096]).cpu())
            return torch.cat(embs).float().numpy()

    def get_prefix(self, m):
        full = self.get_embeddings()
        return full[:, :m]


class StandardFastText(_BaseFT):
    def forward(self, centers, contexts, negatives):
        return neg_sampling_loss(
            self._enrich(centers), self.W_out(contexts), self.W_out(negatives)
        )


class MRLFastText(_BaseFT):
    def __init__(self, vocab_size, ngram_matrix, ngram_lengths, nesting=None):
        super().__init__(vocab_size, ngram_matrix, ngram_lengths)
        self.nesting = nesting or MRL_NESTING
        assert self.nesting[-1] == EMBED_DIM
        self.lam = 1.0 / len(self.nesting)

    def forward(self, centers, contexts, negatives):
        vc = self._enrich(centers)
        vp = self.W_out(contexts)
        vn = self.W_out(negatives)
        loss = vc.new_zeros(1).squeeze()
        for m in self.nesting:
            loss = loss + self.lam * neg_sampling_loss(vc[:,:m], vp[:,:m], vn[:,:,:m])
        return loss


print("Models loaded.")

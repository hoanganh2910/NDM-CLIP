"""
gadms.py — GA-DMS GASS + MTP loss components.

GASS implementation faithfully replicates _ga_dms_ref/model/build.py:
  - _attn_1head()           ← attention_layer() with num_heads=1
  - _sim_qk_lnd()           ← sim_qk() in LND format
  - _enhanced_sim_qk_lnd()  ← enhanced_sim_qk() with multi-scale pooling
  - GASS_Calculator._encode_text_dense()    ← clip_encode_text_dense()
  - GASS_Calculator._grad_eclip_batched()   ← grad_eclip_enhanced_batched()
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    _CLIP_EOT_ID,
    _CLIP_MASK_ID,
    _CLIP_PAD_ID,
    _CLIP_SOT_ID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (match reference top-level functions)
# ─────────────────────────────────────────────────────────────────────────────

def _attn_1head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-head attention — matches reference attention_layer(num_heads=1).

    Args:
        q, k, v: [seq_len, batch, dim]  (LND format)
        causal_mask: [1, seq_len, seq_len] additive mask (−inf for future)
    Returns:
        attn_output: [seq_len, batch, dim]  BEFORE out_proj  ← used for grad
        attn_weights: [batch, seq_len, seq_len]
    """
    seq_len, bsz, dim = q.shape
    scale = float(dim) ** -0.5

    Q = q.transpose(0, 1) * scale   # [B, seq, dim]
    K = k.transpose(0, 1)            # [B, seq, dim]
    V = v.transpose(0, 1)            # [B, seq, dim]

    attn_w = torch.bmm(Q, K.transpose(1, 2))       # [B, seq, seq]
    if causal_mask is not None:
        attn_w = attn_w + causal_mask
    attn_w = F.softmax(attn_w, dim=-1)

    out = torch.bmm(attn_w, V).transpose(0, 1)     # [seq, B, dim]
    return out, attn_w                               # attn_w: [B, seq, seq]


def _pooling_lnd(features: torch.Tensor, scales=(1, 2)) -> torch.Tensor:
    """Multi-scale average-pooling on LND tensor — matches reference multi_scale_pooling."""
    seq_len, bsz, dim = features.shape
    pooled_all = []
    for scale in scales:
        if scale <= 1 or seq_len < scale:
            pooled_all.append(features)
            continue
        full = (seq_len // scale) * scale
        main = features[:full].reshape(full // scale, scale, bsz, dim).mean(dim=1)  # [full//scale, B, dim]
        if full < seq_len:
            tail = features[full:].mean(dim=0, keepdim=True)
            main = torch.cat([main, tail], dim=0)
        up = F.interpolate(
            main.permute(1, 2, 0),          # [B, dim, reduced_seq]
            size=seq_len, mode="linear", align_corners=False,
        ).permute(2, 0, 1)                  # [seq, B, dim]
        pooled_all.append(up)
    return torch.stack(pooled_all, dim=0).mean(dim=0)  # [seq, B, dim]


def _sim_qk_lnd(q: torch.Tensor, k: torch.Tensor, eos_idx: torch.Tensor) -> torch.Tensor:
    """Cosine sim(EOS_query, all_keys) — matches reference sim_qk().

    q, k: [seq_len, batch, dim]
    eos_idx: [batch]
    Returns cosine_qk: [batch, seq_len], min-max normalised per sample.
    """
    q_cls = F.normalize(q[eos_idx, :, :], dim=-1)              # [B, B, dim]
    q_diag = torch.diagonal(q_cls, dim1=0, dim2=1).T.unsqueeze(1)  # [B, 1, dim]
    k_norm = F.normalize(k, dim=-1).permute(1, 0, 2)           # [B, seq, dim]
    cosine = (q_diag * k_norm).sum(-1)                          # [B, seq]
    c_min = cosine.min(dim=1, keepdim=True)[0]
    c_max = cosine.max(dim=1, keepdim=True)[0]
    return (cosine - c_min) / (c_max - c_min + 1e-8)


def _enhanced_sim_qk_lnd(
    q: torch.Tensor, k: torch.Tensor, eos_idx: torch.Tensor, scales=(1, 2)
) -> torch.Tensor:
    """Multi-scale version of _sim_qk_lnd — matches reference enhanced_sim_qk()."""
    q_ms = _pooling_lnd(q, scales)
    k_ms = _pooling_lnd(k, scales)
    return _sim_qk_lnd(q_ms, k_ms, eos_idx)


# ─────────────────────────────────────────────────────────────────────────────
# GASS_Calculator
# ─────────────────────────────────────────────────────────────────────────────

class GASS_Calculator(nn.Module):
    def __init__(self, n_layers: int = 8):
        super().__init__()
        self.n_layers = n_layers  # how many last text-encoder layers to re-forward

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _eos_indices(input_ids: torch.Tensor) -> torch.Tensor:
        has_eot = (input_ids == _CLIP_EOT_ID).any(dim=1)
        first_eot = torch.argmax((input_ids == _CLIP_EOT_ID).to(torch.int64), dim=1)
        fallback = (input_ids != _CLIP_PAD_ID).sum(dim=1).clamp(min=1) - 1
        return torch.where(has_eot, first_eot, fallback)

    @staticmethod
    def _content_mask(input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        has_eot = (input_ids == _CLIP_EOT_ID).any(dim=1)
        first_eot = torch.argmax((input_ids == _CLIP_EOT_ID).to(torch.int64), dim=1)
        first_eot = torch.where(has_eot, first_eot, torch.full_like(first_eot, seq_len))
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        in_sentence = pos < first_eot.unsqueeze(1)
        valid_special = (
            (input_ids != _CLIP_PAD_ID)
            & (input_ids != _CLIP_SOT_ID)
            & (input_ids != _CLIP_EOT_ID)
        )
        return in_sentence & valid_special

    # ── Core: re-forward last N layers with 1-head attention ─────────────────

    @torch.enable_grad()
    def _encode_text_dense(
        self,
        model,                          # HuggingFace CLIPModel
        input_ids: torch.Tensor,        # [B, seq]
        attn_mask: torch.Tensor,        # [B, seq]
        n_layers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List, List, List, List]:
        """Re-forward last n_layers using 1-head attention.

        Mirrors clip_encode_text_dense() from _ga_dms_ref/model/build.py.

        Returns:
            global_txt : [B, dim]  normalised EOS feature (attached to grad graph)
            eos_idx    : [B]
            qs, ks, vs : lists of [seq, B, dim] per re-forwarded layer
            attn_outs  : list of [seq, B, dim]  BEFORE out_proj  ← grad target
            attns      : list of [B, seq, seq]  attention weights
        """
        n = n_layers or self.n_layers
        text_model = model.text_model
        layers = text_model.encoder.layers
        start = max(0, len(layers) - n)

        # ── Step 1: normal forward up to layer `start` (no grad needed) ──────
        with torch.no_grad():
            full_out = text_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states[i] = output of embedding/layer i-1; [start] = input to layer `start`
        x = full_out.hidden_states[start].detach()          # [B, seq, dim]

        # ── Step 2: causal mask only — match reference build_attention_mask() ─────────
        seq_len = x.size(1)
        cm = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype)
        cm = torch.triu(cm, diagonal=1).unsqueeze(0)        # [1, seq, seq]

        # ── Step 3: re-forward last n layers with 1-head attention ───────────
        x = x.permute(1, 0, 2)                              # LND: [seq, B, dim]
        qs, ks, vs, attn_outs, attns = [], [], [], [], []

        for layer in layers[start:]:
            x_in = x
            # LayerNorm (HF uses NLD internally)
            x_ln = layer.layer_norm1(x_in.permute(1, 0, 2)).permute(1, 0, 2)  # [seq, B, dim]

            # Q, K, V  (HF proj: NLD → NLD, convert to LND)
            q = layer.self_attn.q_proj(x_ln.permute(1, 0, 2)).permute(1, 0, 2)
            k = layer.self_attn.k_proj(x_ln.permute(1, 0, 2)).permute(1, 0, 2)
            v = layer.self_attn.v_proj(x_ln.permute(1, 0, 2)).permute(1, 0, 2)

            # 1-head attention — attn_out is BEFORE out_proj (matches reference)
            attn_out, attn_w = _attn_1head(q, k, v, causal_mask=cm)

            # out_proj + residual
            after_proj = layer.self_attn.out_proj(attn_out.permute(1, 0, 2)).permute(1, 0, 2)
            x = x_in + after_proj

            # MLP + residual
            x = x + layer.mlp(layer.layer_norm2(x.permute(1, 0, 2))).permute(1, 0, 2)

            qs.append(q)
            ks.append(k)
            vs.append(v)
            attn_outs.append(attn_out)   # [seq, B, dim], still in grad graph
            attns.append(attn_w)         # [B, seq, seq]

        # ── Step 4: final norm + projection ──────────────────────────────────
        x_nld = text_model.final_layer_norm(x.permute(1, 0, 2))  # [B, seq, dim]
        eos_idx = self._eos_indices(input_ids)
        b_idx = torch.arange(x_nld.size(0), device=x_nld.device)
        feat = model.text_projection(x_nld[b_idx, eos_idx])
        global_txt = F.normalize(feat.float(), p=2, dim=-1)

        return global_txt, eos_idx, qs, ks, vs, attn_outs, attns

    # ── Core: gradient-based saliency (matches grad_eclip_enhanced_batched) ──

    def _grad_eclip_batched(
        self,
        sim_it: torch.Tensor,       # [B, B] similarity matrix
        qs: List, ks: List, vs: List,
        attn_outs: List,
        attns: List,
        eos_idx: torch.Tensor,      # [B]
        valid: torch.Tensor,        # [B, seq]  content mask
        scales: Tuple = (1, 2),
    ) -> torch.Tensor:
        """Gradient-based saliency — matches grad_eclip_enhanced_batched()."""
        n_layers = len(qs)
        bsz, seq_len = valid.shape
        device = sim_it.device

        grad_outputs = torch.eye(sim_it.size(-1), device=device, dtype=sim_it.dtype)
        tmp_maps = []

        for i, (q, k, v, attn_out, attn_w) in enumerate(zip(qs, ks, vs, attn_outs, attns)):
            retain = (i < n_layers - 1)
            grad = torch.autograd.grad(
                sim_it, attn_out,
                grad_outputs=grad_outputs,
                retain_graph=retain,
                allow_unused=True,
            )[0]                                                    # [seq, B, dim]

            if grad is None:
                tmp_maps.append(torch.zeros(bsz, seq_len, device=device))
                continue

            # grad at EOS position per batch item
            grad_cls = grad[eos_idx, :, :]                          # [B, B, dim]
            grad_diag = torch.diagonal(grad_cls, dim1=0, dim2=1).T.unsqueeze(1)  # [B, 1, dim]

            # multi-scale cosine similarity
            cosine_qk = _enhanced_sim_qk_lnd(q, k, eos_idx, scales)  # [B, seq]

            v_perm = v.permute(1, 0, 2)                             # [B, seq, dim]
            importance_raw = (grad_diag * v_perm * cosine_qk.unsqueeze(-1)).sum(-1)  # [B, seq]
            importance_masked = importance_raw * valid.float()

            # attention modulation from EOS row (matches reference)
            eos_attn = attn_w[torch.arange(bsz, device=device), eos_idx, :]  # [B, seq]
            attn_masked = eos_attn * valid.float()
            attn_norm = importance_raw.sum(dim=1, keepdim=True)
            attn_w_norm = attn_masked / (attn_norm + 1e-8)

            layer_score = importance_masked * attn_w_norm
            tmp_maps.append(layer_score)

        if not tmp_maps:
            return torch.zeros(bsz, seq_len, device=device)

        # sum across layers + relu — same as reference stacked_maps.sum(0) then relu_
        saliency = F.relu_(torch.stack(tmp_maps, dim=0).sum(0))     # [B, seq]

        # Apply valid mask (ref: final_mask = token_indices > 0 & < eos_position)
        saliency = saliency * valid.float()

        # Per-sentence min-max normalization
        s_min = saliency.min(dim=1, keepdim=True)[0]
        s_max = saliency.max(dim=1, keepdim=True)[0]
        mask = (s_max > s_min).float()
        saliency = torch.where(
            mask > 0,
            (saliency - s_min) / (s_max - s_min + 1e-8),
            torch.zeros_like(saliency)
        )

        # Mẹo: Áp dụng Căn bậc 2 (Square Root Smoothing) để dàn đều điểm số.
        # Giúp các từ khóa phụ (như màu sắc, áo phụ) vốn bị đè bẹp xuống 0.01 -> 0.05
        # được kéo lên mức 0.1 -> 0.22, đủ để vượt qua ngưỡng Sigmoid Threshold (0.1) của MTP.
        saliency = saliency ** 0.5

        return (saliency * valid.float()).detach()

    # ── Public entry point ────────────────────────────────────────────────────

    def compute_gass(
        self,
        model,                          # HuggingFace CLIPModel (passed from tbps_loss.py)
        global_img: torch.Tensor,       # [B, dim]
        input_ids: torch.Tensor,        # [B, seq]
        attn_mask: torch.Tensor,        # [B, seq]
        scales: Tuple = (1, 2),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full GASS pipeline matching GA-DMS reference.

        Returns:
            saliency  : [B, seq]  GASS scores (detached)
            eos_idx   : [B]
            global_txt: [B, dim]  text features from 1-head re-forward (for SDM/denoised)
        """
        # Ref repo: image features NOT detached — stronger gradient signal for GASS
        # Safe: torch.autograd.grad only computes w.r.t. attn_outs (re-forward graph),
        # it does NOT destroy the main training graph that global_img belongs to.
        img_norm = F.normalize(global_img.float(), p=2, dim=-1)
        valid = self._content_mask(input_ids)

        # Re-forward last N layers with 1-head attention
        global_txt, eos_idx, qs, ks, vs, attn_outs, attns = self._encode_text_dense(
            model, input_ids, attn_mask
        )

        # Similarity matrix [B, B]
        sim_it = img_norm @ global_txt.t()

        # Gradient-based saliency
        saliency = self._grad_eclip_batched(
            sim_it, qs, ks, vs, attn_outs, attns, eos_idx, valid, scales
        )

        return saliency.detach(), eos_idx, global_txt


# ─────────────────────────────────────────────────────────────────────────────
# GADMS_MTP
# ─────────────────────────────────────────────────────────────────────────────

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # PyTorch MHA with batch_first=True to match our NLD pipeline
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.0)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # norm_first=True implementation matching CLIP
        x_norm = self.ln_1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class GADMS_MTP(nn.Module):
    def __init__(self, embed_dim: int = 512, vocab_size: int = 49408, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
        
        # Custom Transformer matching CLIP
        self.transformer_blocks = nn.Sequential(*[
            ResidualAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.ln_pre_t = nn.LayerNorm(embed_dim)
        self.ln_pre_i = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)
        
        # MLM Head
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Exact weight initialization from _ga_dms_ref/model/build.py"""
        scale = self.embed_dim ** -0.5
        proj_std = scale * ((2 * self.num_layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.embed_dim) ** -0.5

        # Init Custom Transformer blocks
        for block in self.transformer_blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[2].weight, std=proj_std)

        # Init Cross Attention
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # Init MLM Head
        nn.init.normal_(self.predictor[0].weight, std=fc_std)
        nn.init.normal_(self.predictor[3].weight, std=proj_std)

    def forward(
        self,
        patch_img: torch.Tensor,
        token_txt: torch.Tensor,
        mtp_labels: torch.Tensor,   # ref: mlm_labels — original ids at masked pos, 0 elsewhere
        input_ids: torch.Tensor,    # kept for interface compatibility, not used directly
    ) -> torch.Tensor:
        fused, _ = self.cross_attn(
            self.ln_pre_t(token_txt),
            self.ln_pre_i(patch_img),
            self.ln_pre_i(patch_img),
            need_weights=False,
        )
        fused = self.transformer_blocks(fused)
        fused = self.ln_post(fused)
        logits = self.predictor(fused)

        # ref: objectives.compute_mlm(scores, mlm_labels)
        # mlm_labels has original token id at masked positions, 0 at unmasked → ignore_index=0
        labels = mtp_labels.detach().long()
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
        )

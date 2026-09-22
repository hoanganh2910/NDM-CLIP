"""
model.py — TBPSLightning (Fixed & Canonical)

Fixes so với phiên bản cũ:
  1. gradient_checkpointing chỉ bật khi cả mtp=0 VÀ ss=0 (tránh block grad flow)
  2. simclr_mlp khởi tạo có điều kiện (chỉ khi ss_w > 0)
  3. GASS cache update dùng self.trainer.datamodule thay vì getattr fragile
  4. Aug features tính 1 lần, truyền vào batch để losses.py dùng lại
  5. Log rõ ràng từng loss + contribution % + gradient norm theo group
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")

from transformers import CLIPModel, get_cosine_schedule_with_warmup, CLIPProcessor
from typing import Dict, Optional

from tbps.losses.tbps_loss import TBPSLoss
from tbps.model_utils import (
    evaluate_and_log,
    log_gass_sample,
    log_grad_chart,
    log_loss_chart,
    metric_eval,
)
from utils import (
    LEARNING_RATE, WARMUP_RATIO, MAX_EPOCHS,
    BATCH_SIZE, MODEL_NAME, WEIGHT_DECAY,
)


class TBPSLightning(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        warmup_ratio:  float = WARMUP_RATIO,
        max_epochs:    int   = MAX_EPOCHS,
        batch_size:    int   = BATCH_SIZE,
        use_sigmoid:   bool  = False,
        loss_weights:  Optional[Dict[str, float]] = None,
        gass_attn_norm_mode: str = "repo",
    ):
        super().__init__()
        self.save_hyperparameters()

        if loss_weights is None:
            raise ValueError("loss_weights must be passed from train.py and cannot be None.")
        self.loss_weights = loss_weights
        self.batch_size = batch_size

        # ── CLIP backbone ─────────────────────────────────────────────
        self.model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
        self.model.train()

        # FIX: gradient checkpointing chỉ bật khi KHÔNG dùng MTP và SS
        # (cả 2 cần grad flow qua text encoder / simclr_mlp)
        need_full_grad = (
            self.loss_weights.get("mtp", 0) > 0
            or self.loss_weights.get("ss",  0) > 0
        )
        if not need_full_grad:
            self.model.gradient_checkpointing_enable()

        self.logit_scale = self.model.logit_scale
        self.logit_bias  = nn.Parameter(torch.zeros(1))

        proj_dim = self.model.config.projection_dim

        # FIX: simclr_mlp chỉ tạo khi ss_w > 0 — tiết kiệm param
        if self.loss_weights.get("ss", 0) > 0:
            self.simclr_mlp = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            )
        else:
            self.simclr_mlp = None

        # ── Loss module ───────────────────────────────────────────────
        self.loss_fn = TBPSLoss(
            use_sigmoid=use_sigmoid,
            loss_weights=self.loss_weights,
            batch_size=self.batch_size,
        )

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)


        # ── Evaluation buffers ────────────────────────────────────────
        self.val_features  = {"img": [], "txt": [], "pid": [], "file_path": []}
        self.test_features = {"img": [], "txt": [], "pid": [], "file_path": []}

        # ── Logging history ───────────────────────────────────────────
        self._grad_groups  = [
            "vision_encoder", "text_encoder", "projections",
            "simclr_head", "mtp_head", "other",
        ]
        self._grad_history = {g: [] for g in self._grad_groups}
        self._grad_steps   = []

        self._loss_keys    = ["total", "nitc", "sdm", "sdm_paper", "ritc", "citc", "ss", "mvs", "mtp"]
        self._loss_history = {k: [] for k in self._loss_keys}
        self._loss_steps   = []
        self._gass_debug_printed = False

    @staticmethod
    def _eos_indices(input_ids: torch.Tensor) -> torch.Tensor:
        eot_id = 49407
        pad_id = 0
        has_eot = (input_ids == eot_id).any(dim=1)
        first_eot = torch.argmax((input_ids == eot_id).to(torch.int64), dim=1)
        fallback_eot = (input_ids != pad_id).sum(dim=1).clamp(min=1) - 1
        return torch.where(has_eot, first_eot, fallback_eot)


    # ================================================================
    # FEATURE EXTRACTION
    # ================================================================

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.vision_model(pixel_values)
        return self.model.visual_projection(outputs.pooler_output)

    def forward(self, batch: Dict, is_training: bool = False) -> Dict:
        # ── Vision ───────────────────────────────────────────────────
        vision_out   = self.model.vision_model(batch["images"])
        patch_embeds = self.model.visual_projection(
            vision_out.last_hidden_state   # KHÔNG bỏ [CLS] để khớp với cross_former của reference code
        )
        image_feats  = self.model.visual_projection(vision_out.pooler_output)

        # ── Text (Luôn dùng augmented caption cho MAIN graph / Contrastive Losses) ─────────
        if is_training and "caption_input_ids_aug" in batch:
            input_ids = batch["caption_input_ids_aug"]
            attn_mask = batch["caption_attention_mask_aug"]
        else:
            input_ids = batch["caption_input_ids"]
            attn_mask = batch["caption_attention_mask"]

        # Main graph không cần capture hooks (GASS đã tự capture ở pass riêng trong losses.py)
        self._capture_text_debug = False
        text_out     = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=False,
            return_dict=True,
        )
        self._capture_text_debug = False
        token_embeds = self.model.text_projection(text_out.last_hidden_state)
        eos_idx = self._eos_indices(input_ids)
        text_feats = token_embeds[
            torch.arange(token_embeds.size(0), device=token_embeds.device),
            eos_idx,
        ]

        return {
            "global_img":    image_feats.float(),
            "global_txt":    text_feats.float(),
            "patch_img":     patch_embeds.float(),
            "token_txt":     token_embeds.float(),
            "input_ids":     input_ids,
            "attn_mask":     attn_mask,
            "pooler_output": text_out.pooler_output,
        }

    # ================================================================
    # TRAINING
    # ================================================================

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        device = batch["images"].device
        feats  = self(batch, is_training=True)

        # FIX: tính aug features 1 lần trước khi gọi loss (SS + MVS dùng chung)
        if (
            self.loss_weights.get("ss",  0) > 0
            or self.loss_weights.get("mvs", 0) > 0
        ) and "aug_images1" in batch and "aug_images2" in batch:
            with torch.no_grad():
                batch["aug1_feat"] = self.get_image_features(batch["aug_images1"].to(device))
                batch["aug2_feat"] = self.get_image_features(batch["aug_images2"].to(device))

        total_loss, losses, gass_current = self.loss_fn(
            self, feats, batch, is_training=True
        )

        # ── Lưu GASS current để on_train_batch_end có thể đẩy vào dataset ───
        self._current_gass = gass_current  # [B, L] hoặc None

        # ── Logging ──────────────────────────────────────────────────
        self.log("train/loss_total", total_loss, prog_bar=True, batch_size=self.batch_size)
        self._loss_steps.append(self.global_step)
        self._loss_history["total"].append(total_loss.item())

        for k in self._loss_keys:
            if k == "total":
                continue
            v = losses.get(k, torch.tensor(0.0, device=device))
            if isinstance(v, torch.Tensor):
                val = v.item()
                self._loss_history[k].append(val)
                self.log(f"Loss_Raw/{k}", v, batch_size=self.batch_size)
                if total_loss.item() > 0 and val != 0:
                    pct = self.loss_weights.get(k, 1.0) * v / total_loss.detach() * 100.0
                    self.log(f"Contribution_Pct/{k}", pct, batch_size=self.batch_size)
            else:
                self._loss_history[k].append(0.0)

        # ── GASS sample log mỗi 50 step ──────────────────────────────
        if self.global_step % 50 == 0:
            self._log_gass_sample(feats["input_ids"], gass_current)

        return total_loss

    # ================================================================
    # CALLBACKS
    # ================================================================

    def on_train_epoch_start(self):
        pass  # DataLoader rebuild handled by reload_dataloaders_every_n_epochs=1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Two-epoch pipeline (giống ref): sau mỗi batch, lưu GASS score
        vào train_dataset để DataLoader của epoch sau dùng tạo masked IDs.
        Ref code: for idx, sim in zip(batch['image_ids'], grad_emap):
                      trainset[idx][-1] = sim.data.cpu().numpy()
        """
        if not (self.loss_weights.get("mtp", 0) > 0):
            return
        # Lấy gass_current từ outputs (cần cất vào training_step)
        gass_scores = getattr(self, "_current_gass", None)
        if gass_scores is None:
            return
        indices = batch.get("indices")
        if indices is None:
            return
        # Lấy trực tiếp từ datamodule
        try:
            train_ds = self.trainer.datamodule.train_dataset
        except Exception:
            return
        if not hasattr(train_ds, "set_gass_score"):
            return
        # Lưu score từng sample: shape [L] numpy
        for i, ds_idx in enumerate(indices.tolist()):
            score_np = gass_scores[i].cpu().float().numpy()  # [L]
            train_ds.set_gass_score(ds_idx, score_np)

    def on_after_backward(self):
        if self.global_step % 10 != 0:
            return
        groups = {g: 0.0 for g in self._grad_groups}
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            gn_sq = param.grad.detach().float().norm().item() ** 2
            if   "simclr_mlp"        in name: groups["simclr_head"]    += gn_sq
            elif "loss_fn.mtp"       in name or "loss_fn.gass" in name: groups["mtp_head"] += gn_sq
            elif "model.vision_model" in name: groups["vision_encoder"]  += gn_sq
            elif "model.text_model"  in name: groups["text_encoder"]    += gn_sq
            elif "projection"        in name or "logit" in name: groups["projections"] += gn_sq
            else:                              groups["other"]           += gn_sq

        self._grad_steps.append(self.global_step)
        for g, sq in groups.items():
            norm = sq ** 0.5
            self._grad_history.setdefault(g, []).append(norm)
            self.log(f"GradNorm/{g}", norm, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        log_grad_chart(self)
        log_loss_chart(self)
        self._grad_history = {g: [] for g in self._grad_groups}
        self._grad_steps   = []
        self._loss_history = {k: [] for k in self._loss_keys}
        self._loss_steps   = []

    # ================================================================
    # VALIDATION / TEST
    # ================================================================

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            feats = self(batch, is_training=False)
            total_loss, _, _ = self.loss_fn(self, feats, batch, is_training=False)
            self.val_features["img"].append(feats["global_img"].detach().cpu())
            self.val_features["txt"].append(feats["global_txt"].detach().cpu())
            self.val_features["pid"].append(batch["pids"].detach().cpu())
            self.val_features["file_path"].extend(batch["file_path"])
        self.log("val_loss", total_loss, prog_bar=True, batch_size=self.batch_size)
        return total_loss

    def on_validation_epoch_end(self):
        self._evaluate_and_log(self.val_features, "val")
        self.val_features = {"img": [], "txt": [], "pid": [], "file_path": []}

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            feats = self(batch, is_training=False)
            self.test_features["img"].append(feats["global_img"].detach().cpu())
            self.test_features["txt"].append(feats["global_txt"].detach().cpu())
            self.test_features["pid"].append(batch["pids"].detach().cpu())
            self.test_features["file_path"].extend(batch["file_path"])

    def on_test_epoch_end(self):
        evaluate_and_log(self, self.test_features, "test")
        self.test_features = {"img": [], "txt": [], "pid": [], "file_path": []}

    def _evaluate_and_log(self, feature_dict: Dict, stage: str):
        evaluate_and_log(self, feature_dict, stage)

    # ================================================================
    # OPTIMIZER
    # ================================================================

    def configure_optimizers(self):
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("simclr_mlp", "loss_fn.mtp", "loss_fn.gass")):
                head_params.append(param)
            else:
                backbone_params.append(param)

        base_lr = self.hparams.learning_rate
        head_lr = base_lr * 10.0  # head cần học nhanh hơn vì random-init

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": base_lr},
                {"params": head_params,     "lr": head_lr},
            ],
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ================================================================
    # METRICS
    # ================================================================

    def metric_eval(
        self,
        similarity: torch.Tensor,
        q_pids: torch.Tensor,
        g_pids: torch.Tensor,
        max_rank: int = 10,
    ) -> Dict[str, float]:
        return metric_eval(similarity, q_pids, g_pids, max_rank)

    # ================================================================
    # LOGGING HELPERS
    # ================================================================

    def _log_gass_sample(self, input_ids_batch: torch.Tensor, gass_scores: Optional[torch.Tensor]):
        log_gass_sample(self, input_ids_batch, gass_scores)

    def _log_grad_chart(self):
        log_grad_chart(self)

    def _log_loss_chart(self):
        log_loss_chart(self)


# Export constants để _log_gass_sample dùng được
_CLIP_SOT_ID = 49406
_CLIP_PAD_ID = 0

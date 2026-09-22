from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import (
    _CLIP_MASK_ID,
    _CLIP_TEXT_VOCAB_UPPER,
)
from .gadms import GADMS_MTP, GASS_Calculator
from .objectives import (
    CITCLoss,
    MVSLoss,
    NITCLoss,
    RITCLoss,
    SDMLoss,
    SDMPaperLoss,
    SelfSupervisedLoss,
)


class TBPSLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        if loss_weights is None:
            raise ValueError("loss_weights must be provided to TBPSLoss from train.py")
        self.weights = loss_weights
        self.use_sigmoid = use_sigmoid
        self.batch_size = batch_size

        print(f"\n🚀 [TBPS LOSS] Trọng số MTP: {self.weights.get('mtp', 0)}\n")

        self.nitc = NITCLoss(use_sigmoid=use_sigmoid) if self.weights.get("nitc", 0) > 0 else None
        self.sdm = SDMLoss(use_sigmoid=use_sigmoid) if self.weights.get("sdm", 0) > 0 else None
        self.sdm_paper = SDMPaperLoss(use_sigmoid=use_sigmoid) if self.weights.get("sdm_paper", 0) > 0 else None
        self.ritc = RITCLoss(use_sigmoid=use_sigmoid) if self.weights.get("ritc", 0) > 0 else None
        self.citc = CITCLoss() if self.weights.get("citc", 0) > 0 else None
        self.ss = SelfSupervisedLoss() if self.weights.get("ss", 0) > 0 else None
        self.mvs = MVSLoss(self.nitc) if self.weights.get("mvs", 0) > 0 else None

        use_mtp = self.weights.get("mtp", 0) > 0
        self.gass = GASS_Calculator() if use_mtp else None
        self.mtp = GADMS_MTP() if use_mtp else None

    def _zero(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(0.0, device=device, requires_grad=False)

    def _sample_masked_input_ids(
        self,
        input_ids: torch.Tensor,
        mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        masked_ids = input_ids.clone()
        if not mask_bool.any():
            return masked_ids

        rand = torch.rand_like(masked_ids.float())
        replace_with_mask = (rand < 0.8) & mask_bool
        replace_with_random = (rand >= 0.8) & (rand < 0.9) & mask_bool

        masked_ids[replace_with_mask] = _CLIP_MASK_ID
        if replace_with_random.any():
            random_tokens = torch.randint(
                low=1,
                high=_CLIP_TEXT_VOCAB_UPPER,
                size=(replace_with_random.sum().item(),),
                device=masked_ids.device,
                dtype=masked_ids.dtype,
            )
            masked_ids[replace_with_random] = random_tokens

        for b in range(masked_ids.size(0)):
            selected_idx = mask_bool[b].nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue
            if (masked_ids[b, selected_idx] == _CLIP_MASK_ID).any():
                continue
            masked_ids[b, selected_idx[0]] = _CLIP_MASK_ID
        return masked_ids

    def _build_mtp_mask(self, input_ids: torch.Tensor, gass_scores: torch.Tensor, mask_informative: bool) -> torch.Tensor:
        device = input_ids.device
        valid = self.gass._content_mask(input_ids)
        if mask_informative:
            probs = 0.3 / (1.0 + torch.exp(-20.0 * (gass_scores - 0.1)))
        else:
            probs = 0.2 / (1.0 + torch.exp(-20.0 * ((1.0 - gass_scores) - 0.9)))

        rand = torch.rand_like(probs)
        mask_bool = (rand < probs) & valid
        for b in range(mask_bool.size(0)):
            if mask_bool[b].any():
                continue
            valid_idx = valid[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                mask_bool[b, valid_idx[0]] = True
        return mask_bool

    @staticmethod
    def _ensure_at_least_one_mask(mask_bool: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if mask_bool.numel() == 0:
            return mask_bool
        out = mask_bool.clone()
        for b in range(out.size(0)):
            if out[b].any():
                continue
            valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            out[b, valid_idx[0]] = True
        return out

    def forward(
        self,
        model,
        student_feats_dict: Dict,
        batch: Dict,
        is_training: bool = True,
    ):
        device = student_feats_dict["global_img"].device
        losses: Dict[str, torch.Tensor] = {}
        gass_current = None

        global_img = student_feats_dict["global_img"]
        global_txt = student_feats_dict["global_txt"]

        losses["nitc"] = self.nitc(model, (global_img, global_txt), batch) if self.nitc else self._zero(device)
        losses["ritc"] = self.ritc(model, (global_img, global_txt), batch) if self.ritc else self._zero(device)
        losses["citc"] = self.citc(model, (global_img, global_txt), batch) if self.citc else self._zero(device)

        if is_training and self.gass is not None and self.mtp is not None:
            input_ids = batch["caption_input_ids"].to(device)
            attn_mask = batch["caption_attention_mask"].to(device)

            # ── Reference-faithful GASS: re-forward last N layers with 1-head attn ──
            with torch.set_grad_enabled(True):
                s_gass, eos_idx, global_txt_gass = self.gass.compute_gass(
                    model=model.model,          # HuggingFace CLIPModel
                    global_img=global_img,
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                )
            gass_current = s_gass                                   # already detached

            b_idx = torch.arange(input_ids.size(0), device=device)
            valid_mask = self.gass._content_mask(input_ids)

            # ref pattern: preferred pre-computed masks from the dataset (two-epoch pipeline)
            if (
                "noise_input_ids" not in batch
                or batch["noise_input_ids"] is None
                or "mtp_input_ids" not in batch
                or batch["mtp_input_ids"] is None
                or "mtp_labels" not in batch
                or batch["mtp_labels"] is None
            ):
                noise_mask_bool = self._build_mtp_mask(input_ids, s_gass, mask_informative=False)
                mtp_mask_bool = self._build_mtp_mask(input_ids, s_gass, mask_informative=True)
                noise_input_ids = self._sample_masked_input_ids(input_ids, noise_mask_bool)
                mtp_input_ids = self._sample_masked_input_ids(input_ids, mtp_mask_bool)
                mtp_labels = torch.where(mtp_mask_bool, input_ids, torch.zeros_like(input_ids))
            else:
                noise_input_ids = batch["noise_input_ids"].to(device)
                mtp_input_ids   = batch["mtp_input_ids"].to(device)
                mtp_labels      = batch["mtp_labels"].to(device)   # ref: mlm_labels

            # Denoised text features for SDM (standard 8-head forward)
            with torch.set_grad_enabled(True):
                text_out_denoised = model.model.text_model(
                    input_ids=noise_input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            text_tokens_denoised = model.model.text_projection(text_out_denoised.last_hidden_state)
            global_txt_denoised  = text_tokens_denoised[b_idx, eos_idx]

            losses["sdm"]       = self.sdm(model, (global_img, global_txt_denoised), batch) if self.sdm       else self._zero(device)
            losses["sdm_paper"] = self.sdm_paper(model, (global_img, global_txt_denoised), batch) if self.sdm_paper else self._zero(device)

            # MTP token forward with masked ids
            with torch.set_grad_enabled(True):
                text_out_mtp = model.model.text_model(
                    input_ids=mtp_input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            token_txt_masked = model.model.text_projection(text_out_mtp.last_hidden_state)
            losses["mtp"] = self.mtp(
                patch_img=student_feats_dict["patch_img"],
                token_txt=token_txt_masked,
                mtp_labels=mtp_labels,          # ref: mlm_labels (original ids at masked pos)
                input_ids=input_ids,
            )
        else:
            losses["sdm"]       = self.sdm(model, (global_img, global_txt), batch) if self.sdm       else self._zero(device)
            losses["sdm_paper"] = self.sdm_paper(model, (global_img, global_txt), batch) if self.sdm_paper else self._zero(device)
            losses["mtp"] = self._zero(device)

        if is_training:
            has_aug = "aug_images1" in batch and "aug_images2" in batch
            ss_w = self.weights.get("ss", 0)
            mvs_w = self.weights.get("mvs", 0)

            if (ss_w > 0 or mvs_w > 0) and has_aug:
                aug1_feat = batch.get("aug1_feat", model.get_image_features(batch["aug_images1"].to(device)))
                aug2_feat = batch.get("aug2_feat", model.get_image_features(batch["aug_images2"].to(device)))
                losses["ss"] = self.ss(model.simclr_mlp(aug1_feat), model.simclr_mlp(aug2_feat)) if self.ss else self._zero(device)
                losses["mvs"] = self.mvs(model, global_txt, aug1_feat, batch) if self.mvs else self._zero(device)
            else:
                losses["ss"] = self._zero(device)
                losses["mvs"] = self._zero(device)
        else:
            losses["ss"] = self._zero(device)
            losses["mvs"] = self._zero(device)

        total = sum(
            self.weights.get(k, 1.0) * v
            for k, v in losses.items()
            if isinstance(v, torch.Tensor)
        )
        return total, losses, gass_current

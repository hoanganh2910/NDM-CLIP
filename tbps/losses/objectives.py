from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SOFTLABEL_RATIO

from .base import BaseLoss


class NITCLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 0.07,
        epsilon: float = 1e-8,
        alpha: float = SOFTLABEL_RATIO,
        use_sigmoid: bool = False,
    ):
        super().__init__(temperature, epsilon)
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid

    def forward(self, model, student_features, batch):
        image_f_s, text_f_s = student_features
        device = image_f_s.device
        pids = batch["pids"].to(device)

        image_f_s = self.normalize_features(image_f_s)
        text_f_s = self.normalize_features(text_f_s)

        logit_scale = torch.clamp(model.logit_scale.exp().float(), max=100.0)
        logit_bias = model.logit_bias.float()

        image_f_stopped = image_f_s.detach() if self.alpha > 0 else None
        text_f_stopped = text_f_s.detach() if self.alpha > 0 else None

        if "aug_images" in batch and self.alpha > 0:
            aug_f = self.normalize_features(
                model.get_image_features(batch["aug_images"].to(device))
            )
            image_f_stopped = aug_f.detach()

        sim_targets = self.prepare_sim_targets(pids, device, self.use_sigmoid)

        logits_i2t = logit_scale * (image_f_s @ text_f_s.t()) + logit_bias
        logits_t2i = logit_scale * (text_f_s @ image_f_s.t()) + logit_bias

        if self.alpha > 0 and image_f_stopped is not None and text_f_stopped is not None:
            logits_i2t_stop = logit_scale * (image_f_stopped @ text_f_stopped.t()) + logit_bias
            logits_t2i_stop = logit_scale * (text_f_stopped @ image_f_stopped.t()) + logit_bias
            soft_i2t = F.softmax(logits_i2t_stop, dim=1)
            soft_t2i = F.softmax(logits_t2i_stop, dim=1)
            targets_i2t = self.alpha * soft_i2t + (1 - self.alpha) * sim_targets
            targets_t2i = self.alpha * soft_t2i + (1 - self.alpha) * sim_targets
        else:
            targets_i2t = sim_targets
            targets_t2i = sim_targets

        if self.use_sigmoid:
            loss_i2t = F.binary_cross_entropy_with_logits(logits_i2t, targets_i2t)
            loss_t2i = F.binary_cross_entropy_with_logits(logits_t2i, targets_t2i)
        else:
            loss_i2t = -(F.log_softmax(logits_i2t, dim=1) * targets_i2t).sum(1).mean()
            loss_t2i = -(F.log_softmax(logits_t2i, dim=1) * targets_t2i).sum(1).mean()

        return 0.5 * (loss_i2t + loss_t2i)


class SDMLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 0.1,
        epsilon: float = 1e-2,
        use_sigmoid: bool = False,
    ):
        super().__init__(temperature, epsilon)
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        model: nn.Module,
        student_features: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict,
    ) -> torch.Tensor:
        image_f_s, text_f_s = student_features
        device = image_f_s.device
        pids = batch["pids"].to(device)

        image_f_s = self.normalize_features(image_f_s)
        text_f_s = self.normalize_features(text_f_s)

        sim_targets = self.prepare_sim_targets(pids, device, self.use_sigmoid)
        logit_scale = torch.clamp(model.logit_scale.exp().float(), max=100.0)
        logit_bias = model.logit_bias.float()

        logits_i2t = logit_scale * (image_f_s @ text_f_s.t()) + logit_bias
        logits_t2i = logit_scale * (text_f_s @ image_f_s.t()) + logit_bias

        q = sim_targets + self.epsilon
        q = q / q.sum(dim=1, keepdim=True)

        log_q_i2t = F.log_softmax(logits_i2t, dim=1)
        log_q_t2i = F.log_softmax(logits_t2i, dim=1)
        loss_i2t = F.kl_div(input=log_q_i2t, target=q, reduction="batchmean")
        loss_t2i = F.kl_div(input=log_q_t2i, target=q, reduction="batchmean")

        return 0.5 * (loss_i2t + loss_t2i)


class SDMPaperLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 0.1,
        epsilon: float = 1e-8,
        use_sigmoid: bool = False,
    ):
        super().__init__(temperature, epsilon)
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        model: nn.Module,
        student_features: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict,
    ) -> torch.Tensor:
        image_f_s, text_f_s = student_features
        device = image_f_s.device
        pids = batch["pids"].to(device)

        image_f_s = self.normalize_features(image_f_s)
        text_f_s = self.normalize_features(text_f_s)

        sim_targets = self.prepare_sim_targets(pids, device, self.use_sigmoid)
        # Reference: labels_distribute = labels / labels.sum(dim=1)
        # Using keepdim=True to avoid broadcast ambiguity, mathematically identical for symmetric labels
        labels_distribute = sim_targets / sim_targets.sum(dim=1, keepdim=True)

        logit_scale = torch.clamp(model.logit_scale.exp().float(), max=100.0)
        logit_bias = model.logit_bias.float()
        
        logits_i2t = logit_scale * (image_f_s @ text_f_s.t()) + logit_bias
        logits_t2i = logit_scale * (text_f_s @ image_f_s.t()) + logit_bias

        # Reference: i2t_loss = i2t_pred * (F.log_softmax(...) - torch.log(labels_distribute + epsilon))
        i2t_pred = F.softmax(logits_i2t, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(logits_i2t, dim=1) - torch.log(labels_distribute + self.epsilon))
        
        t2i_pred = F.softmax(logits_t2i, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(logits_t2i, dim=1) - torch.log(labels_distribute + self.epsilon))

        # Reference: loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        # Note: No 0.5 multiplier!
        loss_i2t = torch.mean(torch.sum(i2t_loss, dim=1))
        loss_t2i = torch.mean(torch.sum(t2i_loss, dim=1))
        return loss_i2t + loss_t2i


class RITCLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 0.1,
        epsilon: float = 1e-2,
        use_sigmoid: bool = False,
    ):
        super().__init__(temperature, epsilon)
        self.use_sigmoid = use_sigmoid

    def forward(
        self,
        model: nn.Module,
        student_features: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict,
    ) -> torch.Tensor:
        image_f_s, text_f_s = student_features
        device = image_f_s.device
        pids = batch["pids"].to(device)

        image_f_s = self.normalize_features(image_f_s)
        text_f_s = self.normalize_features(text_f_s)

        sim_targets = self.prepare_sim_targets(pids, device, self.use_sigmoid)
        logit_scale = torch.clamp(model.logit_scale.exp().float(), max=100.0)
        logit_bias = model.logit_bias.float()

        logits_i2t = logit_scale * (image_f_s @ text_f_s.t()) + logit_bias
        logits_t2i = logit_scale * (text_f_s @ image_f_s.t()) + logit_bias

        log_p_i2t = F.log_softmax(logits_i2t, dim=1)
        log_p_t2i = F.log_softmax(logits_t2i, dim=1)

        q = sim_targets + self.epsilon
        q = q / q.sum(dim=1, keepdim=True)
        log_q = torch.log(q)

        kl_i2t = F.kl_div(input=log_q, target=log_p_i2t, reduction="batchmean", log_target=True)
        kl_t2i = F.kl_div(input=log_q, target=log_p_t2i, reduction="batchmean", log_target=True)
        return 0.5 * (kl_i2t + kl_t2i)


class CITCLoss(BaseLoss):
    def __init__(self, inmodal_weight: float = 0.25, intermodal_weight: float = 0.25):
        super().__init__()
        self.inmodal_weight = inmodal_weight
        self.intermodal_weight = intermodal_weight

    def forward(self, model, student_features, batch):
        image_f_s, text_f_s = student_features
        image_f_s = self.normalize_features(image_f_s)
        text_f_s = self.normalize_features(text_f_s)

        sim_ii = image_f_s @ image_f_s.t()
        sim_tt = text_f_s @ text_f_s.t()
        sim_it = image_f_s @ text_f_s.t()
        sim_ti = sim_it.t()

        inmodal_loss = ((sim_ii - sim_tt) ** 2).mean()
        intermodal_loss = ((sim_it - sim_ti) ** 2).mean()
        return self.inmodal_weight * inmodal_loss + self.intermodal_weight * intermodal_loss


class SelfSupervisedLoss(BaseLoss):
    def __init__(self, temperature_ss: float = 0.1):
        super().__init__(temperature=temperature_ss)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = self.normalize_features(z1)
        z2 = self.normalize_features(z2)
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / self.temperature

        n_all = z.shape[0]
        diag_idx = torch.arange(n_all, device=z.device)
        sim[diag_idx, diag_idx] = -1e4

        n = z1.shape[0]
        positives = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)
        logsumexp = torch.logsumexp(sim, dim=1)
        return -(positives - logsumexp).mean()


class MVSLoss(nn.Module):
    def __init__(self, nitc_module: Optional[NITCLoss]):
        super().__init__()
        self.nitc = nitc_module

    def forward(self, model, text_feats, aug1_feat, batch):
        if self.nitc is None:
            return torch.tensor(0.0, device=text_feats.device)
        return self.nitc(model, (aug1_feat, text_feats), batch)

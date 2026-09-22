import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LOSS_WEIGHTS = {
    "nitc": 1.0,
    "sdm": 1.0,
    "sdm_paper": 0.0,
    "ritc": 0.0,
    "citc": 0.1,
    "ss": 0.4,
    "mvs": 1.0,
    "mtp": 0.5,
}

_CLIP_SOT_ID = 49406
_CLIP_EOT_ID = 49407
_CLIP_PAD_ID = 0
_CLIP_MASK_ID = 49405
_CLIP_TEXT_VOCAB_UPPER = 49405


class BaseLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, epsilon: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def prepare_sim_targets(
        self,
        pids: torch.Tensor,
        device: torch.device,
        use_sigmoid: bool = False,
    ) -> torch.Tensor:
        sim_targets = (pids.view(-1, 1) == pids.view(1, -1)).float().to(device)
        if use_sigmoid:
            return -torch.ones_like(sim_targets) + 2 * sim_targets
        return sim_targets / sim_targets.sum(dim=1, keepdim=True).clamp(min=1e-8)

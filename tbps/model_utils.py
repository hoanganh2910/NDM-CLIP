from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb


def metric_eval(
    similarity: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    max_rank: int = 10,
) -> Dict[str, float]:
    def rank_metrics(sim: torch.Tensor, query_pids: torch.Tensor, gallery_pids: torch.Tensor) -> Dict[str, float]:
        rank_k = min(max_rank, sim.size(1))
        indices = torch.argsort(sim.detach().cpu(), dim=1, descending=True)
        pred_labels = gallery_pids[indices]
        matches = pred_labels.eq(query_pids.view(-1, 1).cpu())

        cmc = matches[:, :rank_k].cumsum(1)
        cmc[cmc > 1] = 1
        cmc = cmc.float().mean(0) * 100.0

        num_rel = matches.sum(1)
        tmp_cmc = matches.cumsum(1)
        precisions = torch.stack(
            [tmp_cmc[:, i] / float(i + 1) for i in range(tmp_cmc.size(1))],
            dim=1,
        ) * matches
        ap = precisions.sum(1) / num_rel
        map_score = ap.mean() * 100.0

        inp = [
            tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
            for i, match_row in enumerate(matches)
        ]
        minp_score = torch.cat(inp).mean() * 100.0

        return {
            "R@1": float(cmc[0].item()) if rank_k >= 1 else 0.0,
            "R@5": float(cmc[min(4, rank_k - 1)].item()) if rank_k >= 1 else 0.0,
            "R@10": float(cmc[min(9, rank_k - 1)].item()) if rank_k >= 1 else 0.0,
            "mAP": float(map_score.item()),
            "mINP": float(minp_score.item()),
        }

    q_pids = q_pids.view(-1).cpu()
    g_pids = g_pids.view(-1).cpu()

    t2i = rank_metrics(similarity, query_pids=q_pids, gallery_pids=g_pids)

    return {
        "R@1_t2i": t2i["R@1"],
        "R@5_t2i": t2i["R@5"],
        "R@10_t2i": t2i["R@10"],
        "mAP_t2i": t2i["mAP"],
        "mINP_t2i": t2i["mINP"],
    }


def evaluate_and_log(module, feature_dict: Dict, stage: str):
    if not feature_dict["img"]:
        return
    txt_feats = F.normalize(torch.cat(feature_dict["txt"], 0).to(module.device).float(), p=2, dim=1)
    txt_pids = torch.cat(feature_dict["pid"], 0).to(module.device)

    all_img_feats = torch.cat(feature_dict["img"], 0).float()
    all_img_pids = torch.cat(feature_dict["pid"], 0)
    all_paths = feature_dict["file_path"]

    unique_img_feats = []
    unique_img_pids = []
    seen_paths = set()
    for i, path in enumerate(all_paths):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_img_feats.append(all_img_feats[i])
        unique_img_pids.append(all_img_pids[i])

    img_feats = F.normalize(torch.stack(unique_img_feats, 0).to(module.device), p=2, dim=1)
    img_pids = torch.stack(unique_img_pids, 0).to(module.device)

    logit_scale = torch.clamp(module.logit_scale.exp().float(), max=100.0)
    similarity = logit_scale * (txt_feats @ img_feats.t()) + module.logit_bias.float()

    metrics = metric_eval(similarity, txt_pids, img_pids)
    module.log_dict({f"{stage}/{k}": v for k, v in metrics.items()}, prog_bar=True, sync_dist=True)

    print(
        f"\n{'='*52}\n"
        f"  queries={txt_feats.size(0)}  gallery={img_feats.size(0)}\n"
        f"  {stage.upper()} — Epoch {module.current_epoch}\n"
        f"{'─'*52}\n"
        f"  t2i  R@1 {metrics['R@1_t2i']:6.2f}  R@5 {metrics['R@5_t2i']:6.2f}"
        f"  R@10 {metrics['R@10_t2i']:6.2f}  mAP {metrics['mAP_t2i']:6.2f}"
        f"  mINP {metrics['mINP_t2i']:6.2f}\n"
        f"{'='*52}"
    )


def log_gass_sample(module, input_ids_batch: torch.Tensor, gass_scores: Optional[torch.Tensor]):
    if gass_scores is None:
        return
    input_ids = input_ids_batch[0]
    s_scores = gass_scores[0].detach().cpu().numpy()
    tokenizer = module.processor.tokenizer

    tokens_info = []
    valid_scores = []
    for i, tid in enumerate(input_ids.tolist()):
        if tid in (49406, 0):
            continue
        if tid == 49407:
            break
        tok = tokenizer.decode([tid]).strip()
        if tok:
            sc = float(s_scores[i])
            valid_scores.append(sc)
            tokens_info.append(f"{tok}({sc:.3f})")

    if not valid_scores:
        return

    display = " | ".join(tokens_info)
    print(
        f"\n{'='*60}\n"
        f"[GASS] Ep {module.current_epoch} Step {module.global_step}\n"
        f"  Tokens: {display}\n"
        f"  Max: {max(valid_scores):.4f}  Min: {min(valid_scores):.4f}\n"
        f"{'='*60}"
    )
    if hasattr(module.logger, "experiment") and hasattr(module.logger.experiment, "log"):
        module.logger.experiment.log(
            {"train/gass_tracker": display, "global_step": module.global_step}
        )


def log_grad_chart(module):
    if not module._grad_steps:
        return
    colors = {
        "vision_encoder": "#1f77b4",
        "text_encoder": "#ff7f0e",
        "projections": "#2ca02c",
        "simclr_head": "#9467bd",
        "mtp_head": "#d62728",
        "other": "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    for group in module._grad_groups:
        vals = module._grad_history.get(group, [])
        if any(v > 0 for v in vals):
            ax.plot(module._grad_steps, vals, label=group, color=colors.get(group), lw=1.2, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Gradient Norm — Epoch {module.current_epoch}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    if hasattr(module.logger, "experiment"):
        module.logger.experiment.log({"GradNorm_Chart": wandb.Image(fig)}, step=module.global_step)
    plt.close(fig)


def log_loss_chart(module):
    if not module._loss_steps:
        return
    colors = {
        "total": "#000000",
        "nitc": "#1f77b4",
        "sdm": "#ff7f0e",
        "sdm_paper": "#8c564b",
        "ritc": "#7f7f7f",
        "citc": "#2ca02c",
        "id": "#bcbd22",
        "ss": "#17becf",
        "mvs": "#e377c2",
        "mtp": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in module._loss_keys:
        vals = module._loss_history.get(key, [])
        if any(v > 0 for v in vals):
            ax.plot(
                module._loss_steps,
                vals,
                label=key,
                color=colors.get(key, "#000"),
                lw=1.5 if key == "total" else 1.0,
                alpha=0.9,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss (log)")
    ax.set_title(f"All Losses — Epoch {module.current_epoch}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    if hasattr(module.logger, "experiment"):
        module.logger.experiment.log({"Loss_Chart": wandb.Image(fig)}, step=module.global_step)
    plt.close(fig)

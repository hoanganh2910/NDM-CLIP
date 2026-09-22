import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT_DIR = "/home/ducpv/hoaganhl/tbps_project"
DEFAULT_CHECKPOINT = "/home/ducpv/hoaganhl/tbps_project/checkpoints/best-r1-epoch13.ckpt"
DEFAULT_JSON = "/home/ducpv/hoaganhl/tbps_project/src_VN3K2/vn3k_preproc_texts_final.json"
DEFAULT_OUTPUT = "/home/ducpv/hoaganhl/tbps_project/src_VN3K2/figure_ndm_clip_2samples.jpg"
MAX_TEXT_LENGTH = 77

DEFAULT_LOSS_CONFIG = {
    "nitc": 1.0,
    "sdm": 0.0,
    "sdm_paper": 1.0,
    "ritc": 0.0,
    "citc": 0.0,
    "ss": 0.0,
    "mvs": 0.0,
    "mtp": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a paper-style NDM-CLIP visualization without NAM comparison."
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Dataset root directory.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to a Lightning .ckpt file.")
    parser.add_argument("--json", default=DEFAULT_JSON, help="Preprocessed JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="all", help="Preferred split: train/val/test/all.")
    parser.add_argument("--model-name", default="NDM-CLIP", help="Column label for your model.")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Use a specific sample by file path substring. Can be passed multiple times.",
    )
    return parser.parse_args()


def find_json(root_dir, explicit_path):
    path = Path(explicit_path)
    if not path.is_absolute():
        path = Path(root_dir) / path
    if path.exists():
        return path
    raise FileNotFoundError(f"JSON not found: {path}")


def get_caption(sample):
    if isinstance(sample.get("caption"), str):
        return sample["caption"]
    if isinstance(sample.get("caption"), list) and sample["caption"]:
        return sample["caption"][0]
    if isinstance(sample.get("captions"), list) and sample["captions"]:
        return sample["captions"][0]
    return ""


def get_file_path(sample):
    return sample.get("file_path") or sample.get("img_path") or sample.get("image") or sample.get("filename")


def resolve_image_path(root_dir, file_path):
    file_path = str(file_path).replace("\\", os.sep).replace("/", os.sep)
    candidates = [
        Path(root_dir) / file_path,
        Path(root_dir) / "images" / file_path,
        Path(root_dir) / "imgs" / file_path,
        REPO_ROOT / file_path,
        REPO_ROOT / "images" / file_path,
        REPO_ROOT / "imgs" / file_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def select_samples(items, root_dir, split, num_samples, seed, sample_filters):
    usable = []
    seen_paths = set()
    for item in items:
        file_path = get_file_path(item)
        caption = get_caption(item)
        if not file_path or not caption:
            continue
        image_path = resolve_image_path(root_dir, file_path)
        if image_path is None:
            continue
        if split != "all":
            item_split = str(item.get("split", "")).lower()
            if item_split and item_split != split.lower():
                continue
        normalized_path = str(file_path).lower().replace("\\", "/")
        if not sample_filters and normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        usable.append((item, image_path))

    if sample_filters:
        chosen = []
        for pattern in sample_filters:
            pattern_norm = pattern.lower().replace("\\", "/")
            match = next(
                (
                    pair
                    for pair in usable
                    if pattern_norm in str(get_file_path(pair[0])).lower().replace("\\", "/")
                ),
                None,
            )
            if match is None:
                raise ValueError(f"No sample matched: {pattern}")
            chosen.append(match)
        return chosen

    if not usable:
        raise ValueError("No usable samples found. Check root-dir and image folder names.")

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    return rng.sample(usable, min(num_samples, len(usable)))


def hook_vision_features(model):
    vision_features = {}

    def forward_hook(_module, _input, output):
        vision_features["acts"] = output[0] if isinstance(output, tuple) else output

    def backward_hook(_module, _grad_in, grad_out):
        vision_features["grads"] = grad_out[0]

    target_layer = model.model.vision_model.encoder.layers[-2]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    return vision_features


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def clean_token(token):
    token = token.replace("</w>", "")
    token = token.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
    token = token.replace("Ġ", " ").replace("Ä ", " ")
    return token


import matplotlib.patches as patches

# ── CHANGED: larger fonts, tighter left margin, cleaner box style ──────────
def draw_highlighted_text(ax, tokens, scores, start_y=0.50, max_chars=40,
                          fontsize=20, char_width=0.0240, line_height=0.150, model_name="NDM-CLIP"):
    """
    Render token-highlighted text inside *ax*.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add model name prefix
    prefix = f"{model_name}: "
    ax.text(
        0.015,
        start_y,
        prefix,
        fontsize=fontsize,
        fontweight="bold",
        family="serif",
        va="top",
        transform=ax.transAxes,
    )
    
    x_pos = 0.34
    y_pos = start_y
    current_chars = int(x_pos / char_width)
    max_x = 0.0

    for token, score in zip(tokens, scores):
        word = clean_token(token)
        if not word.strip():
            continue
        word_len = len(word)
        if current_chars + word_len + 1 > max_chars:
            x_pos = 0.015
            y_pos -= line_height
            current_chars = 0

        score_val = min(max(float(score), 0.0), 1.0)
        alpha = min(score_val ** 0.5 + 0.10, 1.0) if score_val > 0 else 0.05

        ax.text(
            x_pos,
            y_pos,
            word,
            fontsize=fontsize,
            family="monospace",
            fontweight="bold",
            va="top",
            bbox={
                "facecolor": (0.0, 0.9, 0.0, alpha),
                "edgecolor": (0.0, 0.6, 0.0, min(alpha + 0.15, 1.0)),   # subtle green border
                "linewidth": 0.5,
                "pad": 1.5,
            },
            transform=ax.transAxes,
        )
        advance = word_len + 1.2
        new_x = x_pos + advance * char_width
        max_x = max(max_x, new_x)
        
        x_pos = new_x
        current_chars += advance
        
    box_height = (start_y - y_pos) + line_height * 1.1
    
    # Shrink-wrap the dashed box to perfectly match the longest token line
    rect_width = max_x - 0.015 if max_x > 0 else 1.0
    
    rect = patches.Rectangle(
        (0.0, y_pos - line_height * 0.8),
        rect_width,
        box_height,
        linewidth=1.2,
        edgecolor='gray',
        linestyle='--',
        facecolor='none',
        transform=ax.transAxes,
        clip_on=False
    )
    ax.add_patch(rect)
# ──────────────────────────────────────────────────────────────────────────


def make_heatmap(image_path, cam, target_size):
    width, height = target_size
    cam = np.maximum(cam, 0)
    heatmap = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = normalize_scores(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    image_np = np.asarray(Image.open(image_path).convert("RGB"))
    return cv2.addWeighted(image_np, 0.50, heatmap_color, 0.50, 0)


def text_tensors_from_json_or_caption(sample, processor, device, text):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    if isinstance(sample.get("input_ids"), list) and sample["input_ids"]:
        input_ids = [int(token_id) for token_id in sample["input_ids"][:MAX_TEXT_LENGTH]]
        if isinstance(sample.get("attention_mask"), list) and sample["attention_mask"]:
            attention_mask = [int(mask) for mask in sample["attention_mask"][:MAX_TEXT_LENGTH]]
        else:
            attention_mask = [1] * len(input_ids)

        pad_len = MAX_TEXT_LENGTH - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return (
            torch.tensor([input_ids], dtype=torch.long, device=device),
            torch.tensor([attention_mask], dtype=torch.long, device=device),
            "json",
        )

    encoded = processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_TEXT_LENGTH,
        truncation=True,
    )
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        "tokenizer",
    )


def process_sample(model, processor, device, item, image_path, vision_features):
    text = get_caption(item)
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    input_ids, attention_mask, token_source = text_tensors_from_json_or_caption(item, processor, device, text)
    batch = {
        "images": image_inputs["pixel_values"].to(device),
        "caption_input_ids": input_ids,
        "caption_attention_mask": attention_mask,
        "indices": torch.tensor([0], device=device),
        "pids": torch.tensor([0], device=device),
    }

    with torch.set_grad_enabled(True):
        feats = model(batch, is_training=True)
        _total_loss, _losses, gass_current = model.loss_fn(model, feats, batch, is_training=True)
        model.zero_grad(set_to_none=True)
        img_norm = F.normalize(feats["global_img"], dim=-1)
        txt_norm = F.normalize(feats["global_txt"], dim=-1)
        sim = (img_norm * txt_norm).sum()
        sim.backward(retain_graph=True)

    if gass_current is None:
        token_feats = F.normalize(feats["token_txt"][0], dim=-1)
        img_feat = F.normalize(feats["global_img"][0], dim=-1)
        scores = (token_feats @ img_feat).detach().cpu().numpy()
    else:
        scores = gass_current[0].detach().cpu().numpy()
    scores = normalize_scores(scores)

    token_ids = batch["caption_input_ids"][0].detach().cpu().tolist()
    tokens = [processor.tokenizer.decode([token_id]).strip() for token_id in token_ids]

    cam_img = None
    if "acts" in vision_features and "grads" in vision_features:
        acts = vision_features["acts"][0, 1:, :]
        grads = vision_features["grads"][0, 1:, :]
        weights = grads.mean(dim=0, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
        grid_size = int(np.sqrt(cam.numel()))
        if grid_size * grid_size == cam.numel():
            cam = cam.view(grid_size, grid_size).detach().cpu().numpy()
            cam_img = make_heatmap(image_path, cam, image.size)

    return {
        "text": text,
        "tokens": tokens,
        "scores": scores,
        "image_path": image_path,
        "cam_img": cam_img,
        "token_source": token_source,
    }


def load_model(checkpoint, device):
    from model import TBPSLightning

    try:
        return TBPSLightning.load_from_checkpoint(checkpoint, map_location=device, strict=False)
    except Exception:
        return TBPSLightning.load_from_checkpoint(
            checkpoint,
            map_location=device,
            strict=False,
            loss_weights=dict(DEFAULT_LOSS_CONFIG),
        )


# ── CHANGED: draw_figure – wider text column, bigger labels, left=0.01 ────
def draw_figure(results, output_path, model_name):
    """
    Layout (5 columns):
      col 0 – text + token-highlight  (wide)
      col 1 – thin separator gap
      col 2 – original image
      col 3 – NDM-CLIP heatmap
      col 4 – colorbar

    Changes vs. original:
      • figsize wider (13.5 → gives text column more room)
      • width_ratios col-0: 2.95 → 3.40  (text column ~15 % wider)
      • left margin: 0.03 → 0.01  (text starts closer to paper left edge)
      • "Text:" label fontsize 15 → 17, bold, family=serif
      • "Visualization …" sub-label fontsize 16 → 17
      • token highlight fontsize 15 (via draw_highlighted_text kwarg)
      • column headers ("Image", model_name) fontsize 18 → 20
      • footer caption fontsize 14 → 15
      • colorbar label fontsize 12 → 13
    """
    n = len(results)
    
    # Calculate exact physical height needed for text to perfectly size the axes
    max_text_height_inches = 0.0
    for result in results:
        wrapped = textwrap.fill(result["text"], width=37)
        num_lines = wrapped.count('\n') + 1
        
        token_lines = 1
        current_chars = int(0.34 / 0.0217)
        for token in result["tokens"]:
            word = clean_token(token)
            if not word.strip(): continue
            word_len = len(word)
            if current_chars + word_len + 1 > 45:
                token_lines += 1
                current_chars = 0
            current_chars += word_len + 1.2
            
        caption_inches = num_lines * 0.35
        tokens_inches = token_lines * 0.45
        total_inches = caption_inches + 0.45 + tokens_inches
        max_text_height_inches = max(max_text_height_inches, total_inches)
        
    row_height = max_text_height_inches / 0.95  # 5% vertical padding
    fig = plt.figure(figsize=(12.0, row_height * n + 0.55))

    grid = fig.add_gridspec(
        n + 1,
        7,
        # text, gap_text_img, img1, gap_imgs, img2, gap_img_cbar, cbar
        width_ratios=[8.2, 0.25, 1.5, 0.15, 1.5, 0.15, 0.25],
        height_ratios=[*[1.0 for _ in results], 0.08],
        wspace=0.01,
        hspace=0.20,
    )

    for row, result in enumerate(results):
        ax_text = fig.add_subplot(grid[row, 0])
        ax_text.axis("off")

        # ── "Text: …" caption – larger, bold serif label ──────────────────
        # Calculate dynamic centering
        wrapped = textwrap.fill(result["text"], width=37)
        num_lines = wrapped.count('\n') + 1
        
        max_chars = 45
        char_width = 0.0217
        current_chars = int(0.34 / char_width)
        token_lines = 1
        for token in result["tokens"]:
            word = clean_token(token)
            if not word.strip(): continue
            word_len = len(word)
            if current_chars + word_len + 1 > max_chars:
                token_lines += 1
                current_chars = 0
            current_chars += word_len + 1.2
            
        caption_height = (num_lines * 0.35) / row_height
        tokens_height = (token_lines * 0.45) / row_height
        gap = 0.45 / row_height
        total_height = caption_height + gap + tokens_height
        
        caption_start_y = min(0.975, 0.5 + total_height / 2)
        dynamic_start_y = caption_start_y - caption_height - gap

        ax_text.text(
            0.0, caption_start_y,
            "Text: ",
            fontsize=21,
            fontweight="bold",
            family="serif",
            va="top", ha="left",
            transform=ax_text.transAxes,
        )
        # Render the actual caption text right after the bold "Text:" label
        ax_text.text(
            0.16, caption_start_y,           # increased x-offset to prevent overlap
            wrapped,
            fontsize=21,
            va="top", ha="left",
            transform=ax_text.transAxes,
        )

        # ── highlighted tokens ────────────────────────────────────────────
        draw_highlighted_text(
            ax_text,
            result["tokens"],
            result["scores"],
            start_y=dynamic_start_y,
            max_chars=45,       # optimal wrapping width
            fontsize=20,
            char_width=0.0217,  # true physical width mapping
            line_height=0.45 / row_height,
            model_name=model_name
        )

        # ── original image ─────────────────────────────────────────────────
        ax_img = fig.add_subplot(grid[row, 2])
        ax_img.axis("off")
        if row == 0:
            ax_img.set_title("Image", fontsize=20, fontweight="bold", pad=7)
        ax_img.imshow(Image.open(result["image_path"]).convert("RGB"), aspect='auto')
        ax_img.set_anchor('W')

        # ── heatmap ────────────────────────────────────────────────────────
        ax_cam = fig.add_subplot(grid[row, 4])
        ax_cam.axis("off")
        if row == 0:
            ax_cam.set_title(model_name, fontsize=20, fontweight="bold", pad=7)
        if result["cam_img"] is not None:
            ax_cam.imshow(result["cam_img"], aspect='auto')
        else:
            ax_cam.imshow(Image.open(result["image_path"]).convert("RGB"), aspect='auto')
        ax_cam.set_anchor('W')

    # ── attention colorbar (right) ─────────────────────────────────────────
    attn_cax = fig.add_subplot(grid[:n, 6])
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    attn_sm = cm.ScalarMappable(norm=norm, cmap="jet")
    attn_sm.set_array([])
    attn_cbar = fig.colorbar(attn_sm, cax=attn_cax)
    attn_cbar.set_ticks([0.0, 0.5, 1.0])
    attn_cbar.set_ticklabels(["Low", "Mid", "High"])
    attn_cbar.ax.tick_params(labelsize=12)
    attn_cbar.set_label("Attention score", fontsize=13, labelpad=10)

    # ── token-weight colorbar (bottom) ─────────────────────────────────────
    token_cmap = colors.LinearSegmentedColormap.from_list(
        "token_weight_green",
        [(0.94, 1.0, 0.94), (0.0, 0.9, 0.0)],
    )
    token_cax = fig.add_subplot(grid[n, 0])
    token_sm = cm.ScalarMappable(norm=norm, cmap=token_cmap)
    token_sm.set_array([])
    token_cbar = fig.colorbar(token_sm, cax=token_cax, orientation="horizontal")
    token_cbar.set_ticks([0.0, 0.5, 1.0])
    token_cbar.set_ticklabels(["Low", "Mid", "High"])
    token_cbar.ax.tick_params(labelsize=11)
    token_cbar.set_label("Token-text weight score", fontsize=12, labelpad=3)

    # ── footer removed to prevent overlap with colorbar label ────────────────

    # ↓ CHANGED: left margin and pad_inches to shift content to the right
    fig.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.10)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
# ──────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    json_path = find_json(root_dir, args.json).resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    selected = select_samples(items, root_dir, args.split, args.num_samples, args.seed, args.sample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint}")
    model = load_model(str(checkpoint), device)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = True

    processor = CLIPProcessor.from_pretrained(model.model.name_or_path)
    vision_features = hook_vision_features(model)
    results = []
    for idx, (item, image_path) in enumerate(selected, start=1):
        print(f"Processing {idx}/{len(selected)}: {get_file_path(item)}")
        result = process_sample(model, processor, device, item, image_path, vision_features)
        print(f"  Caption: {result['text']}")
        print(f"  Token source: {result['token_source']}")
        results.append(result)

    if len(results) > 1:
        base_path = output_path.parent
        stem = output_path.stem
        ext = output_path.suffix
        for i, res in enumerate(results, start=1):
            out_file = base_path / f"{stem}_{i}{ext}"
            draw_figure([res], out_file, args.model_name)
            print(f"Saved: {out_file}")
    else:
        draw_figure(results, output_path, args.model_name)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
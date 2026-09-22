import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor
from torch.utils.data import DataLoader

# Constants
SOFTLABEL_RATIO = 0.5
# Tỉ lệ dữ liệu (1.0 = 100%, 0.1 = 10%, 0.05 = 5%)
DATA_FRACTION = 1

ROOT_DIR = "/home/ducpv/hoaganhl/tbps_project"
MODEL_NAME = "openai/clip-vit-base-patch16"
BATCH_SIZE = 16
NUM_WORKERS = 2  

CACHE_PATH = os.path.join(ROOT_DIR, "cache")
MAX_EPOCHS = 15
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 77

LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 0.01

def visualize_augmentations(dataset, num_samples: int = 3):
    indices = random.sample(range(len(dataset)), num_samples)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if sample is None:
            continue
            
        # ĐÃ FIX: Lấy đúng key định nghĩa trong dataset.py
        image = sample["images"]
        aug_ss_1 = sample["aug_images1"]
        aug_ss_2 = sample["aug_images2"]
        caption_text = sample["caption_text"]
        file_path = sample["file_path"]
        pid = sample["pids"]

        def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
            tensor = tensor.permute(1, 2, 0)
            tensor = tensor * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            tensor = torch.clamp(tensor, 0, 1)
            return tensor.numpy()

        image_np = tensor_to_image(image)
        aug_ss_1_np = tensor_to_image(aug_ss_1)
        aug_ss_2_np = tensor_to_image(aug_ss_2)

        axes[i][0].imshow(image_np)
        axes[i][0].set_title(f"Original (PID: {pid})")
        axes[i][0].axis("off")
        axes[i][1].imshow(aug_ss_1_np)
        axes[i][1].set_title("Aug 1")
        axes[i][1].axis("off")
        axes[i][2].imshow(aug_ss_2_np)
        axes[i][2].set_title("Aug 2")
        axes[i][2].axis("off")

        print(f"\nSample {i+1} (PID: {pid}, File: {file_path})")
        print(f"Caption: {caption_text}")

    plt.tight_layout()
    plt.show()

def compute_and_cache_features(model, test_loader, cache_path: str = CACHE_PATH) -> dict:
    os.makedirs(cache_path, exist_ok=True)
    cache_file_path = os.path.join(cache_path, "interactive_features.pkl")  
    
    if os.path.exists(cache_file_path): 
        try:
            with open(cache_file_path, "rb") as f:
                return pickle.load(f)
        except EOFError:
            print(f"Warning: Cache file '{cache_file_path}' is corrupted or empty. Recomputing features.")
        except Exception as e:
            print(f"An error occurred while loading cache file '{cache_file_path}': {e}. Recomputing features.")

    model.eval()
    image_features_all, text_features_all = [], []
    paths_all, captions_all, pids_all = [], [], []

    # ĐÃ FIX: Xóa vòng lặp thừa, chỉ chạy 1 lần duy nhất cho sạch và nhanh
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing features"):
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            if isinstance(outputs, dict):
                image_feats = outputs["global_img"]
                text_feats = outputs["global_txt"]
            else:
                image_feats, text_feats = outputs

            image_feats = F.normalize(image_feats, dim=-1).cpu()
            text_feats = F.normalize(text_feats, dim=-1).cpu()
            
            image_features_all.append(image_feats)
            text_features_all.append(text_feats)
            
            paths_all.extend(batch["file_path"])
            captions_all.extend(batch["caption_text"])
            pids_all.extend(batch["pids"].cpu().tolist()) # Đã fix key "pid" thành "pids" cho khớp dataset.py

    cache_data = {
        "image_features": torch.cat(image_features_all),
        "text_features": torch.cat(text_features_all),
        "paths": paths_all,
        "captions": captions_all,
        "pids": pids_all
    }
    
    with open(cache_file_path, "wb") as f:
        pickle.dump(cache_data, f)
        
    return cache_data
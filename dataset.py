import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import pytorch_lightning as pl
from typing import Dict, List, Optional
from augmentations import ImageAugmentation, TextAugmentation, pre_caption
from utils import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

script_dir = os.path.dirname(os.path.abspath(__file__))

# Ref: _ga_dms_ref/datasets/bases.py
def sigmoid(similarity, k, threshold, max_num):
    return max_num / (1 + np.exp(-k * (similarity - threshold)))

def tokenize_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> tuple:
    out = tokenizer(text, padding=False, truncation=True, max_length=max_length)
    return out["input_ids"], out["attention_mask"]

class VN3KDataset(Dataset):
    def __init__(self, root_dir: str, preproc_json: str = "vn3k_preproc_texts_final.json", 
                 split: str = "train", size: int = 224, use_preproc: bool = True, 
                 augment: bool = True, data_fraction: float = 1.0):
        self.root_dir = root_dir
        self.size = size
        self.split = split
        self.augment = augment and (split == "train") 
        
        self.image_aug = ImageAugmentation(size=size, normalize=normalize) if self.augment else None
        self.text_aug = TextAugmentation(p=0.05) if self.augment else None
        self.base_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), normalize])
        
        preproc_path = os.path.join(script_dir, preproc_json)
        
        if use_preproc and os.path.exists(preproc_path):
            with open(preproc_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # 1. XỬ LÝ LINH HOẠT MỌI CẤU TRÚC JSON
            train_items, test_items = [], []
            if isinstance(raw_data, dict):
                train_items = raw_data.get("train", [])
                test_items = raw_data.get("test", []) + raw_data.get("val", [])
            elif isinstance(raw_data, list):
                for it in raw_data:
                    s = str(it.get("split", it.get("spli", "train"))).strip().lower()
                    if s == "test" or s == "val":
                        test_items.append(it)
                    else:
                        train_items.append(it)
            
            # 2. CHIA ĐÔI TẬP TEST (Nếu có yêu cầu lấy VAL)
            train_pid_set = set(str(it.get("person_id")) for it in train_items if it.get("person_id") is not None)
            pool = [it for it in test_items if str(it.get("person_id")) not in train_pid_set]
            pid_groups = defaultdict(list)
            for it in pool:
                pid_groups[str(it.get("person_id"))].append(it)

            pids_sorted = sorted(pid_groups.keys())
            rng = random.Random(42)
            rng.shuffle(pids_sorted)
            half_pid = len(pids_sorted) // 2
            val_items = []
            test_items_new = []
            for pid in pids_sorted[:half_pid]:
                val_items.extend(pid_groups[pid])
            for pid in pids_sorted[half_pid:]:
                test_items_new.extend(pid_groups[pid])
            
            if split == "train":
                items = train_items
            elif split == "val":
                items = val_items
            elif split == "test":
                items = test_items_new
            else:
                items = []

            total_json_samples = len(items)
            print(f"🔍 [{split.upper():<5}] Tìm thấy {total_json_samples} mẫu trong JSON.")
            if total_json_samples == 0:
                raise ValueError(f"❌ Tập '{split}' rỗng!")

            # 3. LỌC ẢNH & KIỂM TRA ĐƯỜNG DẪN 
            if split == "train" and 0.0 < data_fraction < 1.0:
                # Cố định mask pid logic nếu cần (tuỳ bạn), hiện tại giữ như lúc xử lý của bạn
                limit = max(1, int(total_json_samples * data_fraction))
                random.seed(42)
                random.shuffle(items)
                items = items[:limit]

            self.items = []
            pid_map = {}
            pidx = 0
            
            for it in items:
                file_path = it.get("file_path", "")
                if file_path.startswith("images/") or file_path.startswith("images/"):
                    img_path = os.path.join(root_dir, file_path)
                else:
                    img_path = os.path.join(root_dir, "images", file_path)
                
                if not os.path.exists(img_path):
                    continue
                
                pid = int(it.get("person_id", -1))
                if pid not in pid_map:
                    pid_map[pid] = pidx
                    pidx += 1
                
                self.items.append({
                    "file_path": file_path,
                    "image_path": img_path,
                    "caption": it["caption"],
                    "input_ids": it["input_ids"],
                    "attention_mask": it["attention_mask"],
                    "pid": pid_map[pid],
                    "attr_boxes": it.get("attr_boxes", []) 
                })
        else:
            raise ValueError(f"Preproc file {preproc_json} is required at {preproc_path}")

        # Tác giả khởi tạo sim = 0.85 (1 - 0.15) cho Epoch đầu tiên (xem _ga_dms_ref/datasets/luperson.py dòng 155)
        self._gass_scores: List[np.ndarray] = [
            np.full(MAX_TEXT_LENGTH, 0.85, dtype=np.float32) for _ in range(len(self.items))
        ]

    def set_gass_score(self, dataset_idx: int, score: np.ndarray):
        """Cache GASS score for a sample so __getitem__ can use it next epoch."""
        if 0 <= dataset_idx < len(self._gass_scores):
            score = np.asarray(score, dtype=np.float32).reshape(-1)
            score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
            if score.shape[0] < MAX_TEXT_LENGTH:
                pad = np.zeros(MAX_TEXT_LENGTH - score.shape[0], dtype=np.float32)
                score = np.concatenate([score, pad], axis=0)
            else:
                score = score[:MAX_TEXT_LENGTH]
            self._gass_scores[dataset_idx] = np.clip(score, 0.0, 1.0)

    def set_epoch(self, epoch: int):
        # Hook kept for compatibility with LightningDataModule.set_epoch().
        self._epoch = int(epoch)

    @staticmethod
    def _build_random_masked_tokens_and_labels(
        tokens: np.ndarray,   # int64 numpy array, will be mutated in-place (like ref)
        sim: np.ndarray,      # per-token GASS score, same shape as tokens
        max_num: float,
        mode: int,            # 1 = MTP (mask informative), 2 = noise (mask uninformative)
    ):
        """
        Exact replica of FilterDataset._build_random_masked_tokens_and_labels
        from _ga_dms_ref/datasets/bases.py.

        ref call-site:
            mlm_tokens, mlm_labels = _build_random_masked_tokens_and_labels(tokens, sim, 0.3, 1)
            nose_tokens, _         = _build_random_masked_tokens_and_labels(tokens, 1-sim, 0.2, 2)
        """
        MASK_ID     = 49405
        token_range = list(range(1, 49405))   # 1 ~ 49404 — ref: range(1, len(tokenizer.encoder)-3)

        labels = []

        if mode == 1:
            k         = 20.0
            threshold = 0.1
        elif mode == 2:
            k         = 20.0
            threshold = 0.9
        else:
            k, threshold = 20.0, 0.1

        # ref: ori_pro = sigmoid(sim, k, threshold, max_num)
        ori_pro = sigmoid(sim, k, threshold, max_num)  # [L] float — exact ref call

        for i, token in enumerate(tokens.tolist()):   # iterate like ref
            if 0 < token < 49405:                     # ref: `if 0 < token < 49405`
                prob = random.random()
                if prob < ori_pro[i]:
                    prob /= ori_pro[i]                # BERT re-normalise — exact ref line

                    if prob < 0.8:                    # 80% → MASK
                        tokens[i] = MASK_ID
                    elif prob < 0.9:                  # 10% → random token
                        tokens[i] = random.choice(token_range)
                    # else 10% → keep original

                    labels.append(token)              # original token id as label
                else:
                    labels.append(0)
            else:
                labels.append(0)

        # ref: ensure at least 1 token is masked
        if all(l == 0 for l in labels):
            labels[1] = int(tokens[1])                # will be overwritten below
            tokens[1] = MASK_ID
            labels[1] = int(tokens[1] if tokens[1] != MASK_ID else labels[1])  # keep original

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        pil = Image.open(it["image_path"]).convert("RGB")
        orig_W, orig_H = pil.size
        
        image = self.base_transform(pil)
        aug_images1 = self.image_aug.get_augmented_image(pil) if self.augment else image
        aug_images2 = self.image_aug.get_augmented_image(pil) if self.augment else image

        caption = it["caption"]
        ids = it["input_ids"]
        attention_mask = it["attention_mask"]
        
        if self.augment:
            caption_aug = self.text_aug.random_deletion(caption)
            ids_aug, attention_mask_aug = tokenize_text(caption_aug)
        else:
            ids_aug, attention_mask_aug = ids, attention_mask

        grid_size = self.size // 16
        attr_mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.float32)

        for attr_dict in it.get("attr_boxes", []):
            box = attr_dict.get("box", [])
            if len(box) == 4:
                x_min, y_min, x_max, y_max = box
                grid_x_min = max(0, int((x_min / orig_W) * grid_size))
                grid_y_min = max(0, int((y_min / orig_H) * grid_size))
                grid_x_max = min(grid_size - 1, int((x_max / orig_W) * grid_size))
                grid_y_max = min(grid_size - 1, int((y_max / orig_H) * grid_size))
                attr_mask_2d[grid_y_min:grid_y_max+1, grid_x_min:grid_x_max+1] = 1.0

        attr_mask = attr_mask_2d.view(-1)

        # ── Two-epoch pipeline — exact ref FilterDataset pattern ─────────────
        # ref: mlm_tokens, mlm_labels = _build_random_masked_tokens_and_labels(tokens, sim, 0.3, 1)
        #      nose_tokens, _          = _build_random_masked_tokens_and_labels(tokens, 1-sim, 0.2, 2)
        gass_score = self._gass_scores[idx]
        if self.split == "train":
            tokens_np = np.array(ids, dtype=np.int64)
            sim       = gass_score.astype(np.float64)      # float64 for exp() stability

            mtp_ids_tensor, mtp_labels_tensor = self._build_random_masked_tokens_and_labels(
                tokens_np.copy(), sim.copy(), 0.3, 1       # MTP: mask high-score tokens
            )
            noise_ids_tensor, _ = self._build_random_masked_tokens_and_labels(
                tokens_np.copy(), 1.0 - sim.copy(), 0.2, 2  # Noise: mask low-score tokens
            )
        else:
            mtp_ids_tensor    = None
            mtp_labels_tensor = None
            noise_ids_tensor  = None

        return {
            "images": image,
            "aug_images1": aug_images1,
            "aug_images2": aug_images2,
            "caption_input_ids": torch.tensor(ids, dtype=torch.long),
            "caption_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "caption_input_ids_aug": torch.tensor(ids_aug, dtype=torch.long),
            "caption_attention_mask_aug": torch.tensor(attention_mask_aug, dtype=torch.long),
            "pids": it["pid"],
            "file_path": it["file_path"],
            "caption_text": caption,
            "attr_mask": attr_mask,
            "index": idx,
            "noise_input_ids": noise_ids_tensor,    # masked for SDM denoising
            "mtp_input_ids":   mtp_ids_tensor,       # masked for MTP prediction
            "mtp_labels":      mtp_labels_tensor,    # original token ids at masked positions
        }
 
 
def collate_fn(batch: List[Dict]) -> Dict: 
    batch = [b for b in batch if b is not None] 
    
    def _pad_to_max(seq: torch.Tensor, pad_value: int, max_len: int) -> torch.Tensor: 
        if seq.size(1) == max_len: 
            return seq 
        if seq.size(1) < max_len: 
            pad = torch.full((seq.size(0), max_len - seq.size(1)), pad_value, dtype=seq.dtype, device=seq.device) 
            return torch.cat([seq, pad], dim=1) 
        return seq 
 
    caption_input_ids = pad_sequence([b["caption_input_ids"] for b in batch], batch_first=True, padding_value=PAD_TOKEN_ID) 
    caption_attention_mask = pad_sequence([b["caption_attention_mask"] for b in batch], batch_first=True, padding_value=0) 
    caption_input_ids_aug = pad_sequence([b["caption_input_ids_aug"] for b in batch], batch_first=True, padding_value=PAD_TOKEN_ID) 
    caption_attention_mask_aug = pad_sequence([b["caption_attention_mask_aug"] for b in batch], batch_first=True, padding_value=0) 
 
    caption_input_ids = _pad_to_max(caption_input_ids, PAD_TOKEN_ID, MAX_TEXT_LENGTH) 
    caption_attention_mask = _pad_to_max(caption_attention_mask, 0, MAX_TEXT_LENGTH) 
    caption_input_ids_aug = _pad_to_max(caption_input_ids_aug, PAD_TOKEN_ID, MAX_TEXT_LENGTH) 
    caption_attention_mask_aug = _pad_to_max(caption_attention_mask_aug, 0, MAX_TEXT_LENGTH) 

    # ── Two-epoch pipeline — exact ref pattern ───────────────────────────────
    # ref: batch has 'mlm_ids', 'mlm_labels', 'noise_text' (pre-computed at dataset level)
    noise_list      = [b.get("noise_input_ids") for b in batch]
    mtp_list        = [b.get("mtp_input_ids")   for b in batch]
    mtp_labels_list = [b.get("mtp_labels")      for b in batch]

    if (
        all(t is not None for t in noise_list)
        and all(t is not None for t in mtp_list)
        and all(t is not None for t in mtp_labels_list)
    ):
        noise_ids_batch    = pad_sequence(noise_list,      batch_first=True, padding_value=PAD_TOKEN_ID)
        noise_ids_batch    = _pad_to_max(noise_ids_batch,  PAD_TOKEN_ID, MAX_TEXT_LENGTH)
        mtp_ids_batch      = pad_sequence(mtp_list,        batch_first=True, padding_value=PAD_TOKEN_ID)
        mtp_ids_batch      = _pad_to_max(mtp_ids_batch,    PAD_TOKEN_ID, MAX_TEXT_LENGTH)
        mtp_labels_batch   = pad_sequence(mtp_labels_list, batch_first=True, padding_value=0)
        mtp_labels_batch   = _pad_to_max(mtp_labels_batch, 0, MAX_TEXT_LENGTH)
    else:
        noise_ids_batch  = None
        mtp_ids_batch    = None
        mtp_labels_batch = None

    res = { 
        "images": torch.stack([b["images"] for b in batch]), 
        "aug_images1": torch.stack([b["aug_images1"] for b in batch]), 
        "aug_images2": torch.stack([b["aug_images2"] for b in batch]), 
        "caption_input_ids": caption_input_ids, 
        "caption_attention_mask": caption_attention_mask, 
        "caption_input_ids_aug": caption_input_ids_aug, 
        "caption_attention_mask_aug": caption_attention_mask_aug, 
        "pids": torch.tensor([b["pids"] for b in batch], dtype=torch.long), 
        "file_path": [b["file_path"] for b in batch], 
        "caption_text": [b["caption_text"] for b in batch], 
        "attr_mask": torch.stack([b["attr_mask"] for b in batch]), 
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "noise_input_ids": noise_ids_batch,   # ref: 'noise_text'
        "mtp_input_ids":   mtp_ids_batch,     # ref: 'mlm_ids'
        "mtp_labels":      mtp_labels_batch,  # ref: 'mlm_labels'
    } 
 
    return res 
 
 
class VN3KDataModule(pl.LightningDataModule): 
    def __init__(self, root_dir: str, preproc_json: str = "vn3k_preproc_texts_final.json", 
                  batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, size: int = 224, 
                  data_fraction: float = 1.0): 
        super().__init__() 
        self.root_dir = root_dir 
        self.preproc_json = preproc_json 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.size = size 
        
        try: 
            from utils import DATA_FRACTION 
            self.data_fraction = DATA_FRACTION 
        except ImportError: 
            self.data_fraction = data_fraction 
 
 
    def setup(self, stage: str = None): 
        self.train_dataset = VN3KDataset( 
            self.root_dir, self.preproc_json, "train", self.size, 
            augment=True, data_fraction=self.data_fraction 
        ) 
        
        self.val_dataset = VN3KDataset( 
            self.root_dir, self.preproc_json, "val", self.size, 
            augment=False, data_fraction=1.0 
        ) 
        self.test_dataset = VN3KDataset( 
            self.root_dir, self.preproc_json, "test", self.size, 
            augment=False, data_fraction=1.0 
        ) 
 
 
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, 
                           collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=True) 
 
 
    def build_filter_loader(self) -> DataLoader: 
        # Periodic restructuring hook (same dataset, rebuilt per epoch) 
        return self.train_dataloader() 
 
 
    def set_epoch(self, epoch: int): 
        if hasattr(self, "train_dataset") and self.train_dataset is not None: 
            self.train_dataset.set_epoch(epoch) 
 
 
    def val_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
                           collate_fn=collate_fn, pin_memory=torch.cuda.is_available()) 
 
 
    def test_dataloader(self) -> DataLoader: 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
                           collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

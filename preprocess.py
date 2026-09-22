import os
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Tuple, List

# Nhập cấu hình từ utils.py (Đảm bảo utils.py chứa MODEL_NAME = "openai/clip-vit-base-patch16")
from utils import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

# Tự động lấy thư mục chứa file preprocess.py này
script_dir = os.path.dirname(os.path.abspath(__file__))

def pre_caption(caption: str, max_words=77) -> str:
    if not caption:
        return ""
    # Xóa ký tự đặc biệt, chuyển thành chữ thường
    s = re.sub(r"[!\"#$%&()*+,:;.<=>?@[\\\]^_`{|}~]", " ", caption.lower())
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split(" ")
    return " ".join(words[:max_words])

def tokenize_text(text: str, max_length=77) -> Tuple[List[int], List[int]]:
    out = tokenizer(text, padding=False, truncation=True, max_length=max_length)
    return out["input_ids"], out["attention_mask"]

def preprocess_annotations(ann_file: str = "vn3k_pedes_reid_raw_with_boxes.json", 
                           out_file: str = "vn3k_preproc_texts_final.json",       
                           max_length: int = 77) -> str:
    
    ann_path = os.path.join(script_dir, ann_file)
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {ann_path}\nBạn phải chạy Bước 2 (Grounding DINO) trước!")
    
    print(f"📦 Đang đọc dữ liệu từ: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
        
    results = []
    
    # Khởi tạo bộ đếm số lượng mẫu cho từng tập
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    print("⚡ Đang làm sạch Text, Tokenize và đóng gói JSON...")
    for ann in tqdm(anns):
        file_path = ann.get("file_path")
        
        # Trong reid_raw.json của VN3K, ID người dùng khóa là "id" (không phải "person_id")
        pid = ann.get("id") 
        split = ann.get("split", "train")
        
        # Lấy mảng Box mà DINO đã bắt được ở Bước 2
        attr_boxes = ann.get("attr_boxes", []) 

        # Trong reid_raw.json, caption là một List các chuỗi
        caps = ann.get("captions", [])
        
        # ĐẬP PHẲNG: Lặp qua từng caption để tạo ra các mẫu (samples) huấn luyện riêng biệt
        for cap in caps:
            cap_clean = pre_caption(cap)
            if not cap_clean:
                continue
                
            input_ids, attn = tokenize_text(cap_clean, max_length=max_length)
            
            results.append({
                "file_path": file_path,
                "person_id": int(pid) if pid is not None else -1,
                "split": split,
                "caption": cap_clean,
                "input_ids": input_ids,
                "attention_mask": attn,
                "attr_boxes": attr_boxes # Nhúng Box vào từng mẫu
            })
            
            # Cập nhật số lượng đếm
            if split in split_counts:
                split_counts[split] += 1
            else:
                split_counts[split] = 1 # Phòng trường hợp có tên split khác lạ
            
    out_path = os.path.join(script_dir, out_file)
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(results, wf, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("📊 THỐNG KÊ KẾT QUẢ TIỀN XỬ LÝ VN3K-PEDES")
    print("="*50)
    print(f"Tổng số mẫu đã tạo    : {len(results)}")
    print(f" 🔸 Tập Train         : {split_counts.get('train', 0)} mẫu")
    print(f" 🔸 Tập Validation    : {split_counts.get('val', 0)} mẫu")
    print(f" 🔸 Tập Test          : {split_counts.get('test', 0)} mẫu")
    print("="*50)
    
    print(f"✅ Đã lưu file cuối cùng tại: {out_path}")
    return out_path

if __name__ == "__main__":
    preprocess_annotations()
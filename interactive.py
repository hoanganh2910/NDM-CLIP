import os
import random
import sys
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.text import Text
from PIL import Image
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor
from dataset import VN3KDataModule
from model import TBPSLightning
from utils import *

def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except UnicodeDecodeError:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        raw = sys.stdin.buffer.readline()
        return raw.decode('utf-8', errors='replace').rstrip('\n')

def generate_qualitative_result(model: TBPSLightning, test_loader: torch.utils.data.DataLoader, cache_path: str = CACHE_PATH):
    model.eval()
    
    print("Đang trích xuất đặc trưng (Features) từ Model Checkpoint...")
    cache_data = compute_and_cache_features(model, test_loader, cache_path)
    
    if not hasattr(test_loader, "dataset") or not hasattr(test_loader.dataset, "items") or len(test_loader.dataset.items) == 0:
        print("Lỗi: Không tìm thấy dữ liệu JSON trong test_loader.")
        return

    # TẠO MAPPING CHUẨN TỪ RAW JSON ĐỂ LẤY ĐÚNG PID GỐC
    path_to_pid = {}
    for item in test_loader.dataset.items:
        path = item.get("file_path", item.get("image_path", ""))
        raw_pid = str(item.get("pid", item.get("person_id", item.get("id", ""))))
        if path:
            path_to_pid[path] = raw_pid

    # LỌC TRÙNG LẶP GALLERY (DEDUPLICATION)
    raw_gallery_features = cache_data["image_features"].to(model.device)
    raw_gallery_paths = cache_data["paths"]
    
    unique_gallery_feats = []
    unique_gallery_paths = []
    seen_paths = set()
    
    for i, path in enumerate(raw_gallery_paths):
        if path not in seen_paths:
            seen_paths.add(path)
            unique_gallery_paths.append(path)
            unique_gallery_feats.append(raw_gallery_features[i])
            
    gallery_image_features = torch.stack(unique_gallery_feats)
    gallery_paths = unique_gallery_paths

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    def get_text_features(query: str) -> torch.Tensor:
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                return_dict=True,
            )
            token_embeds = model.model.text_projection(outputs.last_hidden_state)
            eos_idx = model._eos_indices(input_ids)
            text_feats = token_embeds[
                torch.arange(token_embeds.size(0), device=token_embeds.device),
                eos_idx,
            ]
        return F.normalize(text_feats, dim=-1)

    def run_multi_row_query(num_rows: int = 4, top_k: int = 5):
        sampled_items = random.sample(test_loader.dataset.items, num_rows)
        
        num_cols = top_k + 2 
        # Cấu hình Figure to và nét hơn, chiều cao bóp lại cho dẹt
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(22, 2.6 * num_rows), 
                                gridspec_kw={'width_ratios': [2.4] + [1]*(num_cols-1)})
        fig.patch.set_facecolor('white')

        for row, item in enumerate(sampled_items):
            caption = item["caption"]
            gt_file_path = item.get("file_path", item.get("image_path", ""))
            gt_pid = str(item.get("pid", item.get("person_id", item.get("id", ""))))
            
            print(f"[{row+1}/{num_rows}] Đang xử lý: {caption[:30]}...")

            query_feats = get_text_features(caption)
            similarity = query_feats @ gallery_image_features.t()
            topk_values, indices = similarity.topk(top_k, dim=1)
            top_indices = indices[0].cpu().tolist()

            # ---------------- CỘT 0: CAPTION ----------------
            ax_text = axs[row, 0]
            ax_text.axis("off")
            wrapped_text = textwrap.fill(caption, width=45) 
            ax_text.text(0.02, 0.5, wrapped_text, fontsize=14, fontfamily='serif', 
                         verticalalignment='center', horizontalalignment='left',
                         transform=ax_text.transAxes) 
            if row == 0:
                ax_text.set_title("Caption", fontsize=16, fontweight='bold', fontfamily='serif', pad=15, loc='left')

            # ---------------- CỘT 1: GROUND TRUTH ----------------
            ax_gt = axs[row, 1]
            gt_full_path = os.path.join(ROOT_DIR, "images", gt_file_path)
            try:
                img_gt = Image.open(gt_full_path).convert("RGB")
                ax_gt.imshow(img_gt)
            except Exception:
                pass
            ax_gt.axis("off")
            
            if row == 0:
                ax_gt.set_title("Ground Truth", fontsize=15, fontweight='bold', color='#0066cc', fontfamily='serif', pad=15)
            
            rect_gt = patches.Rectangle((0, 0), img_gt.width-1, img_gt.height-1, linewidth=6, edgecolor='#0066cc', facecolor='none')
            ax_gt.add_patch(rect_gt)

            # ---------------- CỘT 2 TRỞ ĐI: TOP K RETRIEVAL ----------------
            for col, idx in enumerate(top_indices):
                ax_img = axs[row, col + 2]
                file_path = gallery_paths[idx]
                pid = path_to_pid.get(file_path, "Unknown")
                
                img_path = os.path.join(ROOT_DIR, "images", file_path)
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax_img.imshow(img)
                except Exception:
                    pass
                ax_img.axis("off") 
                
                # Kiểm tra đúng sai
                is_gt = (pid == gt_pid)
                color = '#00cc00' if is_gt else '#cc0000' # Xanh lá (Đúng) / Đỏ (Sai)
                
                if row == 0:
                    ax_img.set_title(f"Rank {col+1}", fontsize=15, fontweight='bold', color=color, fontfamily='serif', pad=15)
                
                # FIX LỚN: Ghi đè Rank lên ảnh, dùng màu sắc để phân biệt
                rank_text = f"Rank {col+1}"
                ax_img.text(0.5, 0.95, rank_text, fontsize=13, fontweight='bold', color=color,
                            family='serif', horizontalalignment='center', verticalalignment='top',
                            transform=ax_img.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                
                rect = patches.Rectangle((0, 0), img.width-1, img.height-1, linewidth=6, edgecolor=color, facecolor='none')
                ax_img.add_patch(rect)
                
        # Căn lề khít rịt để tạo thành 1 bảng vững chãi
        plt.subplots_adjust(wspace=0.08, hspace=0.08)
        
        results_dir = os.path.join(ROOT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"result_retrieval.jpg")

        # Cắt gọt rìa thừa để LaTeX ăn khít
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close(fig)
        print(f"\n[v] Đã xuất ảnh siêu tối giản, chuẩn khoa học (Rank đè lên ảnh) tại: {save_path}")

    print("\n" + "="*60)
    print("TOOL TẠO ẢNH ĐỊNH TÍNH (BẢN RANK ĐÈ LÊN ẢNH)")
    print("="*60)
    while True:
        choice = safe_input("Nhấn [ENTER] để model truy xuất và tạo ảnh (hoặc 'q' để thoát): ").strip().lower()
        if choice == 'q':
            break
        run_multi_row_query(num_rows=4, top_k=5)

def main():
    data_module = VN3KDataModule(ROOT_DIR)
    data_module.setup()

    checkpoint_path = "/home/ducpv/hoaganhl/tbps_project/checkpoints/best-r1-epoch11.ckpt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy Checkpoint: {checkpoint_path}")

    print(f"Đang load model từ: {checkpoint_path}")
    model = TBPSLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    
    generate_qualitative_result(model, data_module.test_dataloader())

if __name__ == "__main__":
    main()
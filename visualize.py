import json
import os
import random
from PIL import Image, ImageDraw, ImageFont

def visualize_bounding_boxes(num_samples=10):
    # ================= CẤU HÌNH ĐƯỜNG DẪN CUHK =================
    CUHK_ROOT = "/home/ducpv/hoaganhl/tbps_project/CUHK-PEDES"
    
    input_json = os.path.join(CUHK_ROOT, "cuhk_preproc_with_boxes.json")
    images_root_dir = os.path.join(CUHK_ROOT, "imgs") 
    output_dir = os.path.join(CUHK_ROOT, "box_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # ================= ĐỌC DỮ LIỆU =================
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_json}")
        return

    valid_data = [item for item in data if len(item.get("attr_boxes", [])) > 0]
    print(f"Tổng số ảnh có Bounding Box: {len(valid_data)}/{len(data)}")
    
    if len(valid_data) == 0:
        print("Không có ảnh nào chứa Box để vẽ!")
        return

    samples = random.sample(valid_data, min(num_samples, len(valid_data)))
    print(f"Bắt đầu vẽ box cho {len(samples)} ảnh mẫu...")
    
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "pink"]
    
    for idx, item in enumerate(samples):
        image_path = os.path.join(images_root_dir, item["file_path"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi đọc ảnh {image_path}: {e}")
            continue
            
        draw = ImageDraw.Draw(image)
        boxes = item["attr_boxes"]
        
        for i, box_info in enumerate(boxes):
            box = box_info["box"] 
            phrase = box_info["phrase"]
            score = box_info["score"]
            color = colors[i % len(colors)]
            
            safe_box = [max(0, c) for c in box]
            draw.rectangle(safe_box, outline=color, width=3)
            
            label = f"{phrase} ({score:.2f})"
            
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = max(0, box[0])
            text_y = max(0, box[1] - text_height - 4)
            
            draw.rectangle(
                [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                fill="black"
            )
            draw.text((text_x + 2, text_y + 2), label, fill=color)

        safe_filename = item["file_path"].replace("/", "_").replace("\\", "_")
        out_path = os.path.join(output_dir, f"{idx+1:02d}_{safe_filename}")
        image.save(out_path)
        print(f"Đã lưu: {out_path}")

    print(f"\n✅ Hoàn thành! Bạn hãy vào thư mục '{output_dir}' để xem kết quả nhé.")

if __name__ == "__main__":
    visualize_bounding_boxes(num_samples=20)
import json
from collections import defaultdict

def check_dynamic_split_integrity(json_path: str):
    print(f"Đang đọc file: {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 1. BÓC TÁCH DỮ LIỆU GIỐNG HỆT dataset.py
    train_items, test_pool_items = [], []
    if isinstance(raw_data, dict):
        train_items = raw_data.get("train", [])
        test_pool_items = raw_data.get("test", []) + raw_data.get("val", [])
    elif isinstance(raw_data, list):
        for it in raw_data:
            s = str(it.get("split", it.get("spli", "train"))).strip().lower()
            if s == "test" or s == "val":
                test_pool_items.append(it)
            else:
                train_items.append(it)

    # 2. CHIA ĐÔI TẬP TEST ĐỂ TẠO VAL (Giống hệt logic trong VN3KDataset)
    train_pid_set = set(str(it.get("person_id")) for it in train_items if it.get("person_id") is not None)
    pool = [it for it in test_pool_items if str(it.get("person_id")) not in train_pid_set]
    
    pid_groups = defaultdict(list)
    for it in pool:
        pid = str(it.get("person_id"))
        pid_groups[pid].append(it)

    pids_sorted = sorted(pid_groups.keys())
    half_pid = len(pids_sorted) // 2
    
    val_items = []
    test_items_new = []
    for pid in pids_sorted[:half_pid]:
        val_items.extend(pid_groups[pid])
    for pid in pids_sorted[half_pid:]:
        test_items_new.extend(pid_groups[pid])

    # 3. GOM VÀO DICT ĐỂ TÍNH TOÁN
    splits = {
        "train": train_items,
        "val": val_items,
        "test": test_items_new
    }

    # 4. TÍNH TOÁN THỐNG KÊ
    stats = {}
    for split_name, items in splits.items():
        if not items:
            continue
            
        pids = set()
        img_paths = set()
        
        for it in items:
            pid = it.get("person_id")
            if pid is not None:
                pids.add(str(pid))
            img_paths.add(it.get("file_path", ""))
            
        stats[split_name] = {
            "total_samples": len(items),
            "unique_pids": pids,
            "unique_images": img_paths
        }

    # ==========================================
    # BÁO CÁO THỐNG KÊ (IN RA MÀN HÌNH)
    # ==========================================
    print("\n" + "="*60)
    print("📊 BÁO CÁO PHÂN BỐ DỮ LIỆU THỰC TẾ (RUNTIME DATASET STATS)")
    print("="*60)
    
    for split_name, data in stats.items():
        print(f"[{split_name.upper():<5}] Tổng số mẫu (ảnh-text): {data['total_samples']:<6} | Số Person ID (Identities): {len(data['unique_pids'])}")

    print("\n" + "="*60)
    print("🛡️ KIỂM TRA RÒ RỈ DỮ LIỆU (DATA LEAKAGE CHECK)")
    print("="*60)
    
    def check_overlap(set1, set2, name1, name2, type_name):
        intersection = set1.intersection(set2)
        if len(intersection) == 0:
            print(f"✅ Pass: Không rò rỉ {type_name} giữa {name1.upper()} và {name2.upper()}")
        else:
            print(f"❌ FAIL: Trùng {len(intersection)} {type_name} giữa {name1.upper()} và {name2.upper()}!")

    # --- Kiểm tra Identity ---
    print("--- 1. Kiểm tra Identity (Person ID) Leakage ---")
    check_overlap(stats["train"]["unique_pids"], stats["val"]["unique_pids"], "train", "val", "Person ID")
    check_overlap(stats["train"]["unique_pids"], stats["test"]["unique_pids"], "train", "test", "Person ID")
    check_overlap(stats["val"]["unique_pids"], stats["test"]["unique_pids"], "val", "test", "Person ID")

    # --- Kiểm tra Image ---
    print("\n--- 2. Kiểm tra Image Leakage ---")
    check_overlap(stats["train"]["unique_images"], stats["val"]["unique_images"], "train", "val", "Image")
    check_overlap(stats["train"]["unique_images"], stats["test"]["unique_images"], "train", "test", "Image")
    check_overlap(stats["val"]["unique_images"], stats["test"]["unique_images"], "val", "test", "Image")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    check_dynamic_split_integrity("vn3k_preproc_texts_final.json")
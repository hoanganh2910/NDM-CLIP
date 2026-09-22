import random
import numpy as np
import torchvision.transforms as transforms
import re

class ImageAugmentation:
    def __init__(self, size: int = 224, normalize: transforms.Normalize = None):
        self.size = size
        self.normalize = normalize
        self.pool = [
            transforms.ColorJitter(0.1, 0.1, 0.1, 0),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(size=size, scale=(0.9, 1.0), ratio=(3/4, 4/3), antialias=True),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
        self.to_tensor = transforms.ToTensor()
        self.random_erasing = transforms.RandomErasing(scale=(0.10, 0.20))

    def get_augmented_image(self, image_pil):
        aug_choice = random.sample(self.pool, 2)
        used_rrc = any(isinstance(a, transforms.RandomResizedCrop) for a in aug_choice)
        for aug in aug_choice:
            image_pil = aug(image_pil)
        if not used_rrc:
            image_pil = transforms.Resize((self.size, self.size))(image_pil)
        t = self.to_tensor(image_pil)
        if random.random() < 0.5:
            t = self.random_erasing(t)
        if self.normalize is not None:
            t = self.normalize(t)
        return t

class TextAugmentation:
    def __init__(self, p: float = 0.05):
        # Xác suất xóa từ ngẫu nhiên chuẩn bài báo (5%)
        self.p = p

    def random_deletion(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)
                
        # Nếu không may xóa sạch, bốc ngẫu nhiên 1 từ từ câu gốc để tránh lỗi
        if len(new_words) == 0:
            return random.choice(words)

        return " ".join(new_words)

def pre_caption(caption: str, max_words: int = 50) -> str:
    if not caption:
        return ""
    s = re.sub(r'[!"#$%&()*+,:;.<=>?@[\\\]^_`{|}~]', " ", caption.lower())
    s = re.sub(r'\s+', " ", s).strip()
    words = s.split(" ")
    return " ".join(words[:max_words])
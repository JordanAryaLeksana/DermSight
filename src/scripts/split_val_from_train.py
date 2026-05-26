import os
import random
import shutil
from pathlib import Path

def split_val_from_train(
    train_dir: str, 
    val_dir:str,
    val_ratio: float = 0.1,
    seed:int = 42,
):
    random.seed(seed)
    train_dir =Path(train_dir)
    val_dir = Path(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        val_class_dir = val_dir / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = [
            img for img in class_dir.iterdir() 
            if img.suffix.lower() in image_extensions
        ]
        
        random.shuffle(images)
        
        num_val = max(1, int(len(images) * val_ratio))
        
        val_images = images[:num_val]
        
        print(f"{class_name}: total={len(images)}, val={len(val_images)}")
        
        for img_path in val_images:
            target_path = val_class_dir / img_path.name
            shutil.move(str(img_path), str(target_path))
            
if __name__ == "__main__":
    split_val_from_train(
        train_dir="/home/jordan/Backup HDD/AI_Machine Learning/DermSight/src/data/train",
        val_dir="/home/jordan/Backup HDD/AI_Machine Learning/DermSight/src/data/val",
        val_ratio=0.1,
        seed=42,
    )
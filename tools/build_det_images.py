"""从 dataset/raw 重建检测数据集的图片目录。

dataset/det/ 里只提交**标注文件**（labels/）+ data.yaml，图片是 raw 的副本，
为避免仓库翻倍，图片不进版本库，用本脚本一条命令重建：

    python tools/build_det_images.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).parent.parent
RAW = HERE / "dataset" / "raw"
DET = HERE / "dataset" / "det"


def main():
    n = 0
    for split in ("train", "val"):
        lab_dir = DET / "labels" / split
        img_dir = DET / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for lab in sorted(lab_dir.glob("*.txt")):
            cls, stem = lab.stem.split("_", 1)          # dog_dog-1 -> dog, dog-1
            src = RAW / cls / f"{stem}.jpg"
            dst = img_dir / f"{lab.stem}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
                n += 1
            else:
                print(f"  [缺图] {lab.name} -> 期望 {src}")
    print(f"重建完成：{n} 张 -> {DET/'images'}")


if __name__ == "__main__":
    main()

"""训练 YOLO11s 检测模型（15 类）。

对应考核提示：
    统一建议模型：YOLO11s
    预训练类直接用官方权重；非预训练类在同一模型上用自建标注集微调。

这里做的正是「在同一个 YOLO11s 上，用自建标注集微调 15 类」：
- 12 个 COCO 类用官方权重自动出框（desktop 按提示用 tv 近似）
- 3 个非 COCO 类（penguin / tafei / nailong）用 GrabCut 前景分割自动出框
所有类别合并成一个 15 类模型，推理时一次拿到类别。

用法：
    python tools/train_yolo11s.py            # 默认 40 epochs
    python tools/train_yolo11s.py 60
"""
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent.parent
DATA = HERE / "dataset" / "det" / "data.yaml"

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 416


def main():
    # 从官方预训练权重起步（COCO 12 类已经会认，微调只是补上 3 个新类并适配画风）
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=8,
        patience=12,
        workers=4,
        device="cpu",
        project=str(HERE / "runs"),
        name="det15_yolo11s",
        exist_ok=True,
        seed=42,
        cache=False,
        verbose=True,
        # 小数据集，增强开足
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
        degrees=0.0, translate=0.12, scale=0.5, shear=0.0,
        fliplr=0.5, mosaic=0.8, close_mosaic=8,
    )
    best = HERE / "runs" / "det15_yolo11s" / "weights" / "best.pt"
    print("\n训练完成:", best, "存在:", best.exists())
    if best.exists():
        m = YOLO(str(best))
        metrics = m.val(data=str(DATA), imgsz=IMGSZ, device="cpu")
        print("mAP50:", metrics.box.map50, " mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()

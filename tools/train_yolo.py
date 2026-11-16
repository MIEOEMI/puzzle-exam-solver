"""训练 YOLOv8 分类模型（15 类）。

为什么用分类而不是检测：
  站点素材是整张照片，没有标注框；而拼图任务是「每格属于哪一类」，
  分类模型完全够用，且省掉标注成本。

为什么必须自己训：
  站点源码 catalog.js 写明 penguin / tafei / nailong 不在 COCO 里，
  预训练权重无法识别，考核要求就是让候选人微调。
"""
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "dataset" / "cls"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
IMGSZ = 224


def main():
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=32,
        patience=15,
        project=str(HERE / "runs"),
        name="cls15",
        exist_ok=True,
        seed=42,
        verbose=True,
    )
    best = HERE / "runs" / "cls15" / "weights" / "best.pt"
    print("\n训练完成，权重:", best, "存在:", best.exists())
    if best.exists():
        m = YOLO(str(best))
        metrics = m.val(data=str(DATA))
        print("验证:", metrics)


if __name__ == "__main__":
    main()

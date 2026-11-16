"""YOLO 类别识别：给照片块和照片格打类别标签。

作用
----
拼图块是参考图的局部切片，同一张图的不同块内容相似，光靠像素相关度容易混淆。
类别信息提供一个独立的、语义层面的约束：**块属于哪一类，就应该落在同一类的格子里**。

模型
----
`models/puzzle_yolov8n_cls15.pt`，15 类，由 tools/train_yolo.py 训练得出
（388 张素材，验证 top1 = 1.000）。

考核站点源码已注明 penguin / tafei / nailong 不在 COCO 中，
所以必须自训，不能直接用预训练权重。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class Classifier:
    """YOLOv8 分类模型的薄封装。"""

    def __init__(self, weights=None, imgsz=224):
        from ultralytics import YOLO

        if weights is None:
            weights = Path(__file__).resolve().parent.parent / "models" / "puzzle_yolov8n_cls15.pt"
        self.path = Path(weights)
        if not self.path.exists():
            raise FileNotFoundError(f"找不到模型权重: {self.path}")
        self.model = YOLO(str(self.path))
        self.names = self.model.names
        self.imgsz = imgsz
        self._cache = {}

    def predict(self, img_bgr, key=None):
        """返回 (类别名, 置信度)。带缓存，同一块只推理一次。"""
        if key is not None and key in self._cache:
            return self._cache[key]
        res = self.model.predict(img_bgr, imgsz=self.imgsz, verbose=False)
        probs = res[0].probs
        idx = int(probs.top1)
        out = (self.names[idx], float(probs.top1conf))
        if key is not None:
            self._cache[key] = out
        return out

    def predict_many(self, images, keys=None):
        return [self.predict(im, None if keys is None else keys[i])
                for i, im in enumerate(images)]

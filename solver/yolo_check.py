"""YOLO 类别识别：给照片块和照片格打类别标签。

作用
----
拼图块是参考图的局部切片，同一张图的不同块内容相似，光靠像素相关度容易混淆。
类别信息提供一个独立的、语义层面的约束：**块属于哪一类，就应该落在同一类的格子里**，
在匈牙利指派的代价矩阵里作为加分项。

模型
----
优先用检测模型 `models/puzzle_yolo11s_det15.pt` —— 对应考核提示的「统一建议模型 YOLO11s」：
12 个 COCO 类用官方权重自动出框，3 个非 COCO 类（penguin / tafei / nailong）
用自建标注集在同一个模型上微调。
若检测模型不存在，退回分类模型 `models/puzzle_yolov8n_cls15.pt`。

为什么只需要类别、不需要框
------------------------
拼图定位靠 OpenCV（布局复现 + 模板匹配），YOLO 只回答"这块是什么"，
所以对整块图推理、取置信度最高的框的类别即可。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
DET_WEIGHTS = HERE / "models" / "puzzle_yolo11s_det15.pt"
CLS_WEIGHTS = HERE / "models" / "puzzle_yolov8n_cls15.pt"


class Classifier:
    """YOLO 封装：检测模型取最高分框的类别，分类模型取 top1。"""

    def __init__(self, weights=None, imgsz=640, conf=0.10):
        from ultralytics import YOLO

        if weights is None:
            weights = DET_WEIGHTS if DET_WEIGHTS.exists() else CLS_WEIGHTS
        self.path = Path(weights)
        if not self.path.exists():
            raise FileNotFoundError(f"找不到模型权重: {self.path}")
        self.model = YOLO(str(self.path))
        self.task = self.model.task                     # 'detect' 或 'classify'
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf = conf
        self._cache = {}

    def predict(self, img_bgr, key=None):
        """返回 (类别名, 置信度)。带缓存，同一块只推理一次。"""
        if key is not None and key in self._cache:
            return self._cache[key]
        if self.task == "classify":
            res = self.model.predict(img_bgr, imgsz=min(self.imgsz, 224),
                                     verbose=False)[0]
            out = (self.names[int(res.probs.top1)], float(res.probs.top1conf))
        else:
            res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf,
                                     verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                out = ("unknown", 0.0)
            else:
                i = int(np.argmax(res.boxes.conf.cpu().numpy()))
                out = (self.names[int(res.boxes.cls[i])], float(res.boxes.conf[i]))
        if key is not None:
            self._cache[key] = out
        return out

    def predict_many(self, images, keys=None):
        return [self.predict(im, None if keys is None else keys[i])
                for i, im in enumerate(images)]

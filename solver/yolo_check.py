"""这里打包了一下YOLO"""

from __future__ import annotations

from pathlib import Path

import numpy as np
#--------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent
DET_WEIGHTS = HERE / "models" / "puzzle_yolo11s_det15.pt"
CLS_WEIGHTS = HERE / "models" / "puzzle_yolov8n_cls15.pt"
#-------------------------------------------------------

class Classifier:


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

    def _parse_result(self, res):
        if self.task == "classify":
            return (self.names[int(res.probs.top1)], float(res.probs.top1conf))
        if res.boxes is None or len(res.boxes) == 0:
            return ("unknown", 0.0)
        confs = res.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        return (self.names[int(res.boxes.cls[i])], float(confs[i]))

    def predict(self, img_bgr, key=None):
        """返回 (类别名, 置信度)。带缓存，同一块只推理一次。"""
        if key is not None and key in self._cache:
            return self._cache[key]
        sz = min(self.imgsz, 224) if self.task == "classify" else self.imgsz
        kw = {} if self.task == "classify" else {"conf": self.conf}
        res = self.model.predict(img_bgr, imgsz=sz, verbose=False, **kw)[0]
        out = self._parse_result(res)
        if key is not None:
            self._cache[key] = out
        return out

    def predict_many(self, images, keys=None):
        """批量推理：把未缓存的图一次性送进 YOLO，避免逐张调用的 Python/PyTorch 开销。"""
        if not images:
            return []
        out = [None] * len(images)
        todo_idx, todo_imgs = [], []
        for i, im in enumerate(images):
            k = None if keys is None else keys[i]
            if k is not None and k in self._cache:
                out[i] = self._cache[k]
            else:
                todo_idx.append(i)
                todo_imgs.append(im)
        if todo_imgs:
            sz = min(self.imgsz, 224) if self.task == "classify" else self.imgsz
            kw = {} if self.task == "classify" else {"conf": self.conf}
            results = self.model.predict(todo_imgs, imgsz=sz, verbose=False, **kw)
            for idx, res in zip(todo_idx, results):
                val = self._parse_result(res)
                out[idx] = val
                if keys is not None and keys[idx] is not None:
                    self._cache[keys[idx]] = val
        return out

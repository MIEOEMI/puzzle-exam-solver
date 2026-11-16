"""自动生成 YOLO 检测标注（自建标注集）。

为什么这样标注
--------------
考核提示写明：预训练类直接用官方权重，**非预训练类（企鹅/永雏塔菲/奶龙）
用自建标注集微调同一个模型**。所以我们要给 15 类都准备检测框。

标注策略分两档：
1. COCO 类（12 类）：跑官方 YOLO11s 权重，取与该类匹配的检测框
2. 非 COCO 类（3 类）：官方权重认不出来，用 GrabCut 前景分割自动出框
   （这些素材主体居中且占比大，GrabCut 以中心矩形为初值足够稳定）

产出 YOLO 格式：
    dataset/det/images/<split>/<name>.jpg
    dataset/det/labels/<split>/<name>.txt    # cls cx cy w h  （归一化）
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent.parent
RAW = HERE / "dataset" / "raw"
OUT = HERE / "dataset" / "det"
WEIGHTS = Path("/home/agent/workspace/hermes-new/home/profiles/lab/cache/scratch/yolo11s.pt")

# 类顺序 = 题库顺序
CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei", "nailong"]
# COCO 里对应的类名（desktop 在 COCO 里没有，用 tv 近似 —— 与题库提示一致）
COCO_ALIAS = {"dog": "dog", "cat": "cat", "elephant": "elephant", "desktop": "tv",
              "bicycle": "bicycle", "car": "car", "chair": "chair", "cup": "cup",
              "airplane": "airplane", "backpack": "backpack",
              "umbrella": "umbrella", "suitcase": "suitcase"}
NON_COCO = {"penguin", "tafei", "nailong"}


def grabcut_box(img, iters=5):
    """用 GrabCut 以前景方式估计主体框（归一化 cx cy w h）。"""
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(sw * 0.06), int(sh * 0.06), int(sw * 0.88), int(sh * 0.88))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        fg = np.zeros((sh, sw), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    ys, xs = np.nonzero(fg)
    if len(xs) < 0.05 * sh * sw:            # 分割失败 → 退回中心大框
        return 0.5, 0.5, 0.86, 0.86
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # GrabCut 常把主体边缘（头发/帽子/手）切掉一点，外扩 5% 补偿
    pad_x, pad_y = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    x0, x1 = max(0, x0 - pad_x), min(sw - 1, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(sh - 1, y1 + pad_y)
    cx, cy = (x0 + x1) / 2 / sw, (y0 + y1) / 2 / sh
    bw, bh = (x1 - x0) / sw, (y1 - y0) / sh
    return (float(np.clip(cx, 0.02, 0.98)), float(np.clip(cy, 0.02, 0.98)),
            float(np.clip(bw, 0.05, 0.99)), float(np.clip(bh, 0.05, 0.99)))


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    names = model.names

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    report = {}
    for ci, cls in enumerate(CLASSES):
        files = sorted((RAW / cls).glob("*.jpg"))
        random.shuffle(files)
        n_val = max(1, int(len(files) * 0.2))
        got = {"pretrained": 0, "grabcut": 0, "center": 0}
        for i, f in enumerate(files):
            split = "val" if i < n_val else "train"
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            box = None
            if cls not in NON_COCO:
                target = COCO_ALIAS[cls]
                res = model.predict(img, imgsz=640, verbose=False, conf=0.15)[0]
                best = None
                for b in res.boxes:
                    nm = names[int(b.cls)]
                    if nm == target:
                        conf = float(b.conf)
                        if best is None or conf > best[0]:
                            best = (conf, [float(v) for v in b.xywhn[0].tolist()])
                if best is not None:
                    box = best[1]
                    got["pretrained"] += 1
            if box is None:
                if cls in NON_COCO:
                    box = grabcut_box(img)
                    got["grabcut"] += 1
                else:
                    box = (0.5, 0.5, 0.86, 0.86)     # 官方权重没检测到 → 中心大框兜底
                    got["center"] += 1
            stem = f"{cls}_{f.stem}"
            shutil.copy2(f, OUT / "images" / split / f"{stem}.jpg")
            cx, cy, bw, bh = box
            (OUT / "labels" / split / f"{stem}.txt").write_text(
                f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        report[cls] = got
        print(f"{cls:9s} n={len(files):3d} 预训练框={got['pretrained']:3d} "
              f"GrabCut={got['grabcut']:3d} 中心兜底={got['center']:3d}")

    # 不写死绝对路径：data.yaml 与 images/ 同级，换机器照样能用
    (OUT / "data.yaml").write_text(
        "train: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (OUT / "label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    n_tr = len(list((OUT / "images" / "train").glob("*.jpg")))
    n_va = len(list((OUT / "images" / "val").glob("*.jpg")))
    print(f"\n总计 train {n_tr} / val {n_va}，类别 {len(CLASSES)}")


if __name__ == "__main__":
    main()

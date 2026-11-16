"""把 raw 素材整理成 YOLO 分类数据集（train/val 按 8:2 分层抽样）。

输出结构：
    dataset/cls/train/<class>/*.jpg
    dataset/cls/val/<class>/*.jpg
"""
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "dataset" / "raw"
OUT = HERE / "dataset" / "cls"
SEED = 42
VAL_RATIO = 0.2


def main():
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    keys = sorted([d.name for d in RAW.iterdir() if d.is_dir()])
    summary = {}
    for key in keys:
        files = sorted((RAW / key).glob("*.jpg"))
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_RATIO))
        val, train = files[:n_val], files[n_val:]
        for split, group in (("train", train), ("val", val)):
            d = OUT / split / key
            d.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, d / f.name)
        summary[key] = {"train": len(train), "val": len(val)}
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    tot_tr = sum(v["train"] for v in summary.values())
    tot_va = sum(v["val"] for v in summary.values())
    print(f"\n类别 {len(keys)}  train {tot_tr}  val {tot_va}")


if __name__ == "__main__":
    main()

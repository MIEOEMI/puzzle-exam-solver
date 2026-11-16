"""下载站点全部素材图，按类别分目录，用于训练 YOLO 分类模型。

素材来源：https://puzzle.moyunliuyin.me/assets/catalog/<key>/<key>-<n>.jpg
站点源码 catalog.js 里给了每类的 key、label、数量、是否 COCO 预训练可覆盖。
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://puzzle.moyunliuyin.me/assets/catalog"
VER = "20261115n"
OUT = Path(__file__).parent / "dataset" / "raw"

# (key, label, count, in_coco)
CATEGORIES = [
    ("dog", "Dog", 40, True),
    ("cat", "Cat", 40, True),
    ("elephant", "Elephant", 40, True),
    ("desktop", "Desktop computer", 40, True),
    ("bicycle", "Bicycle", 20, True),
    ("car", "Car", 20, True),
    ("chair", "Chair", 20, True),
    ("cup", "Cup", 20, True),
    ("airplane", "Airplane", 20, True),
    ("backpack", "Backpack", 20, True),
    ("umbrella", "Umbrella", 20, True),
    ("suitcase", "Suitcase", 20, True),
    ("penguin", "Penguin", 39, False),
    ("tafei", "永雏塔菲", 13, False),
    ("nailong", "奶龙", 16, False),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (compatible; puzzle-dataset/1.0)"
    total = ok = fail = 0
    manifest = []
    for key, label, count, coco in CATEGORIES:
        d = OUT / key
        d.mkdir(parents=True, exist_ok=True)
        got = 0
        for i in range(1, count + 1):
            total += 1
            fp = d / f"{key}-{i}.jpg"
            if fp.exists() and fp.stat().st_size > 1000:
                got += 1; ok += 1
                manifest.append({"path": str(fp.relative_to(OUT.parent)), "key": key,
                                 "label": label, "in_coco": coco})
                continue
            url = f"{BASE}/{key}/{key}-{i}.jpg?v={VER}"
            for attempt in range(3):
                try:
                    r = sess.get(url, timeout=30)
                    if r.status_code == 200 and len(r.content) > 1000:
                        fp.write_bytes(r.content)
                        got += 1; ok += 1
                        manifest.append({"path": str(fp.relative_to(OUT.parent)), "key": key,
                                         "label": label, "in_coco": coco})
                        break
                    time.sleep(0.5)
                except Exception as e:
                    time.sleep(1)
            else:
                fail += 1
                print(f"  [fail] {key}-{i}")
        print(f"{key:10s} {got}/{count}  {'(COCO)' if coco else '(需微调)'}")

    (OUT.parent / "manifest.json").write_text(
        json.dumps({"categories": [{"key": k, "label": l, "count": c, "in_coco": b}
                                   for k, l, c, b in CATEGORIES],
                    "items": manifest,
                    "total": total, "ok": ok, "fail": fail}, ensure_ascii=False, indent=1))
    print(f"\n总计 {ok}/{total} 张，失败 {fail}")


if __name__ == "__main__":
    main()

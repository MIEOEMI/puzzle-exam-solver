#!/usr/bin/env python3
"""一键连续演示脚本：依次跑完 基础题(9关) -> 附加题重叠(9关) -> 附加题居中(9关)，共 27 关。

用法：
    python run_demo_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver"))

from console import force_utf8
force_utf8()

import run_exam
import run_overlap

SUITE_PAUSE = 5.0  # 每道大题之间的停顿秒数


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ev_root = HERE / "runs" / "evidence" / f"all_{stamp}"

    steps = [
        ("基础题 · 全部 9 关（Part 1/2/3 × 9/16/25）",
         lambda: run_exam.main(auto_close=True, evidence_dir=ev_root / "basic")),
        ("附加题 A · 初始重叠开局（全部 9 关）",
         lambda: run_overlap.main(mode="overlap", auto_close=True, evidence_dir=ev_root / "overlap")),
        ("附加题 B · 初始居中开局（全部 9 关）",
         lambda: run_overlap.main(mode="center", auto_close=True, evidence_dir=ev_root / "center")),
    ]

    results = []
    for idx, (title, runner) in enumerate(steps):
        print("\n" + "=" * 68)
        print(f"▶ 正在演示：{title}")
        print("=" * 68, flush=True)
        try:
            rc = runner()
        except Exception as e:
            print(f"[错误] {title} 执行异常：{type(e).__name__}: {e}")
            rc = 1
        results.append((title, rc == 0))
        if idx < len(steps) - 1 and SUITE_PAUSE > 0:
            time.sleep(SUITE_PAUSE)

    print("\n" + "=" * 68)
    print("演示汇总：")
    for title, ok in results:
        print(f"  {'✅ 通关' if ok else '❌ 未完成'}  {title}")
    print(f"网站 JSON 完成证明目录：{ev_root}")
    print("=" * 68)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())

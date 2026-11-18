#!/usr/bin/env python3
"""一键连续演示脚本：基础题 + 附加题 A（初始重叠）+ 附加题 B（初始居中）。

用法：
    python run_demo_all.py                     # 依次演示：基础题(Part3·5×5) -> 附加题重叠 -> 附加题居中
    python run_demo_all.py --bonus-diff 9      # 基础题跑 5×5，两个附加题跑 3×3（更快）
    python run_demo_all.py --all-basic         # 先把基础题 Part 1 / 2 / 3 各演示一局，再演示两个附加题
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver"))
try:
    from console import force_utf8
    force_utf8()
except Exception:
    pass


def run_step(title: str, cmd: list[str], pause: float = 2.0) -> int:
    print("\n" + "=" * 68)
    print(f"▶ 正在演示：{title}")
    print(f"  命令：{' '.join(cmd)}")
    print("=" * 68, flush=True)
    r = subprocess.run(cmd, cwd=str(HERE))
    if pause > 0:
        time.sleep(pause)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="一键连续演示基础题与两个附加题（默认均跑全部 9 关）")
    ap.add_argument("--basic-part", type=int, default=None, help="基础题指定部分（不传则默认跑全部 Part 1/2/3）")
    ap.add_argument("--basic-diff", type=int, default=None, help="基础题指定难度（不传则默认跑全部 9/16/25）")
    ap.add_argument("--bonus-part", type=int, default=None, help="附加题指定部分（不传则默认跑全部 Part 1/2/3）")
    ap.add_argument("--bonus-diff", type=int, default=None, help="附加题指定难度（不传则默认跑全部 9/16/25）")
    ap.add_argument("--pause", type=float, default=2.0, help="每题之间的停顿秒数")
    args = ap.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if args.basic_part is not None or args.basic_diff is not None:
        cmd_basic = [py, "run_basic.py", "--auto-close"]
        if args.basic_part is not None:
            cmd_basic += ["--part", str(args.basic_part)]
        if args.basic_diff is not None:
            cmd_basic += ["--diff", str(args.basic_diff)]
        steps.append(("基础题（指定关卡）", cmd_basic))
    else:
        steps.append((
            "基础题 · 全部 9 关（Part 1/2/3 × 9/16/25）",
            [py, "run_basic.py", "--auto-close"],
        ))

    cmd_ov = [py, "run_bonus_overlap.py", "--auto-close"]
    cmd_ct = [py, "run_bonus_center.py", "--auto-close"]
    if args.bonus_part is not None:
        cmd_ov += ["--part", str(args.bonus_part)]
        cmd_ct += ["--part", str(args.bonus_part)]
    if args.bonus_diff is not None:
        cmd_ov += ["--diff", str(args.bonus_diff)]
        cmd_ct += ["--diff", str(args.bonus_diff)]

    steps.append(("附加题 A · 初始重叠开局（全部 9 关）" if args.bonus_diff is None else "附加题 A · 初始重叠开局", cmd_ov))
    steps.append(("附加题 B · 初始居中开局（全部 9 关）" if args.bonus_diff is None else "附加题 B · 初始居中开局", cmd_ct))

    results = []
    for title, cmd in steps:
        rc = run_step(title, cmd, pause=args.pause)
        results.append((title, rc == 0))

    print("\n" + "=" * 68)
    print("演示汇总：")
    for title, ok in results:
        print(f"  {'✅ 通关' if ok else '❌ 未完成'}  {title}")
    print("=" * 68)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())

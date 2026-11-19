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


def run_step(title: str, cmd: list[str], pause: float = 5.0) -> int:
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
    ap.add_argument("--pause", type=float, default=5.0, help="每道大题之间的停顿秒数（默认 5.0s）")
    ap.add_argument("--level-pause", type=float, default=5.0, help="每一局通关后的中间停顿秒数（默认 5.0s）")
    ap.add_argument("--pause-enter", action="store_true", help="每局通关并导出证明后等待终端按回车继续")
    ap.add_argument("--evidence-dir", default=None, help="统一指定网站 JSON 完成证明导出根目录")
    ap.add_argument("--no-evidence", action="store_true", help="关闭自动导出网站 JSON 完成证明")
    args = ap.parse_args()

    py = sys.executable
    common_flags: list[str] = ["--pause", str(args.level_pause)]
    if args.pause_enter:
        common_flags.append("--pause-enter")
    if args.no_evidence:
        common_flags.append("--no-evidence")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    ev_root = None if args.no_evidence else (
        Path(args.evidence_dir) if args.evidence_dir else (HERE / "runs" / "evidence" / f"all_{stamp}")
    )

    def _with_ev(base_cmd: list[str], sub_name: str) -> list[str]:
        cmd = list(base_cmd) + common_flags
        if ev_root is not None:
            cmd += ["--evidence-dir", str(ev_root / sub_name)]
        return cmd

    steps: list[tuple[str, list[str]]] = []

    if args.basic_part is not None or args.basic_diff is not None:
        cmd_basic = [py, "run_basic.py", "--auto-close"]
        if args.basic_part is not None:
            cmd_basic += ["--part", str(args.basic_part)]
        if args.basic_diff is not None:
            cmd_basic += ["--diff", str(args.basic_diff)]
        steps.append(("基础题（指定关卡）", _with_ev(cmd_basic, "basic")))
    else:
        steps.append((
            "基础题 · 全部 9 关（Part 1/2/3 × 9/16/25）",
            _with_ev([py, "run_basic.py", "--auto-close"], "basic"),
        ))

    cmd_ov = [py, "run_bonus_overlap.py", "--auto-close"]
    cmd_ct = [py, "run_bonus_center.py", "--auto-close"]
    if args.bonus_part is not None:
        cmd_ov += ["--part", str(args.bonus_part)]
        cmd_ct += ["--part", str(args.bonus_part)]
    if args.bonus_diff is not None:
        cmd_ov += ["--diff", str(args.bonus_diff)]
        cmd_ct += ["--diff", str(args.bonus_diff)]

    steps.append(("附加题 A · 初始重叠开局（全部 9 关）" if args.bonus_diff is None else "附加题 A · 初始重叠开局", _with_ev(cmd_ov, "overlap")))
    steps.append(("附加题 B · 初始居中开局（全部 9 关）" if args.bonus_diff is None else "附加题 B · 初始居中开局", _with_ev(cmd_ct, "center")))

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

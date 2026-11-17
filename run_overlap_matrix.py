"""附加题矩阵：每个难度连续多轮 + 轮次间隔。

用法
----
    python run_overlap_matrix.py --mode overlap --rounds 2 --pause 5
    python run_overlap_matrix.py --mode center  --rounds 1 --pause 5 --diffs 9,16,25

输出
----
    <outdir>/SUMMARY.md   一行一局：模式 / 难度 / 结果 / 用时
    <outdir>/<mode>_d<diff>_r<n>.log   每局完整日志

设计说明
--------
本脚本只做「编排」：循环难度、循环轮次、轮间 sleep，
每一局都通过 subprocess 调 run_overlap.py（独立浏览器进程），
所以某局崩溃不会污染下一局。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "solver" / "run_overlap.py"


def parse_diffs(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_pause(s: str) -> float:
    """支持 '5' 或 '3-8'（区间随机），与旧项目保持一致。"""
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        import random
        return random.uniform(float(a), float(b))
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overlap", "center"], default="overlap")
    ap.add_argument("--diffs", default="9,16,25", help="逗号分隔，例如 9,16,25")
    ap.add_argument("--rounds", type=int, default=2, help="每个难度连续跑几轮")
    ap.add_argument("--pause", default="5", help="难度之间的等待秒数，支持 3-8 区间")
    ap.add_argument("--round-gap", default="2", help="同一难度内两轮之间的等待秒数")
    ap.add_argument("--theme", default="mixed")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-seconds", type=float, default=600, help="单局超时")
    ap.add_argument("--ident-min", default="0.55")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"/tmp/bonus_matrix_{args.mode}")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "SUMMARY.md"
    summary.write_text(
        f"# 附加题矩阵 · {args.mode}\n\n"
        f"| 难度 | 轮次 | 结果 | 用时 | 状态 |\n|---|---|---|---|---|\n",
        encoding="utf-8")

    diffs = parse_diffs(args.diffs)
    rows: list[tuple[int, int, str, str, str]] = []

    for di, diff in enumerate(diffs):
        for rnd in range(1, args.rounds + 1):
            log = outdir / f"{args.mode}_d{diff}_r{rnd}.log"
            cmd = [sys.executable, str(RUNNER),
                   "--mode", args.mode, "--diff", str(diff),
                   "--theme", args.theme, "--ident-min", args.ident_min]
            print(f"[{args.mode}] 难度 {diff} 第 {rnd}/{args.rounds} 轮 -> {log.name}",
                  flush=True)
            t0 = time.time()
            try:
                with log.open("w", encoding="utf-8") as fh:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=args.max_seconds, cwd=str(HERE))
                status = "done"
            except subprocess.TimeoutExpired:
                status = "timeout"
            dt = time.time() - t0

            text = log.read_text(encoding="utf-8", errors="replace")
            res = ""
            for line in text.splitlines():
                if "结果" in line and "/" in line:
                    res = line.strip().strip("=").strip()
            if not res:
                res = f"无结果行（{status}）"
            won = "🎉" in text and "未完成" not in text
            rows.append((diff, rnd, res, f"{dt:.1f}s", "✅ 通关" if won else "❌ 未完成"))
            with summary.open("a", encoding="utf-8") as fh:
                fh.write(f"| {diff} | {rnd} | {res} | {dt:.1f}s | "
                         f"{'✅ 通关' if won else '❌ 未完成'} |\n")

            # 同难度内轮间隔；换难度时用 --pause
            last_round_of_diff = (rnd == args.rounds)
            last_diff = (di == len(diffs) - 1)
            if last_round_of_diff and not last_diff:
                wait = parse_pause(args.pause)
                print(f"  ── 换难度，等待 {wait:.1f}s ──", flush=True)
            elif not last_round_of_diff:
                wait = parse_pause(args.round_gap)
                print(f"  ── 同难度下一轮，等待 {wait:.1f}s ──", flush=True)
            else:
                wait = 0
            if wait:
                time.sleep(wait)

    # 汇总统计
    total = len(rows)
    passed = sum(1 for r in rows if r[4].startswith("✅"))
    with summary.open("a", encoding="utf-8") as fh:
        fh.write(f"\n**总计 {passed}/{total} 局通关**\n")

    print(f"\n=== 矩阵完成 {passed}/{total} 局通关 ===")
    for r in rows:
        print(f"  难度 {r[0]:>2} 第{r[1]}轮  {r[2]:<22} {r[3]:>8}  {r[4]}")
    print(f"汇总: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

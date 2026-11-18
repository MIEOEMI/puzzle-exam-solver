"""附加题入口 B：初始居中开局（碎片堆在棋盘中央，需先分开）。

求解器：run_overlap（独立黑底检验台 + 边带暂存 + 匈牙利全局指派 + 白块齿型约束）。
用法：
    python run_bonus_center.py                     # 默认依次跑完初始居中全部 9 关（Part 1/2/3 × 9/16/25）
    python run_bonus_center.py --part 3 --diff 25  # 单独跑混合 · 5×5
"""
import sys
from pathlib import Path

# 运行器放在仓库根目录，求解器/浏览器实现都在 solver/ 下
sys.path.insert(0, str(Path(__file__).resolve().parent / "solver"))

import run_overlap


def _strip_mode(argv):
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--mode":
            i += 2
            continue
        if a.startswith("--mode="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main():
    sys.argv = [sys.argv[0], "--mode", "center"] + _strip_mode(sys.argv[1:])
    run_overlap.main()


if __name__ == "__main__":
    main()

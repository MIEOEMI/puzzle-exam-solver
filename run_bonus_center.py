"""附加题入口 B：初始居中开局（碎片堆在棋盘中央，需先分开）。

求解器：run_overlap（同重叠，但暂存区与剥离策略不同），与基础题不同。
用法：
    python run_bonus_center.py                     # 混合 · 5×5
    python run_bonus_center.py --diff 9            # 混合 · 3×3
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

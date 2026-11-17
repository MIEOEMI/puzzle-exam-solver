"""附加题入口 A：初始重叠开局（碎片互相压住，需先分开）。

求解器：run_overlap（堆叠剥离 + 暂存识别 + 白块试放），与基础题不同。
用法：
    python run_bonus_overlap.py                    # 混合 · 5×5
    python run_bonus_overlap.py --diff 16          # 混合 · 4×4
    python run_bonus_overlap.py --theme white      # 全白
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
    sys.argv = [sys.argv[0], "--mode", "overlap"] + _strip_mode(sys.argv[1:])
    run_overlap.main()


if __name__ == "__main__":
    main()

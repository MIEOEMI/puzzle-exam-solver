"""基础题入口：Part 1 / 2 / 3 × 9 / 16 / 25 格。

求解器：solver_core.Analysis（模板匹配 + 匈牙利全局指派 + 锚点顺序）
        white_shape（白块齿型约束）
用法：
    python run_basic.py                          # 默认 Part3 · 5×5
    python run_basic.py --part 1 --diff 25       # 全照片 5×5
    python run_basic.py --part 2 --diff 16       # 全白 4×4
    python run_basic.py --no-yolo                # 纯 OpenCV
"""
import sys

import run_exam


def main():
    # 基础题入口固定走 run_exam 的流程，不接受附加题的 --mode
    argv = [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = [sys.argv[0]] + argv
    run_exam.main()


if __name__ == "__main__":
    main()

"""基础题入口：Part 1 / 2 / 3 × 9 / 16 / 25 格。

求解器：solver_core.Analysis（模板匹配 + 匈牙利全局指派 + 锚点顺序）
        white_shape（白块齿型约束）
用法：
    python run_basic.py                          # 默认一口气依次跑完全部 9 个关卡（Part 1/2/3 × 9/16/25）
    python run_basic.py --part 3 --diff 25       # 单独跑混合 5×5
    python run_basic.py --part 1                 # 跑第 1 部分的 9 / 16 / 25
    python run_basic.py --no-yolo                # 纯 OpenCV
"""
import sys
from pathlib import Path

# 运行器放在仓库根目录，求解器/浏览器实现都在 solver/ 下
sys.path.insert(0, str(Path(__file__).resolve().parent / "solver"))

import run_exam


def main():
    # 基础题入口固定走 run_exam 的流程，不接受附加题的 --mode
    argv = [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = [sys.argv[0]] + argv
    run_exam.main()


if __name__ == "__main__":
    main()

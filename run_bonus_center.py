import sys
from pathlib import Path

# 运行器放在仓库根目录，求解器/浏览器实现都在 solver/ 下
sys.path.insert(0, str(Path(__file__).resolve().parent / "solver"))

import run_overlap


if __name__ == "__main__":
    sys.exit(run_overlap.main(mode="center"))

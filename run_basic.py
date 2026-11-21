import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "solver"))

import run_exam


if __name__ == "__main__":
    sys.exit(run_exam.main())

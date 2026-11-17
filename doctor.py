#!/usr/bin/env python3
"""环境自检：一条命令查出为什么跑不起来。

用法（在仓库根目录执行）：
    python doctor.py

它会逐项检查 Python 版本、依赖包、Chrome、模型文件、导入链，
每项都给出 **怎么办** 的具体命令。
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, BAD, WARN = "✅", "❌", "⚠️ "

problems: list[str] = []


def line(tag: str, name: str, detail: str = "", fix: str = ""):
    print(f"{tag} {name:<22} {detail}")
    if tag != OK and fix:
        print(f"    → {fix}")


def check_python():
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro} ({platform.system()})"
    if v < (3, 9):
        line(BAD, "Python 版本", s, "需要 3.9+，请升级 Python")
        problems.append("python")
    else:
        line(OK, "Python 版本", s)
    print(f"   解释器: {sys.executable}")


def check_pkg(mod: str, pip_name: str, need: str = ""):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "")
        line(OK, f"依赖 {mod}", str(ver))
        return True
    except Exception as e:
        line(BAD, f"依赖 {mod}", f"{type(e).__name__}: {e}",
             f"pip install {pip_name}{(' ' + need) if need else ''}")
        problems.append(mod)
        return False


def check_imports():
    """检查仓库内部的导入链（最常见的坑）。"""
    sys.path.insert(0, str(ROOT / "solver"))
    try:
        import solver_core  # noqa: F401
        line(OK, "导入 solver_core")
    except Exception as e:
        line(BAD, "导入 solver_core", f"{type(e).__name__}: {e}",
             "确认 solver/solver_core.py 存在；不要在别的目录运行")
        problems.append("solver_core")

    try:
        import white_shape  # noqa: F401
        line(OK, "导入 white_shape")
    except Exception as e:
        line(BAD, "导入 white_shape", f"{type(e).__name__}: {e}")
        problems.append("white_shape")

    for runner in ("run_basic.py", "run_bonus_overlap.py", "run_bonus_center.py"):
        p = ROOT / runner
        if not p.exists():
            line(BAD, f"运行器 {runner}", "文件不存在", "git pull 拉取最新代码")
            problems.append(runner)
            continue
        r = subprocess.run([sys.executable, str(p), "--help"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            line(OK, f"运行器 {runner}")
        else:
            err = (r.stderr or r.stdout).strip().splitlines()
            line(BAD, f"运行器 {runner}", err[-1] if err else "退出码 %d" % r.returncode,
                 f"单独运行 python {runner} --help 看完整报错")
            problems.append(runner)


def check_chrome():
    cands = []
    if platform.system() == "Windows":
        for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
            cands.append(Path(p))
    elif platform.system() == "Darwin":
        cands.append(Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            w = shutil.which(name)
            if w:
                cands.append(Path(w))
    found = next((c for c in cands if c.exists()), None)
    if found:
        line(OK, "Chrome 浏览器", str(found))
    else:
        line(WARN, "Chrome 浏览器", "未在默认位置找到",
             "装 Google Chrome；或把 playwright 的 channel 改成 chromium：\n"
             "       pip install playwright && playwright install chromium\n"
             "       然后在 run_exam.py / run_overlap.py 里把 channel=\"chrome\" 改成 channel=None")


def check_files():
    for rel, desc in [("models/puzzle_yolo11s_det15.pt", "YOLO11s 检测模型"),
                      ("models/puzzle_yolov8n_cls15.pt", "分类模型"),
                      ("dataset/raw", "训练素材"),
                      ("solver/solver_core.py", "求解器核心"),
                      ("browser/place_exam.py", "浏览器驱动")]:
        p = ROOT / rel
        if p.exists():
            if p.is_dir():
                n = len(list(p.rglob("*.jpg")))
                line(OK, desc, f"{rel} ({n} 张)")
            else:
                line(OK, desc, f"{rel} ({p.stat().st_size // 1024} KB)")
        else:
            line(BAD, desc, f"缺失 {rel}", "git pull（模型较大，确认 LFS/完整克隆）")
            problems.append(rel)


def check_display():
    """有头模式需要显示环境；Linux 无 DISPLAY 会起不来。"""
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        line(WARN, "显示环境 DISPLAY", "未设置",
             "无界面服务器可加 --headful 去掉前先设 DISPLAY，或用 xvfb-run python run_basic.py ...")


def main():
    print("=" * 64)
    print("拼图考核求解器 · 环境自检")
    print(f"仓库目录: {ROOT}")
    print("=" * 64)
    check_python()
    print()
    print("--- 依赖包 ---")
    check_pkg("numpy", "numpy")
    check_pkg("cv2", "opencv-python")
    check_pkg("scipy", "scipy")
    check_pkg("playwright", "playwright")
    has_torch = check_pkg("torch", "torch")
    if has_torch:
        check_pkg("ultralytics", "ultralytics")
    print()
    print("--- 文件 ---")
    check_files()
    print()
    print("--- 浏览器 ---")
    check_chrome()
    check_display()
    print()
    print("--- 导入链 ---")
    check_imports()
    print("=" * 64)
    if problems:
        print(f"发现 {len(problems)} 个问题：{', '.join(problems)}")
        print("\n最常见的两步修复：")
        print("  pip install -r requirements.txt")
        print("  python -m playwright install chrome")
        return 1
    print("环境检查全部通过。可以运行：")
    print("  python run_basic.py --part 3 --diff 25")
    return 0


if __name__ == "__main__":
    sys.exit(main())

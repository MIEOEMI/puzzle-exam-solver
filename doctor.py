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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, BAD, WARN = "✅", "❌", "⚠️ "

# Windows 控制台默认 cp936，直接打印 ✅ 会 UnicodeEncodeError，自检脚本自己先崩
sys.path.insert(0, str(ROOT / "solver"))
try:
    from console import force_utf8
except Exception:                     # solver/ 缺失时也要能跑完自检
    def force_utf8():
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
force_utf8()

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
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
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
        env = os.environ
        for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                  str(Path(env.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"),
                  str(Path(env.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe")):
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


def check_launch():
    """真正的判据不是'文件存在'，而是 Playwright 能不能把 Chrome 拉起来。"""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        line(BAD, "Chrome 能否启动", f"{type(e).__name__}: {e}", "pip install playwright")
        problems.append("launch")
        return
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
            ver = b.version
            b.close()
        line(OK, "Chrome 能否启动", f"Chromium {ver}")
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:70]
        line(BAD, "Chrome 能否启动", msg,
             "装 Google Chrome；或改用自带 chromium：playwright install chromium，\n"
             "      并把 run_exam.py 里的 channel=\"chrome\" 改成 channel=None")
        problems.append("launch")


def check_site():
    """站点可达性 + 素材下载速度。开局卡在「准备中…」多半是这里慢。"""
    import time as _t
    try:
        import requests
    except Exception:
        return
    url = "https://puzzle.moyunliuyin.me/?mode=exam"
    t0 = _t.time()
    try:
        r = requests.get(url, timeout=20)
        dt = _t.time() - t0
    except Exception as e:
        line(BAD, "站点可达", f"{type(e).__name__}", "检查网络/代理，浏览器能打开再说")
        problems.append("site")
        return
    if r.status_code != 200:
        line(BAD, "站点可达", f"HTTP {r.status_code}", "考核站是否在维护？")
        problems.append("site")
        return
    line(OK, "站点可达", f"HTTP 200 · {dt:.2f}s")

    # 站点版本：每次改版都可能让坐标/判据失效，先记下来
    import re as _re
    build = _re.search(r"main\.js\?v=([0-9a-z]+)", r.text)
    if build:
        line(OK, "站点版本", build.group(1))

    # 新版站点有 StateGuard（1.2s 对账，判篡改就锁死拖拽）。
    # 素材走本地 + 开局重试就是为了绕开它自己的加载窗口，见 README §六。
    try:
        js = requests.get("https://puzzle.moyunliuyin.me/src/security/StateGuard.js",
                          timeout=20)
        if js.status_code == 200 and "verifyState" in js.text:
            line(OK, "状态对账 StateGuard", "存在（自动重试已处理）")
    except Exception:
        pass

    # 3 张素材的下载耗时，折算开局要下的 55 张
    sample = ["dog/dog-1.jpg", "cat/cat-2.jpg", "penguin/penguin-3.jpg"]
    times = []
    for s in sample:
        t = _t.time()
        try:
            requests.get(f"https://puzzle.moyunliuyin.me/assets/catalog/{s}?v=20261115n",
                         timeout=20)
            times.append(_t.time() - t)
        except Exception:
            pass
    if times:
        avg = sum(times) / len(times)
        est = avg * 55 / 6
        detail = f"单张 {avg*1000:.0f}ms，开局约需 {est:.0f}s（55 张 / 6 并发）"
        if est > 60:
            line(WARN, "素材下载速度", detail, "网络偏慢，用 --start-timeout 120 再跑")
        else:
            line(OK, "素材下载速度", detail)


def start_test(headful=False):
    """实测一次开局要多久 —— 「一直卡在选择难度界面」就是这一段。"""
    import json as _json
    sys.path.insert(0, str(ROOT / "solver"))
    try:
        import run_exam
    except Exception as e:
        print(f"{BAD} 无法导入 run_exam：{e}")
        return 1
    from playwright.sync_api import sync_playwright
    print("\n--- 实测开局（Part 3 · 5×5）---")
    t0 = time.time()
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=not headful, args=["--no-sandbox"])
        page = b.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1.0)
        page.goto(run_exam.URL, wait_until="load", timeout=60000)
        page.wait_for_timeout(2500)
        try:
            cur, tot = run_exam.start_exam(page, 3, 25, timeout=120)
            print(f"{OK} 开局成功：{cur}/{tot}，总耗时 {time.time()-t0:.1f}s")
            ok = 0
        except Exception as e:
            print(f"{BAD} 开局失败：{e}")
            ok = 1
        b.close()
    return ok


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
    check_launch()
    check_display()
    print()
    print("--- 站点 ---")
    check_site()
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
    # python doctor.py              环境自检
    # python doctor.py --start-test 只实测一次开局（定位"卡在选择难度界面"）
    if "--start-test" in sys.argv:
        sys.exit(start_test(headful="--headful" in sys.argv))
    sys.exit(main())

"""控制台输出修正（Windows 专用问题）。

Windows 控制台默认是 cp936(GBK)，脚本里一打印 ✅ / ❌ / 🎉 就抛
    UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'
进程当场死掉——表现就是"跑不起来"，而且报错位置往往离真正原因很远。

force_utf8() 做两件事：
  1) 把 stdout/stderr 切到 UTF-8 + errors="replace"（永不因编码崩）
  2) Windows 上把控制台代码页切成 65001，中文和 emoji 都能正常显示
"""
from __future__ import annotations

import sys

__all__ = ["force_utf8"]


def force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

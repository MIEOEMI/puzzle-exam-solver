"""实测各 (part, diff) 组合的真实块数与棋盘尺寸，确定诱饵数。"""
import base64, json, sys, time
import cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from playwright.sync_api import sync_playwright
from solver_core import compute_layout

def to_bgr(u):
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",",1)[1]), np.uint8), cv2.IMREAD_COLOR)

JS_START = """([part, diff]) => {
  const p = document.querySelector('input[name="exam-part"][value="'+part+'"]');
  const d = document.querySelector('input[name="exam-difficulty"][value="'+diff+'"]');
  if (!p || !d) return 'no-radio';
  p.checked = true; p.dispatchEvent(new Event('change', {bubbles:true}));
  d.checked = true; d.dispatchEvent(new Event('change', {bubbles:true}));
  const b = document.getElementById('btn-exam-start'); if (!b) return 'no-btn';
  b.click(); return 'ok';
}"""
JS_PROG = "() => document.getElementById('progress-display').innerText"

def board_rect(canvas):
    """在画布上测出棋盘矩形（深蓝底 BGR≈42,23,15）。"""
    target = np.array([42, 23, 15], np.int16)
    d = np.abs(canvas.astype(np.int16) - target).sum(axis=2)
    m = (d < 30).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return None
    i = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA])
    x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    return (x, y, w, h, int(st[i, cv2.CC_STAT_AREA]))

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
    pg = br.new_page(viewport={"width":1280,"height":800})
    for part in (1,2,3):
        for diff in (9,16,25):
            pg.goto("https://puzzle.moyunliuyin.me/?mode=exam", wait_until="load")
            pg.wait_for_timeout(1200)
            r = pg.evaluate(JS_START, [part, diff])
            pg.wait_for_timeout(2500)
            prog = pg.evaluate(JS_PROG)
            cv_ = to_bgr(pg.evaluate("() => document.getElementById('puzzle-canvas').toDataURL('image/png')"))
            rect = board_rect(cv_)
            # 两种假设下的棋盘尺寸
            pred = {}
            for nd in (0, round(diff*0.2)):
                L = compute_layout(cv_.shape[1], cv_.shape[0], int(diff**.5), int(diff**.5), nd)
                pred[nd] = (round(L["board"]["w"]), round(L["board"]["h"]))
            print(f"part={part} diff={diff:2d} 开局={prog:8s} 实测棋盘={rect[2] if rect else None}x{rect[3] if rect else None} "
                  f"预测(诱饵0)={pred[0]} 预测(诱饵{round(diff*0.2)})={pred[round(diff*0.2)]}")
    br.close()

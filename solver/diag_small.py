import base64, json, sys, time
import cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from playwright.sync_api import sync_playwright
from solver_core import compute_layout, fg_mask

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

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
    pg = br.new_page(viewport={"width":1280,"height":800})
    for part, diff in [(3,9),(3,16),(3,25)]:
        pg.goto("https://puzzle.moyunliuyin.me/?mode=exam", wait_until="load")
        pg.wait_for_timeout(1200)
        print("start:", pg.evaluate(JS_START, [part, diff]))
        pg.wait_for_timeout(3000)
        prog = pg.evaluate("() => document.getElementById('progress-display').innerText")
        cv_ = to_bgr(pg.evaluate("() => document.getElementById('puzzle-canvas').toDataURL('image/png')"))
        rows = cols = int(round(diff**0.5)); nd = round(diff*0.2)
        L = compute_layout(cv_.shape[1], cv_.shape[0], rows, cols, nd)
        pw, ph = L["piece"]["w"], L["piece"]["h"]
        filled = 0; total = 0
        for slot in L["slots"]:
            inner = cv_[int(round(slot["y"])):int(round(slot["y"]+ph)),
                        int(round(slot["x"])):int(round(slot["x"]+pw))]
            if inner.size == 0: continue
            total += 1
            if fg_mask(inner).mean() >= 0.5: filled += 1
        print(f"part={part} diff={diff} 进度={prog} 画布={cv_.shape[1]}x{cv_.shape[0]} "
              f"槽位={total} 有内容={filled} 棋盘={L['board']['w']:.0f}x{L['board']['h']:.0f} 格子={pw:.1f}x{ph:.1f}")
        cv2.imwrite(f"/tmp/diag_{part}_{diff}.png", cv_)
    br.close()

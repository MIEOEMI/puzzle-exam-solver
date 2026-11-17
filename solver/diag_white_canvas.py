import base64, sys, time
import cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from playwright.sync_api import sync_playwright

def to_bgr(u):
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",",1)[1]), np.uint8), cv2.IMREAD_COLOR)

JS_SWITCH = "() => { const b=document.getElementById('btn-exam-switch'); if(b){b.click(); return 'switch';} return 'none'; }"
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
    for diff in (9, 16, 25):
        pg.goto("https://puzzle.moyunliuyin.me/?mode=exam", wait_until="load")
        pg.wait_for_timeout(1200)
        print(diff, "switch:", pg.evaluate(JS_SWITCH)); pg.wait_for_timeout(500)
        print(diff, "start:", pg.evaluate(JS_START, [3, diff]))
        for wait in (1500, 3000, 6000):
            pg.wait_for_timeout(wait if wait == 1500 else 1500)
            cv_ = to_bgr(pg.evaluate("() => document.getElementById('puzzle-canvas').toDataURL('image/png')"))
            prog = pg.evaluate("() => document.getElementById('progress-display').innerText")
            print(f"   diff={diff} t={wait}ms 进度={prog} 画布均值={cv_.mean():.1f} 非白像素占比={(cv_.mean(axis=2)<250).mean():.3f}")
    br.close()

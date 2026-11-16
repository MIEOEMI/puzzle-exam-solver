#!/usr/bin/env python3
"""附加题：初始重叠 / 初始居中 开局求解器（Part 3 混合 · 5×5）。

为什么不能用主流程的做法
------------------------
主流程假设每块拼图都躺在自己的托盘槽位上，按槽位切图就能拿到块。
这两种开局把块**堆在一起**：重叠模式堆在左右两条边带，居中模式堆在棋盘中间。
大量块几乎完全被压住，逐格模板匹配连"位置"都不可信 —— 抓下去抓到的往往是
盖在上面的另一块，于是同一个分数反复失败。

所以这里换个思路：**先摊开，再识别**
------------------------------------
1. 从堆里随便抓一块（抓到的一定是最上面那块，至于是哪块无所谓）
2. 拖到一块**空闲暂存区**（重叠模式用空棋盘中央，居中模式用空边带）
   —— 它摊开后完整露出来
3. 这时再做模板匹配，就能准确认出它是哪个格子
4. 拖到它自己的格子：能吸附就吸附；暂时不满足锚点条件就先放在那儿
5. 反复直到堆里没块
6. 最后对"放在格子上但没吸附"的块，按从外到内重新拿起来放一次（此时锚点已满足）

纯白块没有纹理，模板认不出来，单独用白色连通域处理。

用法
----
    python run_overlap.py --mode overlap     # 初始重叠
    python run_overlap.py --mode center      # 初始居中
    python run_overlap.py --mode overlap --headful --video ./vid
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from solver_core import compute_layout, is_white_cell, split_reference

URL = "https://puzzle.moyunliuyin.me/"

JS_RECT = """() => { const r = document.getElementById('puzzle-canvas').getBoundingClientRect();
  return JSON.stringify({left:r.left, top:r.top, w:r.width, h:r.height}); }"""
JS_PNG = "() => document.getElementById('puzzle-canvas').toDataURL('image/png')"
JS_REF = """() => { const im = document.getElementById('preview-img');
  return JSON.stringify({src: im.src, w: im.naturalWidth, h: im.naturalHeight}); }"""
JS_PROGRESS = "() => { const e = document.getElementById('progress-display'); return e ? e.innerText : ''; }"

JS_SETUP = """([theme, diff, mode]) => {
  const sel = document.getElementById('puzzle-select');
  const dif = document.getElementById('difficulty-select');
  if (!sel || !dif) return 'missing-select';
  sel.value = theme; sel.dispatchEvent(new Event('change', {bubbles: true}));
  dif.value = String(diff); dif.dispatchEvent(new Event('change', {bubbles: true}));
  const ov = document.getElementById('overlap-toggle');
  const ct = document.getElementById('center-toggle');
  if (ov) { ov.checked = (mode === 'overlap'); ov.dispatchEvent(new Event('change', {bubbles:true})); }
  if (ct) { ct.checked = (mode === 'center'); ct.dispatchEvent(new Event('change', {bubbles:true})); }
  const btn = document.getElementById('btn-shuffle');
  if (btn) btn.click();
  return JSON.stringify({theme: sel.value, diff: dif.value,
                         overlap: ov ? ov.checked : null, center: ct ? ct.checked : null});
}"""


def to_bgr(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def parse_progress(txt):
    try:
        a, b = txt.replace(" ", "").split("/")
        return int(a), int(b)
    except Exception:
        return 0, 0


def drag(page, sx, sy, tx, ty, steps=10, delay=0.0):
    page.mouse.move(sx, sy)
    page.mouse.down()
    if delay:
        time.sleep(delay)
    page.mouse.move(tx, ty, steps=steps)
    page.mouse.up()


def in_any(x, y, regions):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in regions)


def find_content_point(canvas, regions, bg, step=5, min_diff=45,
                       forbid=None, forbid_r=26):
    """在堆叠区里找一个"确实压在块身上"的点。

    关键：游戏是按**块轮廓**做命中判定的。
    抓到透明角、块边缘的投影、或是块之间的缝隙，mousedown 不会命中任何块
    （实测：抓在 (55,5) 这种灰色阴影上，整块纹丝不动）。
    所以要求 9×9 邻域内每个采样点都明显不同于底色。
    """
    h, w = canvas.shape[:2]
    forbid = forbid or []
    c = canvas.astype(np.int16)
    for y in range(4, h - 4, step):
        for x in range(4, w - 4, step):
            if not in_any(x, y, regions):
                continue
            if any((x - fx) ** 2 + (y - fy) ** 2 < forbid_r ** 2 for fx, fy in forbid):
                continue
            ok = True
            for dy in (-4, 0, 4):
                for dx in (-4, 0, 4):
                    py, px_ = y + dy, x + dx
                    d = (abs(c[py, px_, 0] - bg[0]) + abs(c[py, px_, 1] - bg[1])
                         + abs(c[py, px_, 2] - bg[2]))
                    if d < min_diff * 2:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return x, y
    return None


def grab_here(page, rect, cv, gx, gy, probe=12):
    """按住这个点小幅移动，看画面变不变，判断是否真的抓到了块。

    必要性：游戏按**块轮廓**做命中判定，落点在透明角、投影、缝隙上时
    mousedown 不会抓住任何块（实测抓在 (100,8) 画面纹丝不动）。
    与其猜"什么点算命中"，不如按下后试探一下，没抓住就换下一个点。
    """
    px0 = cv[gy, gx].astype(int)
    page.mouse.move(rect["left"] + gx, rect["top"] + gy)
    page.mouse.down()
    page.mouse.move(rect["left"] + gx + probe, rect["top"] + gy + probe, steps=4)
    time.sleep(0.05)
    after = to_bgr(page.evaluate(JS_PNG))
    px1 = after[gy, gx].astype(int)
    return int(np.abs(px1 - px0).sum()) > 40


def candidate_points(canvas, regions, bg, min_dist=13, step=9):
    """找出"一定在块内部"的点：对非底色区域做距离变换，取离边缘最远的点。

    比盲目铺点快一个数量级 —— 距离变换一次就能拿到所有"深在块里"的位置，
    不需要逐点按下去试。
    """
    c = canvas.astype(np.int16)
    diff = np.abs(c - np.array(bg, np.int16)).sum(axis=2)
    mask = (diff > 75).astype(np.uint8)
    reg = np.zeros_like(mask)
    for x0, y0, x1, y1 in regions:
        reg[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 1
    mask &= reg
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    pts = []
    h, w = mask.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            if dist[y, x] >= min_dist:
                pts.append((float(dist[y, x]), x, y))
    pts.sort(key=lambda t: -t[0])          # 越深越优先
    return [(x, y) for _, x, y in pts]


def identify(canvas, tmpl_of, remaining, pw, ph, centre, radius):
    """在暂存台附近认出这块是哪个格子。返回 (score, cell)。"""
    cx, cy = centre
    region = [(cx - radius, cy - radius, cx + radius, cy + radius)]
    best = (-1.0, None)
    for cell in remaining:
        tmpl = cv2.resize(tmpl_of[cell], (int(round(pw)), int(round(ph))),
                          interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = [int(v) for v in region[0]]
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(canvas.shape[1], x1)
        y1 = min(canvas.shape[0], y1)
        if x1 - x0 < tmpl.shape[1] or y1 - y0 < tmpl.shape[0]:
            continue
        res = cv2.matchTemplate(canvas[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), cell,
                    (x0 + loc[0] + tmpl.shape[1] / 2, y0 + loc[1] + tmpl.shape[0] / 2))
    if best[1] is None:
        return -1.0, None, None
    return best[0], best[1], best[2]


def white_blobs(canvas, regions, min_area=500):
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    for x0, y0, x1, y1 in regions:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(canvas.shape[1], int(x1)), min(canvas.shape[0], int(y1))
        sub = hsv[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] = (((sub[:, :, 1] < 12) & (sub[:, :, 2] > 244))
                              .astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = [{"cx": float(cents[i][0]), "cy": float(cents[i][1]),
            "area": int(stats[i, cv2.CC_STAT_AREA])}
           for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    out.sort(key=lambda b: -b["area"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overlap", "center"], default="overlap")
    ap.add_argument("--theme", default="mixed")
    ap.add_argument("--diff", type=int, default=25)
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--slow", type=float, default=0.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--ident-min", type=float, default=0.55, help="识别为照片块的最低分")
    args = ap.parse_args()

    rows = cols = int(round(args.diff ** 0.5))
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headful,
                                    args=["--no-sandbox", "--disable-dev-shm-usage"])
        if args.video:
            Path(args.video).mkdir(parents=True, exist_ok=True)
            ctx = browser.new_context(viewport={"width": 1280, "height": 800},
                                      record_video_dir=args.video,
                                      record_video_size={"width": 1280, "height": 800})
            page = ctx.new_page()
        else:
            ctx = None
            page = browser.new_page(viewport={"width": 1280, "height": 800})

        page.goto(URL, wait_until="load", timeout=60000)
        page.wait_for_timeout(1500)
        print("配置:", page.evaluate(JS_SETUP, [args.theme, args.diff, args.mode]))
        page.wait_for_timeout(3000)

        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        print(f"开局: {cur}/{tot} 模式={args.mode}")

        rect = json.loads(page.evaluate(JS_RECT))
        canvas = to_bgr(page.evaluate(JS_PNG))
        ref = to_bgr(json.loads(page.evaluate(JS_REF))["src"])

        # 重叠/居中开局没有托盘槽位，棋盘按满尺寸
        L = compute_layout(canvas.shape[1], canvas.shape[0], rows, cols, 0, overlap=True)
        pw, ph = L["piece"]["w"], L["piece"]["h"]
        bx, by = L["board"]["x"], L["board"]["y"]
        bw, bh = L["board"]["w"], L["board"]["h"]
        CW, CH = canvas.shape[1], canvas.shape[0]
        print(f"画布 {CW}x{CH} 棋盘 {bw:.0f}x{bh:.0f} 格子 {pw:.1f}x{ph:.1f}")

        cells = split_reference(ref, rows, cols)
        white_cells = [(c["row"], c["col"]) for c in cells if is_white_cell(c["img"])]
        tmpl_of = {(c["row"], c["col"]): c["img"] for c in cells
                   if not is_white_cell(c["img"])}
        remaining = set(tmpl_of)
        print(f"参考格: 照片 {len(remaining)} 纯白 {len(white_cells)} -> {white_cells}")

        # 堆叠区 与 暂存台
        if args.mode == "overlap":
            pile = [(0, 0, bx + pw * 0.7, CH), (bx + bw - pw * 0.7, 0, CW, CH)]
            stage = (bx + bw / 2, by + bh / 2)          # 空棋盘中央
        else:
            pile = [(bx - pw * 0.4, by - ph * 0.4, bx + bw + pw * 0.4, by + bh + ph * 0.4)]
            stage = (max(8, bx * 0.5), CH / 2)          # 空的左侧边带
        bg = canvas[2, 2].astype(np.int16)
        print(f"暂存台 {stage[0]:.0f},{stage[1]:.0f}  底色 {bg.tolist()}")

        def cell_centre(cell):
            return (bx + cell[1] * pw + pw / 2, by + cell[0] * ph + ph / 2)

        placed = set()
        staged = {}       # cell -> (x, y) 放在格子上但没吸附
        white_park = []   # 白块临时停放点

        def anchor_ok(cell):
            r, c = cell
            if r in (0, rows - 1) or c in (0, cols - 1):
                return True
            return any((r + dr, c + dc) in placed
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

        if shots:
            cv2.imwrite(str(shots / f"01_start_{args.mode}.png"), canvas)

        # ---------- 阶段 A：摊开 + 识别 + 归位 ----------
        for step in range(len(remaining) + 8):
            cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
            if cur >= tot:
                break
            canvas = to_bgr(page.evaluate(JS_PNG))
            forbid = ([cell_centre((r, c)) for r in range(rows) for c in range(cols)]
                      if args.mode == "center" else [])
            cands = [p for p in candidate_points(canvas, pile, bg)
                     if not any((p[0] - fx) ** 2 + (p[1] - fy) ** 2 < 26 ** 2
                                for fx, fy in forbid)]
            grabbed = None
            for gx, gy in cands:
                if grab_here(page, rect, canvas, gx, gy):
                    grabbed = (gx, gy)
                    break
            if grabbed is None:
                print("  试遍候选点都没抓住块（可能已经放完）")
                break
            gx, gy = grabbed
            # 已经按住并偏移了 12px，直接拖到暂存台
            page.mouse.move(rect["left"] + stage[0], rect["top"] + stage[1], steps=12)
            page.mouse.up()
            time.sleep(0.12)
            cur2, _ = parse_progress(page.evaluate(JS_PROGRESS))
            if cur2 > cur:
                print(f"  抓 ({gx},{gy}) 拖到暂存台时意外吸附 ✅ {cur2}/{tot}")
                continue

            canvas = to_bgr(page.evaluate(JS_PNG))
            sc, cell, loc = identify(canvas, tmpl_of, remaining, pw, ph,
                                     stage, radius=int(max(pw, ph) * 1.1))
            if cell is None or sc < args.ident_min:
                # 认不出 = 纯白块（或露得太少）。白块没有纹理，直接按排除法试放。
                print(f"  暂存台上的块识别不出来（最高 {sc:.2f}）→ 按白块处理")
                done = False
                for wc in [c for c in white_cells if c not in placed and anchor_ok(c)]:
                    wx, wy = cell_centre(wc)
                    c0, _ = parse_progress(page.evaluate(JS_PROGRESS))
                    drag(page, rect["left"] + stage[0], rect["top"] + stage[1],
                         rect["left"] + wx, rect["top"] + wy, delay=args.slow)
                    time.sleep(0.10)
                    c1, _ = parse_progress(page.evaluate(JS_PROGRESS))
                    if c1 > c0:
                        placed.add(wc)
                        print(f"    白块 → {wc} ✅ {c1}/{tot}")
                        done = True
                        break
                    drag(page, rect["left"] + wx, rect["top"] + wy,
                         rect["left"] + stage[0], rect["top"] + stage[1], delay=args.slow)
                    time.sleep(0.06)
                if not done:
                    # 还没到能放的时候，先停在棋盘角上让出暂存台
                    px_ = bx + 14 + len(white_park) * 10
                    py_ = by + bh - 14
                    drag(page, rect["left"] + stage[0], rect["top"] + stage[1],
                         rect["left"] + px_, rect["top"] + py_, delay=args.slow)
                    white_park.append((px_, py_))
                    print(f"    白块暂存到 ({px_:.0f},{py_:.0f})")
                continue

            cx_t, cy_t = cell_centre(cell)
            # 从暂存台上这块的**真实中心**抓起来，落到格心，偏移正好为 0
            gx2, gy2 = loc if loc is not None else (stage[0], stage[1])
            drag(page, rect["left"] + gx2, rect["top"] + gy2,
                 rect["left"] + cx_t, rect["top"] + cy_t, delay=args.slow)
            time.sleep(0.12)
            cur3, _ = parse_progress(page.evaluate(JS_PROGRESS))
            remaining.discard(cell)
            if cur3 > cur2:
                placed.add(cell)
                print(f"  ✅ 识别 {cell} score={sc:.2f} 归位吸附 {cur3}/{tot}")
            else:
                staged[cell] = (cx_t, cy_t)
                print(f"  → 识别 {cell} score={sc:.2f} 放在格子上但未吸附（等锚点）")

        # ---------- 阶段 B：按从外到内重新放一次未吸附的 ----------
        for _ in range(len(staged) + 4):
            pend = [c for c in staged if anchor_ok(c)]
            if not pend:
                break
            cell = pend[0]
            sx, sy = staged.pop(cell)
            cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
            # 拿起来晃一下再放下，触发重新判定锚点
            drag(page, rect["left"] + sx, rect["top"] + sy,
                 rect["left"] + sx + 6, rect["top"] + sy + 6)
            time.sleep(0.08)
            drag(page, rect["left"] + sx + 6, rect["top"] + sy + 6,
                 rect["left"] + sx, rect["top"] + sy)
            time.sleep(0.12)
            cur2, _ = parse_progress(page.evaluate(JS_PROGRESS))
            if cur2 > cur:
                placed.add(cell)
                print(f"  ✅ {cell} 补吸附 {cur2}/{tot}")
            else:
                staged[cell] = (sx, sy)
                print(f"  ❌ {cell} 补吸附失败")

        # ---------- 阶段 C：白块 ----------
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        print(f"\n照片阶段结束 {cur}/{tot} 待放白格 {[c for c in white_cells if c not in placed]}")
        guard = 0
        while cur < tot and guard < 100:
            guard += 1
            left = [c for c in white_cells if c not in placed]
            if not left:
                break
            cv_now = to_bgr(page.evaluate(JS_PNG))
            centres = [cell_centre((r, c)) for r in range(rows) for c in range(cols)]
            pts = [b for b in white_blobs(cv_now, pile if args.mode == "overlap" else
                                          [(bx - pw * .4, by - ph * .4,
                                            bx + bw + pw * .4, by + bh + ph * .4)])
                   if not any(abs(b["cx"] - ex) < 5 and abs(b["cy"] - ey) < 5
                              for ex, ey in centres)]
            if not pts:
                print("  找不到可抓的白块")
                break
            cell = next((c for c in left if anchor_ok(c)), None)
            if cell is None:
                print("  白格都不满足锚点")
                break
            # 关键：不能每次都抓同一个白块（pts[0] 永远最大）。
            # 按轮次轮换抓取点，才能在"白块 × 白格"里试出正确组合。
            b = pts[guard % len(pts)]
            before = cur
            sx, sy = rect["left"] + b["cx"], rect["top"] + b["cy"]
            tx, ty = cell_centre(cell)
            drag(page, sx, sy, rect["left"] + tx, rect["top"] + ty, delay=args.slow)
            time.sleep(0.10)
            cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
            if cur > before:
                placed.add(cell)
                print(f"  白块 ({b['cx']:.0f},{b['cy']:.0f}) → {cell} ✅ {cur}/{tot}")
            else:
                drag(page, rect["left"] + tx, rect["top"] + ty, sx, sy, delay=args.slow)
                white_cells.append(white_cells.pop(0))
                print(f"  白块 ({b['cx']:.0f},{b['cy']:.0f}) 试 {cell} ✗（已拖回）")

        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if shots:
            cv2.imwrite(str(shots / f"02_final_{args.mode}.png"),
                        to_bgr(page.evaluate(JS_PNG)))
            page.screenshot(path=str(shots / f"03_page_{args.mode}.png"))

        print(f"\n=== {args.mode} 结果 {cur}/{tot} | 用时 {round(time.time()-t0,1)}s "
              f"| {'🎉 通关' if cur == tot else '未完成'} ===")

        if args.headful:
            input("按回车关闭…")
        if ctx is not None:
            page.close()
            ctx.close()
        browser.close()
        return 0 if cur == tot else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""附加题：初始重叠 / 初始居中 开局求解器（支持 Part 1 / 2 / 3 × 9 / 16 / 25 全部 9 关）。

用法
----
    python run_bonus_overlap.py                  # 默认依次跑完初始重叠全部 9 关（Part 1/2/3 × 9/16/25）
    python run_bonus_center.py                   # 默认依次跑完初始居中全部 9 关（Part 1/2/3 × 9/16/25）
    python run_bonus_overlap.py --part 3 --diff 9  # 单独跑某一关
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from console import force_utf8
from solver_core import (
    PAD, anchor_order, compute_layout, fg_mask, is_white_cell, is_white_piece,
    locate_piece, split_reference, template_score,
)
from white_shape import board_color, cell_required_edges, white_piece_info

force_utf8()

URL = "https://puzzle.moyunliuyin.me/?mode=exam"

JS_RECT = """() => { const r = document.getElementById('puzzle-canvas').getBoundingClientRect();
  return JSON.stringify({left:r.left, top:r.top, w:r.width, h:r.height}); }"""
JS_PNG = "() => document.getElementById('puzzle-canvas').toDataURL('image/png')"
JS_REF = """() => { const im = document.getElementById('preview-img');
  return JSON.stringify({src: im.src, w: im.naturalWidth, h: im.naturalHeight}); }"""
JS_PROGRESS = "() => { const e = document.getElementById('progress-display'); return e ? e.innerText : ''; }"

JS_START_BONUS = """([part, diff, mode]) => {
  const p = document.querySelector('input[name="exam-part"][value="'+part+'"]');
  const d = document.querySelector('input[name="exam-difficulty"][value="'+diff+'"]');
  if (!p || !d) return 'no-radio';
  p.checked = true; p.dispatchEvent(new Event('change', {bubbles:true}));
  d.checked = true; d.dispatchEvent(new Event('change', {bubbles:true}));
  const ov = document.getElementById('exam-overlap-toggle');
  const ct = document.getElementById('exam-center-toggle');
  if (ov) { ov.checked = (mode === 'overlap'); ov.dispatchEvent(new Event('change', {bubbles:true})); }
  if (ct) { ct.checked = (mode === 'center'); ct.dispatchEvent(new Event('change', {bubbles:true})); }
  const btn = document.getElementById('btn-exam-start');
  if (!btn) return 'no-button';
  btn.click();
  return 'ok';
}"""


def to_bgr(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def load_ref(page, tries=20):
    for _ in range(tries):
        info = json.loads(page.evaluate(JS_REF))
        src = (info.get("src") or "").strip()
        if src.startswith("data:"):
            return to_bgr(src)
        if src.startswith("http"):
            import requests
            r = requests.get(src, timeout=20)
            return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        page.wait_for_timeout(200)
    raise RuntimeError("参考图未取到")


def parse_progress(txt):
    try:
        a, b = txt.replace(" ", "").split("/")
        return int(a), int(b)
    except Exception:
        return 0, 0


def drag(page, sx, sy, tx, ty, steps=None, delay=0.0):
    if steps is None:
        steps = 10 if delay else 5
    page.mouse.move(sx, sy)
    page.mouse.down()
    if delay:
        time.sleep(delay)
    page.mouse.move(tx, ty, steps=steps)
    page.mouse.up()


def css_xy(rect, x, y):
    return (rect["left"] + x * rect.get("sx", 1.0),
            rect["top"] + y * rect.get("sy", 1.0))


def screen_metrics(browser, fallback=(1600, 900, 1.0)):
    try:
        ctx = browser.new_context(no_viewport=True)
        pg = ctx.new_page()
        w, h = pg.evaluate("() => [screen.availWidth, screen.availHeight]")
        dpr = pg.evaluate("() => devicePixelRatio")
        ctx.close()
        return int(w), int(h), float(dpr)
    except Exception:
        return fallback


def maximize_window(page):
    try:
        cdp = page.context.new_cdp_session(page)
        wid = cdp.send("Browser.getWindowForTarget")["windowId"]
        cdp.send("Browser.setWindowBounds",
                 {"windowId": wid, "bounds": {"windowState": "maximized"}})
    except Exception:
        pass


def candidate_points(canvas, regions, bg, min_dist=8, step=7, forbid=None):
    """对堆叠区非底色掩码做距离变换，按离边缘由深到浅返回候选抓取点。"""
    forbid = forbid or []
    c = canvas.astype(np.int16)
    diff = np.abs(c - np.array(bg, np.int16)).sum(axis=2)
    mask = (diff > 38).astype(np.uint8)
    reg = np.zeros_like(mask)
    h, w = mask.shape
    for x0, y0, x1, y1 in regions:
        reg[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = 1
    mask &= reg
    for fx, fy, fr in forbid:
        cv2.circle(mask, (int(fx), int(fy)), int(fr), 0, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    pts = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            if dist[y, x] >= min_dist:
                pts.append((float(dist[y, x]), x, y))
    pts.sort(key=lambda t: -t[0])
    return [(x, y) for _, x, y in pts]


def inspect_isolated_piece(canvas, ix0, iy0, ix1, iy1, pw, ph, tmpl_of,
                           has_white_cells=True, pure_white_mode=False,
                           piece_clf=None, cell_class=None, exclude=None,
                           decoy_thr=0.74):
    """在纯黑背景的检验台区域 [ix0:ix1, iy0:iy1] 分析刚拖过来的单块拼图。"""
    sub = canvas[iy0:iy1, ix0:ix1]
    if sub.size == 0:
        return {"kind": "empty"}
    fg = fg_mask(sub, thr=18)
    area = int(fg.sum())
    if area < pw * ph * 0.16:
        return {"kind": "empty"}

    # 利用黑底检验台上单块拼图的闭合轮廓填洞，得到完整实心剪影 sil（即使深色照片也能 100% 填满本体）
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(fg)
    if cnts:
        main_cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(sil, [main_cnt], -1, 1, cv2.FILLED)
    else:
        sil = closed

    # 凸齿头部最宽为 0.44*ph / 0.44*pw，而块本体投影 >= 0.60*ph / 0.60*pw。
    # 用 0.52 阈值并对首尾取中点，即可 100% 剥离四边凸齿，把块本体中心定位到 ±1.5px 精度！
    b_cols = np.where(sil.sum(axis=0) >= ph * 0.52)[0]
    b_rows = np.where(sil.sum(axis=1) >= pw * 0.52)[0]
    ys, xs = np.where(sil > 0)
    if b_cols.size >= 10:
        cx_body = ix0 + 0.5 * (float(b_cols[0]) + float(b_cols[-1]))
    else:
        cx_body = ix0 + float(np.median(xs))
    if b_rows.size >= 10:
        cy_body = iy0 + 0.5 * (float(b_rows[0]) + float(b_rows[-1]))
    else:
        cy_body = iy0 + float(np.median(ys))

    slot_guess = {"x": cx_body - pw / 2.0, "y": cy_body - ph / 2.0}

    # 1) 判断是否为纯白块（Part 1 无白格时绝不判白；Part 2 全白模式或中心+四角纯白时判白）
    wh = ((sub[:, :, 0] > 248) & (sub[:, :, 1] > 248) & (sub[:, :, 2] > 248)).astype(np.uint8)
    if has_white_cells and int(wh.sum()) >= pw * ph * 0.48:
        valid_cols = np.where(wh.sum(axis=0) >= ph * 0.52)[0]
        valid_rows = np.where(wh.sum(axis=1) >= pw * 0.52)[0]
        if valid_cols.size >= 10 and valid_rows.size >= 10:
            w_cx = ix0 + 0.5 * (float(valid_cols[0]) + float(valid_cols[-1]))
            w_cy = iy0 + 0.5 * (float(valid_rows[0]) + float(valid_rows[-1]))
            w_slot = {"x": w_cx - pw / 2.0, "y": w_cy - ph / 2.0}
        else:
            w_slot = slot_guess
        if pure_white_mode or is_white_piece(canvas, w_slot, pw, ph)[0]:
            edges, (wcx, wcy), is_dec = white_piece_info(
                canvas, w_slot, pw, ph, pure_white_mode=pure_white_mode
            )
            return {"kind": "white", "cx": wcx, "cy": wcy, "edges": edges, "is_decoy": is_dec}

    # 2) 照片块：在 slot_guess 周围 ±25% 窗口内匹配（兼容半边全黑的深色照片块），
    #    同时用前景几何包络约束 cand_cx/cy 绝不滑出拼图块本体到黑底背景上！
    H, W = canvas.shape[:2]
    fg_x_min = ix0 + float(xs.min())
    fg_x_max = ix0 + float(xs.max())
    fg_y_min = iy0 + float(ys.min())
    fg_y_max = iy0 + float(ys.max())
    cx_lo = min(cx_body, fg_x_max - 0.76 * pw)
    cx_hi = max(cx_body, fg_x_min + 0.76 * pw)
    cy_lo = min(cy_body, fg_y_max - 0.76 * ph)
    cy_hi = max(cy_body, fg_y_min + 0.76 * ph)

    pad_x = max(10, int(round(pw * 0.25)))
    pad_y = max(10, int(round(ph * 0.25)))
    x0_g = int(round(slot_guess["x"]))
    y0_g = int(round(slot_guess["y"]))
    x0_p = max(0, x0_g - pad_x)
    y0_p = max(0, y0_g - pad_y)
    x1_p = min(W, int(round(slot_guess["x"] + pw)) + pad_x)
    y1_p = min(H, int(round(slot_guess["y"] + ph)) + pad_y)
    pad_crop = canvas[y0_p:y1_p, x0_p:x1_p]
    off_x0 = x0_p - x0_g
    off_y0 = y0_p - y0_g
    w_full, h_full = int(round(pw)), int(round(ph))

    exclude = exclude or set()
    cells = [c for c in tmpl_of if c not in exclude] or list(tmpl_of)
    scored = []
    for cell in cells:
        tmpl_full = cv2.resize(tmpl_of[cell], (w_full, h_full), interpolation=cv2.INTER_AREA)
        best_sc = -1.0
        best_c_px = (cx_body, cy_body)
        for frac in (0.08, 0.20):
            mx, my = int(round(w_full * frac)), int(round(h_full * frac))
            tmpl_in = tmpl_full[my:h_full - my, mx:w_full - mx]
            pad_in = pad_crop[my:pad_crop.shape[0] - my, mx:pad_crop.shape[1] - mx]
            if (tmpl_in.shape[0] < 8 or tmpl_in.shape[1] < 8
                    or pad_in.shape[0] < tmpl_in.shape[0]
                    or pad_in.shape[1] < tmpl_in.shape[1]):
                continue
            res = cv2.matchTemplate(pad_in, tmpl_in, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-1.0)
            rh, rw = res.shape
            rxs = np.arange(rw, dtype=np.float32)
            rys = np.arange(rh, dtype=np.float32)
            cand_cxs = slot_guess["x"] + (rxs + off_x0) + pw / 2.0
            cand_cys = slot_guess["y"] + (rys + off_y0) + ph / 2.0
            valid_x = (cand_cxs >= cx_lo) & (cand_cxs <= cx_hi)
            valid_y = (cand_cys >= cy_lo) & (cand_cys <= cy_hi)
            if valid_x.any() and valid_y.any():
                mask2d = np.outer(valid_y, valid_x)
                res = np.where(mask2d, res, -1.0)
            _, sc_val, _, loc = cv2.minMaxLoc(res)
            if float(sc_val) > best_sc:
                best_sc = float(sc_val)
                best_c_px = (
                    slot_guess["x"] + (loc[0] + off_x0) + pw / 2.0,
                    slot_guess["y"] + (loc[1] + off_y0) + ph / 2.0,
                )
        scored.append([best_sc, best_sc, cell, best_c_px])

    if not scored:
        return {"kind": "decoy", "cx": cx_body, "cy": cy_body, "score": -1.0, "raw_ncc": -1.0, "all_scores": {}}

    scored.sort(key=lambda t: -t[1])
    best_c_px = scored[0][3]
    pname = None
    if piece_clf is not None and cell_class:
        px0 = max(0, int(round(best_c_px[0] - pw / 2)))
        py0 = max(0, int(round(best_c_px[1] - ph / 2)))
        px1 = min(canvas.shape[1], int(round(px0 + pw)))
        py1 = min(canvas.shape[0], int(round(py0 + ph)))
        crop = canvas[py0:py1, px0:px1]
        if crop.size > 0:
            pname = piece_clf.predict(crop)[0]
            for item in scored:
                if cell_class.get(item[2]) == pname and item[1] >= 0.35:
                    item[0] = item[1] + 0.12

    all_scores = {item[2]: (item[0], item[1], item[3]) for item in scored}
    scored.sort(key=lambda t: -t[0])
    total_sc, raw_ncc, best_cell, best_pos = scored[0]
    if raw_ncc < decoy_thr:
        return {"kind": "decoy", "cx": cx_body, "cy": cy_body,
                "score": total_sc, "raw_ncc": raw_ncc, "pname": pname, "all_scores": all_scores}
    return {"kind": "photo", "cx": best_pos[0], "cy": best_pos[1],
            "cell": best_cell, "score": total_sc, "raw_ncc": raw_ncc,
            "pname": pname, "all_scores": all_scores}


def solve_one_bonus_level(page, part_id, diff, mode, piece_clf, cell_clf, args, shots=None):
    """在已打开的浏览器页面上完成一局附加题 (part_id, diff, mode)，返回 (cur, tot, elapsed)。"""
    from run_exam import JS_TAMPER, JS_SWITCH, wait_start_ready, exam_status, exam_running

    t_level = time.time()
    rows = cols = int(round(diff ** 0.5))
    n_decoys = max(1, round(diff * 0.2))
    part_names = {1: "照片", 2: "纯白", 3: "混合"}
    mode_names = {"overlap": "初始重叠", "center": "初始居中"}
    print("\n" + "=" * 64)
    print(f"▶ 附加题（{mode_names.get(mode, mode)}）：Part {part_id}（{part_names.get(part_id, '')}）"
          f"· {rows}×{cols}（{diff} 格 + {n_decoys} 诱饵）")
    print("=" * 64)

    for clf in (piece_clf, cell_clf):
        if clf is not None and hasattr(clf, "_cache"):
            clf._cache.clear()

    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    def _safe_goto(pg, target_url):
        for nav_i in range(1, 5):
            try:
                pg.goto(target_url, wait_until="load", timeout=60000)
                return
            except Exception as e:
                print(f"[网络] 打开页面波动 ({e})，等待重试 {nav_i}/4…")
                time.sleep(1.5)

    cur = tot = 0
    for _retry in range(1, 6):
        _safe_goto(page, URL)
        s = exam_status(page)
        if not s.get("panel_open"):
            try:
                page.evaluate(JS_SWITCH)
            except Exception:
                pass
            page.wait_for_timeout(200)
        wait_start_ready(page, 25.0)
        r_start = page.evaluate(JS_START_BONUS, [part_id, diff, mode])
        print(f"[开局 {_retry}] 点击开始 (part={part_id}, diff={diff}, mode={mode}) -> {r_start}")

        t1 = time.time()
        while time.time() - t1 < args.start_timeout:
            page.wait_for_timeout(100)
            s = exam_status(page)
            if exam_running(s, part_id, diff):
                cur, tot = parse_progress(s.get("progress", ""))
                break
        for _ in range(100):
            page.wait_for_timeout(100)
            if float(to_bgr(page.evaluate(JS_PNG)).mean()) > 5:
                break
        page.wait_for_timeout(150)
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))

        t = json.loads(page.evaluate(JS_TAMPER))
        if not t.get("tamper"):
            break
        print(f"[开局] 站点对账把这次开局判成篡改（{t['tamper']}），刷新重开 {_retry}/5")

    rect = json.loads(page.evaluate(JS_RECT))
    canvas = to_bgr(page.evaluate(JS_PNG))
    rect["sx"] = rect["w"] / canvas.shape[1]
    rect["sy"] = rect["h"] / canvas.shape[0]
    ref = load_ref(page)

    L = compute_layout(canvas.shape[1], canvas.shape[0], rows, cols, 0, overlap=True)
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    bx, by = L["board"]["x"], L["board"]["y"]
    bw, bh = L["board"]["w"], L["board"]["h"]
    CW, CH = canvas.shape[1], canvas.shape[0]
    print(f"画布 {CW}x{CH} 棋盘 ({bx:.0f},{by:.0f}) {bw:.0f}x{bh:.0f} 格子 {pw:.1f}x{ph:.1f}")

    cells = split_reference(ref, rows, cols)
    white_cells = [(c["row"], c["col"]) for c in cells if is_white_cell(c["img"])]
    tmpl_of = {(c["row"], c["col"]): c["img"] for c in cells
               if not is_white_cell(c["img"])}
    pure_white_mode = len(tmpl_of) == 0
    has_white_cells = len(white_cells) > 0

    # 为每个照片格构建两个尺度的模板：84% 常规区 + 56% 无凹槽核心区
    tmpl_resized = {}
    for cell, im_c in tmpl_of.items():
        ch_c, cw_c = im_c.shape[:2]
        t_list = []
        for frac in (0.08, 0.22):
            mx0, my0 = int(cw_c * frac), int(ch_c * frac)
            cropped = im_c[my0:ch_c - my0, mx0:cw_c - mx0]
            tw_f = max(14, int(round(pw * (1.0 - 2 * frac))))
            th_f = max(14, int(round(ph * (1.0 - 2 * frac))))
            t_list.append(cv2.resize(cropped, (tw_f, th_f), interpolation=cv2.INTER_AREA))
        tmpl_resized[cell] = t_list

    remaining_photo = set(tmpl_of)
    print(f"参考格: 照片 {len(remaining_photo)} 纯白 {len(white_cells)}")

    cell_class = {}
    if cell_clf is not None and tmpl_of:
        c_keys = list(tmpl_of.keys())
        c_imgs = [tmpl_of[k] for k in c_keys]
        preds = cell_clf.predict_many(c_imgs, keys=c_keys)
        cell_class = {k: pr[0] for k, pr in zip(c_keys, preds)}

    # 独立黑底检验台：
    #  - overlap 模式：棋盘正上方中央 (bx + bw*0.5, by*0.5) 100% 净空
    #  - center 模式：右侧边带正中央 (bx + bw + 右带宽*0.53, CH*0.50) 556x530px 100% 独占净空！
    #    （暂存位优先使用左侧边带与棋盘下方，右侧边带全程空置，彻底杜绝棋盘中心堆叠块或画布上下边界干扰）
    if mode == "overlap":
        inspect_pt = (bx + bw * 0.5, by * 0.5)
        ix0 = max(0, int(inspect_pt[0] - pw * 1.15))
        iy0 = 0
        ix1 = min(CW, int(inspect_pt[0] + pw * 1.15))
        iy1 = min(int(by - 3), int(CH))

        # 检验台必须容得下整块（含凸齿），否则块会被画布上边界裁掉，
        # 模板匹配恒为 -1.00，全部被误判成诱饵 -> 整关 0/N。
        # 实测：画布 919x590 时检验台仅 68px，而 3x3 块高 149px，必然失败。
        bench_h = iy1 - iy0
        need_h = ph * 1.35
        if bench_h < need_h:
            need_canvas_h = int(by * 2 + bh + 20)
            print(f"[检验台不足] 画布 {CW}x{CH} 棋盘高 {bh:.0f} -> 上方净空 {bench_h}px，"
                  f"而块高 {ph:.0f}px 需要 >= {need_h:.0f}px")
            print(f"             附加题依赖『棋盘上方黑底净空』做单块识别，屏幕太矮会整关失败。")
            print(f"             建议：屏幕高度 >= 1200，或用虚拟屏运行：")
            print(f'               xvfb-run -a -s "-screen 0 1920x1200x24" python run_bonus_overlap.py')
            print(f"             该关仍会继续跑，但结果大概率不完整。")
    else:
        inspect_pt = (bx + bw + (CW - (bx + bw)) * 0.53, CH * 0.50)
        ix0 = max(int(bx + bw + 24), int(inspect_pt[0] - pw * 1.20))
        iy0 = max(0, int(inspect_pt[1] - ph * 1.45))
        ix1 = CW
        iy1 = min(CH, int(inspect_pt[1] + ph * 1.45))

    # 废件丢弃点：棋盘正下方最左端
    trash_pt = (bx + pw * 0.30, CH - ph * 0.52)

    # 暂存位：
    #  - center 模式：左侧空边带 (12~30格) + 棋盘正下方横带 (5~13格)，全程不占用右侧检验台！
    #  - overlap 模式：仅内圈未锚定白块需要暂存，放在棋盘正下方横带
    side_slots = []
    step_x, step_y = pw * 1.04, ph * 1.04
    cols_side = max(1, int(max(0.0, bx - 12.0) / step_x))
    rows_side = max(1, int(CH / step_y))
    for c_s in range(cols_side):
        for r_s in range(rows_side):
            px = 6.0 + c_s * step_x + pw / 2
            py = 6.0 + r_s * step_y + ph / 2
            if px + pw / 2 <= bx - 4 and py + ph / 2 <= CH:
                side_slots.append((px, py))

    overlap_park_slots = []
    # 棋盘正下方净空区 [bx, bx+bw] x [by+bh, CH]（完全在左右 pile 之外，且绝不触碰上方检验台）
    if rows >= 5:
        y_rows = [by + bh + ph * 0.50, (by + bh + CH) * 0.50, CH - ph * 0.50]
    else:
        y_rows = [by + bh + ph * 0.48, CH - ph * 0.48]
    for ry in y_rows:
        for k in range(cols):
            px = bx + bw - pw * (0.50 + 1.00 * k)
            if math.hypot(px - trash_pt[0], ry - trash_pt[1]) >= max(pw, ph) * 0.85:
                overlap_park_slots.append((px, ry))

    if mode == "overlap":
        pile = [(0, 0, bx - 6, CH), (bx + bw + 6, 0, CW, CH)]
        bg = np.array([0, 0, 0], dtype=np.int16)
    else:
        pile = [(bx - pw * 0.08, by - ph * 0.05, bx + bw + pw * 0.08, by + bh + ph * 0.05)]
        bg = np.array([42, 23, 15], dtype=np.int16)

    def cell_centre(cell):
        return (bx + cell[1] * pw + pw / 2.0, by + cell[0] * ph + ph / 2.0)

    placed: set[tuple[int, int]] = set()
    placed_white_edges: dict[tuple[int, int], dict[str, int]] = {}
    parked_photo: dict[tuple[int, int], tuple[float, float]] = {}
    staged_photo_list: list[dict] = []
    parked_white: list[dict] = []
    failed_grabs: list[tuple[float, float, float]] = [
        (trash_pt[0], trash_pt[1], max(pw, ph) * 0.65)
    ]

    def anchor_ok(cell):
        r, c = cell
        if r in (0, rows - 1) or c in (0, cols - 1):
            return True
        return any((r + dr, c + dc) in placed
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def border_ok(edges, cell):
        r, c = cell
        if (r == 0 and edges["top"] != 0) or (r != 0 and edges["top"] == 0):
            return False
        if (r == rows - 1 and edges["bottom"] != 0) or (r != rows - 1 and edges["bottom"] == 0):
            return False
        if (c == 0 and edges["left"] != 0) or (c != 0 and edges["left"] == 0):
            return False
        if (c == cols - 1 and edges["right"] != 0) or (c != cols - 1 and edges["right"] == 0):
            return False
        return True

    NEIGH = (
        (-1, 0, "top", "bottom"),
        (1, 0, "bottom", "top"),
        (0, -1, "left", "right"),
        (0, 1, "right", "left"),
    )

    def white_compat(edges, cell):
        """返回 (冲突数, 匹配数)"""
        r, c = cell
        mm, mt = 0, 0
        for dr, dc, side, opp in NEIGH:
            nb = (r + dr, c + dc)
            if nb in placed_white_edges:
                ne = placed_white_edges[nb][opp]
                if edges[side] != 0 and ne != 0:
                    if edges[side] + ne == 0:
                        mt += 1
                    else:
                        mm += 1
        return mm, mt

    def snap_from(pos, cell):
        nonlocal cur, tot
        tx, ty = cell_centre(cell)
        before = cur
        if abs(pos[0] - tx) < 6 and abs(pos[1] - ty) < 6:
            drag(page, *css_xy(rect, tx, ty), *css_xy(rect, tx + 18, ty), delay=args.slow)
            time.sleep(0.02)
            drag(page, *css_xy(rect, tx + 18, ty), *css_xy(rect, tx, ty), delay=args.slow)
        else:
            drag(page, *css_xy(rect, pos[0], pos[1]), *css_xy(rect, tx, ty), delay=args.slow)
        time.sleep(0.02 if not args.slow else args.slow + 0.05)
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if cur > before:
            placed.add(cell)
            remaining_photo.discard(cell)
            parked_photo.pop(cell, None)
            print(f"  ✅ 归位 {cell} 进度 {cur}/{tot}")
            return True
        return False

    def flush_anchored_parked():
        progressed = True
        while progressed:
            progressed = False
            for cell, pos in list(parked_photo.items()):
                if cell not in placed and anchor_ok(cell):
                    if snap_from(pos, cell):
                        progressed = True
                    else:
                        # 该内圈格已满足锚点却吸不上，说明暂存在此的是高仿诱饵或错配块：立刻移走并腾出该格
                        parked_photo.pop(cell, None)
                        tx, ty = cell_centre(cell)
                        drag(page, *css_xy(rect, tx, ty), *css_xy(rect, trash_pt[0], trash_pt[1]), delay=args.slow)
                        time.sleep(0.015)

    if shots:
        cv2.imwrite(str(shots / f"01_start_{mode}.png"), canvas)

    # ---------- 阶段 A：逐块抓到黑底检验台 -> 识别 -> 即刻分流 ----------
    max_pulls = (rows * cols + n_decoys) + 14
    side_slot_idx = 0
    ov_park_idx = 0

    for pull_i in range(max_pulls):
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if cur >= tot:
            break

        canvas = to_bgr(page.evaluate(JS_PNG))
        cands = candidate_points(canvas, pile, bg, forbid=failed_grabs)
        if not cands:
            print("  堆叠区已搬空")
            break

        gx, gy = cands[0]
        sub_before = canvas[iy0:iy1, ix0:ix1].astype(np.int16)
        drag(page, *css_xy(rect, gx, gy), *css_xy(rect, inspect_pt[0], inspect_pt[1]), delay=args.slow)
        time.sleep(0.02)
        canvas = to_bgr(page.evaluate(JS_PNG))
        sub_after = canvas[iy0:iy1, ix0:ix1].astype(np.int16)
        if float(np.abs(sub_after - sub_before).mean()) < 2.0:
            failed_grabs.append((gx, gy, 18))
            continue

        # 若抓取点在拼图块极边缘导致拖到检验台后贴近上下边界 (y<14 或 y>iy1-14)，做一次居中微调确保四边凸齿完整暴露
        fg_chk = fg_mask(canvas[iy0:iy1, ix0:ix1], thr=18)
        if int(fg_chk.sum()) >= pw * ph * 0.16:
            ys_c, xs_c = np.where(fg_chk > 0)
            if (int(ys_c.min()) < 14 or iy0 + int(ys_c.max()) > iy1 - 14
                    or int(xs_c.min()) < 14 or ix0 + int(xs_c.max()) > ix1 - 14):
                cx_q = ix0 + 0.5 * (float(xs_c.min()) + float(xs_c.max()))
                cy_q = iy0 + 0.5 * (float(ys_c.min()) + float(ys_c.max()))
                drag(page, *css_xy(rect, cx_q, cy_q), *css_xy(rect, inspect_pt[0], inspect_pt[1]), delay=args.slow)
                time.sleep(0.02)
                canvas = to_bgr(page.evaluate(JS_PNG))

        info = inspect_isolated_piece(
            canvas, ix0, iy0, ix1, iy1, pw, ph, tmpl_of,
            has_white_cells=has_white_cells, pure_white_mode=pure_white_mode,
            piece_clf=piece_clf, cell_class=cell_class,
            exclude=(placed | set(parked_photo.keys())) if mode == "overlap" else placed,
            decoy_thr=(0.74 if mode == "overlap" else 0.45),
        )
        kind = info["kind"]
        if kind == "empty":
            failed_grabs.append((gx, gy, 18))
            continue

        cx, cy = info["cx"], info["cy"]
        if kind == "white":
            edges = info["edges"]
            is_dec_w = bool(info.get("is_decoy"))
            if not white_cells or (mode == "overlap" and is_dec_w):
                print(f"  [白块诱饵] {edges} -> 丢弃废件区")
                drag(page, *css_xy(rect, cx, cy), *css_xy(rect, trash_pt[0], trash_pt[1]), delay=args.slow)
                time.sleep(0.015)
                continue

            # overlap 模式下棋盘是空的：角块/边块及已满足锚点的匹配白格可当场尝试直接吸附！
            if mode == "overlap":
                rem_w = [c for c in white_cells if c not in placed]
                cands_w = [
                    c for c in rem_w
                    if anchor_ok(c) and border_ok(edges, c) and white_compat(edges, c)[0] == 0
                ]
                cands_w.sort(key=lambda c: (-white_compat(edges, c)[1],
                                            -((1 if c[0] in (0, rows - 1) else 0) + (1 if c[1] in (0, cols - 1) else 0))))
                cur_pos = (cx, cy)
                snapped_w = False
                # 若候选格不超过 3 个（如角块唯一、边块 1~3 格），直接顺次试吸
                if 1 <= len(cands_w) <= 3:
                    for wc in cands_w:
                        if snap_from(cur_pos, wc):
                            placed_white_edges[wc] = edges
                            snapped_w = True
                            flush_anchored_parked()
                            break
                        else:
                            cur_pos = cell_centre(wc)
                if snapped_w:
                    continue
                wp_pt = overlap_park_slots[ov_park_idx % len(overlap_park_slots)]
                ov_park_idx += 1
                drag(page, *css_xy(rect, cur_pos[0], cur_pos[1]), *css_xy(rect, wp_pt[0], wp_pt[1]), delay=args.slow)
                parked_white.append({"pos": wp_pt, "edges": edges, "is_decoy": is_dec_w})
                failed_grabs.append((wp_pt[0], wp_pt[1], max(pw, ph) * 0.55))
                print(f"  [白块] 齿型 {edges} -> 暂存 ({wp_pt[0]:.0f},{wp_pt[1]:.0f})")
            else:
                all_park = side_slots + overlap_park_slots
                slot_pt = all_park[min(side_slot_idx, len(all_park) - 1)]
                side_slot_idx += 1
                drag(page, *css_xy(rect, cx, cy), *css_xy(rect, slot_pt[0], slot_pt[1]), delay=args.slow)
                parked_white.append({"pos": slot_pt, "edges": edges, "is_decoy": is_dec_w})
                failed_grabs.append((slot_pt[0], slot_pt[1], max(pw, ph) * 0.55))
                tag_d = " [疑似诱饵置后]" if is_dec_w else ""
                print(f"  [白块] 齿型 {edges}{tag_d} -> 边带暂存 ({slot_pt[0]:.0f},{slot_pt[1]:.0f})")
            time.sleep(0.015)
            continue

        if kind == "decoy":
            print(f"  [照片诱饵] raw={info.get('raw_ncc', 0):.2f} -> 丢弃废件区")
            drag(page, *css_xy(rect, cx, cy), *css_xy(rect, trash_pt[0], trash_pt[1]), delay=args.slow)
            time.sleep(0.015)
            continue

        # kind == 'photo'（raw_ncc >= 0.76，必为真照片块）
        cell = info["cell"]
        print(f"  ✔ 认出 {cell} ({info.get('pname')}) raw={info['raw_ncc']:.2f} score={info['score']:.2f}")
        if mode == "overlap":
            if anchor_ok(cell):
                if snap_from((cx, cy), cell):
                    flush_anchored_parked()
                else:
                    cur_pos = cell_centre(cell)
                    # 检查是否有次优高置信候选格
                    runners = sorted(
                        [(c2, sc2) for c2, sc2 in info.get("all_scores", {}).items()
                         if c2 != cell and c2 not in placed and c2 not in parked_photo and sc2[1] >= 0.76],
                        key=lambda kv: -kv[1][0]
                    )
                    handled = False
                    for c2, _ in runners[:2]:
                        if anchor_ok(c2):
                            if snap_from(cur_pos, c2):
                                flush_anchored_parked()
                                handled = True
                                break
                            else:
                                cur_pos = cell_centre(c2)
                        else:
                            tx2, ty2 = cell_centre(c2)
                            drag(page, *css_xy(rect, cur_pos[0], cur_pos[1]), *css_xy(rect, tx2, ty2), delay=args.slow)
                            parked_photo[c2] = (tx2, ty2)
                            handled = True
                            break
                    if not handled:
                        drag(page, *css_xy(rect, cur_pos[0], cur_pos[1]), *css_xy(rect, trash_pt[0], trash_pt[1]), delay=args.slow)
            else:
                tx, ty = cell_centre(cell)
                drag(page, *css_xy(rect, cx, cy), *css_xy(rect, tx, ty), delay=args.slow)
                parked_photo[cell] = (tx, ty)
                print(f"    暂存内圈格 {cell} 等待邻居锚点")
        else:
            all_park = side_slots + overlap_park_slots
            slot_pt = all_park[min(side_slot_idx, len(all_park) - 1)]
            side_slot_idx += 1
            drag(page, *css_xy(rect, cx, cy), *css_xy(rect, slot_pt[0], slot_pt[1]), delay=args.slow)
            staged_photo_list.append({"pos": slot_pt, "scores": info["all_scores"]})
            failed_grabs.append((slot_pt[0], slot_pt[1], max(pw, ph) * 0.55))
        time.sleep(0.015)

    # ---------- 阶段 B：匈牙利全局指派 + 按锚点顺序把暂存照片块全部归位 ----------
    if staged_photo_list and remaining_photo:
        from scipy.optimize import linear_sum_assignment
        rem_cells = sorted(remaining_photo - set(parked_photo.keys()))
        if rem_cells:
            cost = np.full((len(staged_photo_list), len(rem_cells)), 2.0, dtype=np.float32)
            for pi, item in enumerate(staged_photo_list):
                for ci, c_key in enumerate(rem_cells):
                    sc_entry = item["scores"].get(c_key)
                    if sc_entry is not None:
                        cost[pi, ci] = -float(sc_entry[0])
            r_ind, c_ind = linear_sum_assignment(cost)
            assigned_pis = set()
            for pi, ci in zip(r_ind, c_ind):
                parked_photo[rem_cells[ci]] = staged_photo_list[pi]["pos"]
                assigned_pis.add(pi)
            spare_photos = [staged_photo_list[pi] for pi in range(len(staged_photo_list)) if pi not in assigned_pis]
        else:
            spare_photos = list(staged_photo_list)
    else:
        spare_photos = []

    if parked_photo:
        assign_plan = {k: {"row": cell[0], "col": cell[1]}
                       for k, cell in enumerate(parked_photo.keys())}
        idx_to_cell = {k: cell for k, cell in enumerate(parked_photo.keys())}
        for k in anchor_order(assign_plan):
            cell = idx_to_cell[k]
            if cell in placed or cell not in parked_photo:
                continue
            pos = parked_photo[cell]
            if not snap_from(pos, cell):
                tx, ty = cell_centre(cell)
                drag(page, *css_xy(rect, tx, ty), *css_xy(rect, pos[0], pos[1]), delay=0)
                time.sleep(0.02)
                for sp in list(spare_photos):
                    if snap_from(sp["pos"], cell):
                        spare_photos.remove(sp)
                        break
                    else:
                        drag(page, *css_xy(rect, tx, ty), *css_xy(rect, sp["pos"][0], sp["pos"][1]), delay=0)
                        time.sleep(0.02)

    # ---------- 阶段 C：白块按直边 + 四邻凹凸互补动态贪心归位 ----------
    used_w: set[int] = set()
    cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
    if cur < tot and white_cells and parked_white:
        rem_wcells = [c for c in white_cells if c not in placed]
        req_edges = {}
        if placed:
            cv_board = to_bgr(page.evaluate(JS_PNG))
            bg_col = board_color(cv_board, L)
            for cell in rem_wcells:
                req_edges[cell] = cell_required_edges(cv_board, L, cell[0], cell[1], bg_col)

        def compat_rank_w(wi, cell):
            e = parked_white[wi]["edges"]
            b_bad = 0 if border_ok(e, cell) else 1
            r, c = cell
            mm, mt = 0, 0
            for dr, dc, side, opp in NEIGH:
                nb = (r + dr, c + dc)
                if nb in placed_white_edges:
                    ne = placed_white_edges[nb][opp]
                    if e[side] != 0 and ne != 0:
                        if e[side] + ne == 0:
                            mt += 2
                        else:
                            mm += 2
                elif nb in placed and cell in req_edges:
                    re_val = req_edges[cell][side]
                    if re_val != 0 and e[side] != 0:
                        if e[side] == re_val:
                            mt += 1
                        else:
                            mm += 1
            n_border = (1 if r in (0, rows - 1) else 0) + (1 if c in (0, cols - 1) else 0)
            dec_pen = 1 if parked_white[wi].get("is_decoy") else 0
            return (b_bad, mm, dec_pen, -mt, -n_border, wi)

        for _round in range(2):
            if cur >= tot or not rem_wcells:
                break
            tried_w: set[tuple[int, tuple[int, int]]] = set()
            while cur < tot and rem_wcells:
                anchored_w = [c for c in rem_wcells if anchor_ok(c)] or list(rem_wcells)
                avail_w = [wi for wi in range(len(parked_white)) if wi not in used_w]
                cands = [
                    (compat_rank_w(wi, c), wi, c)
                    for c in anchored_w
                    for wi in avail_w
                    if (wi, c) not in tried_w
                ]
                if not cands:
                    break
                cands.sort(key=lambda t: t[0])
                _, best_wi, best_c = cands[0]
                tried_w.add((best_wi, best_c))
                pos = parked_white[best_wi]["pos"]
                if snap_from(pos, best_c):
                    used_w.add(best_wi)
                    rem_wcells.remove(best_c)
                    placed_white_edges[best_c] = parked_white[best_wi]["edges"]
                    flush_anchored_parked()
                else:
                    tx, ty = cell_centre(best_c)
                    drag(page, *css_xy(rect, tx, ty), *css_xy(rect, pos[0], pos[1]), delay=0)
                    time.sleep(0.015)

    # ---------- 阶段 D：兜底补漏（若仍有未归位格与未用暂存块） ----------
    cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
    if cur < tot:
        flush_anchored_parked()
        rem_all = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in placed]
        leftover_pos = (
            [parked_white[wi]["pos"] for wi in range(len(parked_white)) if wi not in used_w]
            + [sp["pos"] for sp in spare_photos]
        )
        for cell in list(rem_all):
            if cur >= tot:
                break
            for p_pos in list(leftover_pos):
                if snap_from(p_pos, cell):
                    leftover_pos.remove(p_pos)
                    break
                else:
                    tx, ty = cell_centre(cell)
                    drag(page, *css_xy(rect, tx, ty), *css_xy(rect, p_pos[0], p_pos[1]), delay=0)
                    time.sleep(0.015)

    cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
    if shots:
        cv2.imwrite(str(shots / f"02_final_{mode}.png"), to_bgr(page.evaluate(JS_PNG)))
        page.screenshot(path=str(shots / f"03_page_{mode}.png"))

    elapsed = round(time.time() - t_level, 1)
    print(f"\n=== {mode} Part {part_id} · {diff}格 结果 {cur}/{tot} | 用时 {elapsed}s "
          f"| {'🎉 通关' if cur == tot else '未完成'} ===")
    return cur, tot, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overlap", "center"], default="overlap")
    ap.add_argument("--part", type=int, default=None,
                    help="指定部分 (1=照片, 2=纯白, 3=混合)；不传则默认跑全部 Part 1/2/3")
    ap.add_argument("--theme", default=None, help="兼容旧参数 animals/white/mixed")
    ap.add_argument("--diff", type=int, default=None,
                    help="指定难度 (9/16/25)；不传则默认跑全部 9 -> 16 -> 25")
    ap.add_argument("--headful", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--maximize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dpr", type=float, default=None)
    ap.add_argument("--vw", type=int, default=1280)
    ap.add_argument("--vh", type=int, default=800)
    ap.add_argument("--start-timeout", type=float, default=60.0)
    ap.add_argument("--slow", type=float, default=0.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--ident-min", type=float, default=0.52)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--auto-close", action="store_true",
                    help="通关后展示 1.5s 自动关闭窗口")
    args = ap.parse_args()

    theme_to_part = {"animals": 1, "white": 2, "mixed": 3}
    if args.part is None and args.theme in theme_to_part:
        args.part = theme_to_part[args.theme]
    if args.diff == 4:
        print("[提示] 新版考核站已移除 4 格 (2×2) 选项，自动切换为 9 格。")
        args.diff = 9

    parts = [args.part] if args.part is not None else [1, 2, 3]
    diffs = [args.diff] if args.diff is not None else [9, 16, 25]
    schedule = [(pt, df) for pt in parts for df in diffs]
    shots_root = Path(args.shots) if args.shots else None

    t_start = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headful,
                                    args=["--no-sandbox", "--disable-dev-shm-usage"])

        if args.maximize and args.headful:
            aw, ah, real_dpr = screen_metrics(browser)
            vw_css, vh_css = aw, max(600, ah - 8)
            scale = args.dpr or real_dpr
            print(f"[窗口] 屏幕 {aw}x{ah} · DPR {scale:g} -> 视口 {vw_css}x{vh_css}")
        else:
            vw_css, vh_css = args.vw, args.vh
            scale = args.dpr or 1.0
        viewport = {"width": vw_css, "height": vh_css}

        if args.video:
            Path(args.video).mkdir(parents=True, exist_ok=True)
            ctx = browser.new_context(viewport=viewport, device_scale_factor=scale,
                                      record_video_dir=args.video,
                                      record_video_size={"width": args.vw, "height": args.vh})
            page = ctx.new_page()
        else:
            ctx = None
            page = browser.new_page(viewport=viewport, device_scale_factor=scale)

        if args.maximize and args.headful:
            maximize_window(page)

        page.on("dialog", lambda d: d.accept())

        from run_exam import install_catalog_cache
        install_catalog_cache(page.context)

        # 全部关卡开始前只加载并预热一次 YOLO
        piece_clf = cell_clf = None
        if not args.no_yolo:
            try:
                from yolo_check import Classifier, DET_WEIGHTS, CLS_WEIGHTS
                if CLS_WEIGHTS.exists():
                    piece_clf = Classifier(weights=CLS_WEIGHTS)
                if DET_WEIGHTS.exists():
                    cell_clf = Classifier(weights=DET_WEIGHTS)
                if piece_clf is None:
                    piece_clf = cell_clf
                if cell_clf is None:
                    cell_clf = piece_clf
                dummy = np.full((240, 320, 3), 128, np.uint8)
                for clf in {id(c): c for c in (piece_clf, cell_clf) if c}.values():
                    clf.predict(dummy)
            except Exception as e:
                print(f"[warn] YOLO 未启用：{e}")

        summary = []
        part_names = {1: "Part 1 · 照片", 2: "Part 2 · 纯白", 3: "Part 3 · 混合"}
        for idx, (pt, df) in enumerate(schedule):
            sub_shots = (
                (shots_root / f"p{pt}_{df}" if len(schedule) > 1 else shots_root)
                if shots_root else None
            )
            cur, tot, dt = solve_one_bonus_level(
                page, pt, df, args.mode, piece_clf, cell_clf, args, shots=sub_shots
            )
            summary.append((pt, df, cur, tot, dt))
            if idx < len(schedule) - 1:
                page.wait_for_timeout(800)

        if len(summary) > 1:
            total_dt = round(time.time() - t_start, 1)
            passed = sum(1 for _, _, c, t, _ in summary if c == t and t > 0)
            print("\n" + "=" * 64)
            print(f"附加题 ({args.mode}) 全部关卡汇总：{passed}/{len(summary)} 关通关 | 总用时 {total_dt}s")
            print("-" * 64)
            for pt, df, c, t, dt in summary:
                r_sz = int(round(df ** 0.5))
                ok_s = "✅ 通关" if (c == t and t > 0) else "❌ 未完成"
                print(f"  {part_names.get(pt, f'Part {pt}'):<14} | {r_sz}×{r_sz} ({df:>2}格) | "
                      f"{c:>2}/{t:<2} | {dt:>5.1f}s | {ok_s}")
            print("=" * 64)

        if args.headful:
            if args.auto_close:
                page.wait_for_timeout(1500)
            elif sys.stdin and sys.stdin.isatty():
                try:
                    input("按回车关闭…")
                except (EOFError, KeyboardInterrupt):
                    pass
        if ctx is not None:
            page.close()
            ctx.close()
        browser.close()
        all_ok = all(c == t and t > 0 for _, _, c, t, _ in summary)
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""puzzle.moyunliuyin.me 考核自动求解器（Part 3 混合 · 5x5）。

分工
----
* OpenCV：参考图切格、按槽位裁块、模板匹配定位、纯白块识别
* YOLO   ：照片格的动物类别识别（辅助校验，见 yolo_classify.py）
* 鼠标   ：CDP 真实鼠标事件，先按下再分步移动再抬起

关键约束（读源码得来，都实测过）
--------------------------------
1. 页面不暴露 window.app（只 localhost 有），拿不到答案，只能看像素
2. 吸附阈值 max(pieceWidth*0.14, 10)，比旧站紧一倍，必须瞄到格心
3. 必须有锚点：块要么在棋盘外圈，要么紧邻一块已放好的块 —— 所以要从外向内放
4. 只有真实指针拖拽（mousedown 时选中）才允许吸附
5. 托盘块数 = 格数 + 20% 诱饵块，诱饵放不进任何格子，必须识别后丢弃

流程
----
1. 复现布局几何（board 矩形 / 格子尺寸 / 30 个槽位）
2. 参考图按 rows×cols 切格，分出「照片格」和「纯白格」
3. 每个槽位裁块，模板匹配照片格 -> 20 个高置信映射
4. 剩余的低分块 = 纯白块 + 诱饵块
5. 按锚点顺序（先外圈、再逐层向内）拖拽落子
6. 剩下的纯白块用「排除法 + 吸附反馈」补完
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 布局几何

def compute_layout(cw, ch, rows, cols, n_decoys=0):
    """复现 setupBoardLayout / applyBoardGeometry / traySlots。"""
    padding = 30
    max_board_w = min(cw * 0.65, 750)
    max_board_h = min(ch - padding * 2, 550)
    bw, bh = max_board_w, max_board_w * 0.75
    if bh > max_board_h:
        bh = max_board_h
        bw = bh / 0.75
    wanted = rows * cols + n_decoys

    def geo(scale):
        w, h = bw * scale, bh * scale
        return w, h, max(padding, (cw - w) / 2), (ch - h) / 2

    def slots_for(w, h, board_x):
        pw, ph = w / cols, h / rows
        step_x, step_y = pw * 1.16, ph * 1.16
        cols_per_side = int(max(0.0, board_x) / step_x)
        n_rows = int(ch / step_y)
        out = []
        for side in range(2):
            origin_x = 0.0 if side == 0 else board_x + w
            for col in range(cols_per_side):
                for row in range(n_rows):
                    x = origin_x + col * step_x + (step_x - pw) / 2
                    y = row * step_y + (step_y - ph) / 2
                    if x < 0 or y < 0 or x + pw > cw or y + ph > ch:
                        continue
                    out.append({"x": x, "y": y, "side": side})
        return out

    scale, chosen = 1.0, None
    while scale >= 0.3:
        w_, h_, bx_, by_ = geo(scale)
        if len(slots_for(w_, h_, bx_)) >= wanted:
            chosen = scale
            break
        scale -= 0.04
    final = scale if chosen is None else chosen
    w, h, bx, by = geo(final)
    pw, ph = w / cols, h / rows
    return {"board": {"x": bx, "y": by, "w": w, "h": h},
            "piece": {"w": pw, "h": ph},
            "slots": slots_for(w, h, bx), "rows": rows, "cols": cols}


# ---------------------------------------------------------------- 图像工具

def fg_mask(img, thr=28):
    return (img.astype(np.int32).sum(axis=2) > thr).astype(np.uint8)


def piece_stats(img):
    m = fg_mask(img).astype(bool)
    if m.sum() < 20:
        return {"sat": 0.0, "std": 0.0, "mean": 0.0}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"sat": float(hsv[..., 1][m].mean()),
            "std": float(gray[m].std()),
            "mean": float(gray[m].mean())}


def center_stats(canvas, slot, pw, ph, frac=0.30):
    """块中心区统计（避开边缘：邻块凸齿会侵进来）。"""
    x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
    cx0, cx1 = x0 + int(pw * frac), x0 + int(pw * (1 - frac))
    cy0, cy1 = y0 + int(ph * frac), y0 + int(ph * (1 - frac))
    crop = canvas[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return {"sat": 999.0, "std": 999.0, "mean": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {"sat": float(hsv[..., 1].mean()), "std": float(g.std()),
            "mean": float(g.mean())}


def is_white_piece(canvas, slot, pw, ph):
    """纯白块判定：块中心区是纯 255。

    实测分界非常干净：真白块中心 sat=0 std=0 mean=255.0，
    次低的照片块 sat=4.8。所以用 sat<3 / std<3 / mean>250 卡死，不会误判。
    """
    st = center_stats(canvas, slot, pw, ph)
    white = st["sat"] < 3.0 and st["std"] < 3.0 and st["mean"] > 250.0
    return white, st


def is_white_cell(img):
    """参考图里的纯白格是真正的 255 纯白（std=0, sat=0）。

    实测：真白格 mean=255.0 std=0.0 sat=0.0；
          白底照片 mean≈230 std≈22。所以阈值必须卡死。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (float(hsv[..., 1].mean()) < 3.0
            and float(gray.mean()) > 250.0
            and float(gray.std()) < 3.0)


def split_reference(ref, rows, cols):
    h, w = ref.shape[:2]
    cells = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            cells.append({"row": r, "col": c, "img": ref[y0:y1, x0:x1]})
    return cells


def template_score(pad, cell_img, pw, ph):
    """在 pad（槽位外扩区）里搜 cell 模板，返回 (NCC, dx, dy)。dx/dy 相对 pad 左上角。"""
    tmpl = cv2.resize(cell_img, (int(round(pw)), int(round(ph))),
                      interpolation=cv2.INTER_AREA)
    if pad.shape[0] < tmpl.shape[0] or pad.shape[1] < tmpl.shape[1]:
        return -1.0, 0, 0
    res = cv2.matchTemplate(pad, tmpl, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1.0)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), int(loc[0]), int(loc[1])


# ---------------------------------------------------------------- 视觉分析

PAD = 8


class Analysis:
    def __init__(self, canvas, ref, rows, cols, n_decoys):
        self.canvas = canvas
        self.ref = ref
        self.rows, self.cols = rows, cols
        self.n_decoys = n_decoys
        ch, cw = canvas.shape[:2]
        self.L = compute_layout(cw, ch, rows, cols, n_decoys)
        self.pw = self.L["piece"]["w"]
        self.ph = self.L["piece"]["h"]
        self.cw, self.ch = cw, ch
        self.cells = split_reference(ref, rows, cols)
        self.white_cells = [c for c in self.cells if is_white_cell(c["img"])]
        self.photo_cells = [c for c in self.cells if not is_white_cell(c["img"])]

    def crop_pad(self, slot):
        x0 = max(0, int(round(slot["x"])) - PAD)
        y0 = max(0, int(round(slot["y"])) - PAD)
        x1 = min(self.cw, int(round(slot["x"] + self.pw)) + PAD)
        y1 = min(self.ch, int(round(slot["y"] + self.ph)) + PAD)
        return self.canvas[y0:y1, x0:x1], (x0, y0)

    def match_photo_pieces(self, classifier=None, class_bonus=0.20):
        """照片块 vs 照片格 -> 全局最优指派。

        托盘有 30 块（25 真 + 5 诱饵），照片格有 20 个。
        对 25 个照片块 × 20 个照片格做匈牙利指派，
        落选的 5 个照片块就是诱饵。
        """
        from scipy.optimize import linear_sum_assignment

        photo_slots, white_slots = [], []
        stats_all, pads = {}, {}
        for i, slot in enumerate(self.L["slots"]):
            pad, origin = self.crop_pad(slot)
            # 空槽位（块数 < 槽位数时会出现）：格子区几乎全黑，直接跳过
            inner = self.canvas[int(round(slot["y"])):int(round(slot["y"] + self.ph)),
                                int(round(slot["x"])):int(round(slot["x"] + self.pw))]
            if inner.size == 0 or fg_mask(inner).mean() < 0.5:
                continue
            white, st = is_white_piece(self.canvas, slot, self.pw, self.ph)
            stats_all[i] = st
            pads[i] = (pad, origin)
            (white_slots if white else photo_slots).append(i)

        cells = self.photo_cells
        score = np.full((len(photo_slots), len(cells)), -2.0, dtype=np.float32)
        offs = np.zeros((len(photo_slots), len(cells), 2), dtype=np.int32)
        for a, i in enumerate(photo_slots):
            pad, _ = pads[i]
            for b, cell in enumerate(cells):
                sc, dx, dy = template_score(pad, cell["img"], self.pw, self.ph)
                score[a, b] = sc
                offs[a, b] = (dx, dy)

        # YOLO 类别一致性：块与格同属一类时加分。
        # 独立的语义约束，专门解决"两张照片内容相似"的混淆。
        self.class_hits = 0
        if classifier is not None:
            cell_cls = [classifier.predict(c["img"], key=(c["row"], c["col"]))[0]
                        for c in cells]
            for a, i in enumerate(photo_slots):
                slot = self.L["slots"][i]
                crop = self.canvas[int(round(slot["y"])):int(round(slot["y"] + self.ph)),
                                   int(round(slot["x"])):int(round(slot["x"] + self.pw))]
                name = classifier.predict(crop, key=("piece", i))[0]
                for b, cell_name in enumerate(cell_cls):
                    if b < score.shape[1] and cell_name == name and score[a, b] > 0:
                        score[a, b] += class_bonus
                        self.class_hits += 1

        # 匈牙利求最小代价 -> 用 -score
        rows, cols = linear_sum_assignment(-score)
        assign = {}
        for a, b in zip(rows, cols):
            i = photo_slots[a]
            cell = cells[b]
            dx, dy = offs[a, b]
            assign[i] = {"row": cell["row"], "col": cell["col"],
                         "score": float(score[a, b]),
                         "jitter": (int(dx) - PAD, int(dy) - PAD)}
        decoys = [i for i in photo_slots if i not in assign]
        return {"assign": assign, "white_slots": white_slots, "decoys": decoys,
                "stats": stats_all, "pads": pads,
                "white_cells": [(c["row"], c["col"]) for c in self.white_cells]}

    def analyse(self, conf_threshold=0.55):
        """返回 plan：{slot_index: {"row","col"}}，以及未分配的槽位。"""
        assign = {}
        white_slots, unknown = [], []
        info = {}

        for i, slot in enumerate(self.L["slots"]):
            pad, (px, py) = self.crop_pad(slot)
            inner = self.canvas[int(round(slot["y"])):int(round(slot["y"] + self.ph)),
                                int(round(slot["x"])):int(round(slot["x"] + self.pw))]
            white, st = is_white_piece(inner)

            if white:
                white_slots.append(i)
                info[i] = {"white": True, "stats": st}
                continue

            best = (-1.0, None, 0, 0)
            for cell in self.photo_cells:
                sc, dx, dy = template_score(pad, cell["img"], self.pw, self.ph)
                if sc > best[0]:
                    best = (sc, cell, dx, dy)
            sc, cell, dx, dy = best
            info[i] = {"white": False, "best": sc, "stats": st,
                       "cell": (cell["row"], cell["col"]) if cell else None,
                       "jitter": (dx - PAD, dy - PAD),
                       "pad_origin": (px, py)}
            if sc >= conf_threshold:
                assign[i] = {"row": cell["row"], "col": cell["col"],
                             "score": sc, "jitter": (dx - PAD, dy - PAD),
                             "pad_origin": (px, py)}
            else:
                unknown.append(i)

        # 去重：同一格只保留分数最高的块
        by_cell = {}
        for i, a in assign.items():
            key = (a["row"], a["col"])
            if key not in by_cell or a["score"] > assign[by_cell[key]]["score"]:
                by_cell[key] = i
        assign = {i: assign[i] for i in by_cell.values()}
        unknown = [i for i in range(len(self.L["slots"])) if i not in assign and i not in white_slots]

        return {"assign": assign, "white_slots": white_slots,
                "unknown": unknown, "info": info,
                "white_cells": [(c["row"], c["col"]) for c in self.white_cells]}


# ---------------------------------------------------------------- 任意块重定位

def locate_piece(canvas, cell_img, pw, ph, L, min_score=0.25):
    """在整张画布上按内容模板搜索一块，返回其中心坐标（找不到返回 None）。

    用于落子失败后的重试：块可能被上一轮的拖动挪走，或初始抖动估计不准。
    """
    tmpl = cv2.resize(cell_img, (int(round(pw)), int(round(ph))),
                      interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(canvas, tmpl, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1.0)
    # 排除棋盘区域（已放好的块精确对齐格心，不参与）
    bx, by = L["board"]["x"], L["board"]["y"]
    bw, bh = L["board"]["w"], L["board"]["h"]
    m = 3
    x0 = max(0, int(bx) + m); x1 = min(canvas.shape[1], int(bx + bw) - m)
    y0 = max(0, int(by) + m); y1 = min(canvas.shape[0], int(by + bh) - m)
    if x1 > x0 and y1 > y0:
        res[y0:y1, x0:x1] = -1.0
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < min_score:
        return None
    return {"cx": float(loc[0] + pw / 2), "cy": float(loc[1] + ph / 2), "score": float(mx)}


# ---------------------------------------------------------------- 白块定位

def locate_piece_white(canvas, L, pw, ph, slot):
    """找指定槽位那块白块现在的位置（按与槽位中心的距离挑最近的）。"""
    blobs = find_white_pieces(canvas, L, pw, ph, skip_aligned=False)
    if not blobs:
        return None
    cx = slot["x"] + pw / 2
    cy = slot["y"] + ph / 2
    best = min(blobs, key=lambda b: (b["cx"] - cx) ** 2 + (b["cy"] - cy) ** 2)
    d = ((best["cx"] - cx) ** 2 + (best["cy"] - cy) ** 2) ** 0.5
    if d > max(pw, ph) * 2.5:      # 离太远说明不是这块
        return None
    return best


def find_white_pieces(canvas, L, pw, ph, skip_aligned=True):
    """在画布上扫描「纯白」连通域，返回每块的中心坐标。

    纯白块在白底之外没有任何内容，是最容易定位的一类：
    三通道都接近 255 的像素即为块体。棋盘区域（深蓝）和背景（黑）都不满足。
    棋盘上已放好的白块会被排除（只在棋盘矩形之外找）。
    """
    b, g, r = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    white = ((b > 248) & (g > 248) & (r > 248)).astype(np.uint8)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(white, connectivity=8)
    area_min = pw * ph * 0.25
    area_max = pw * ph * 2.5

    # 已放好的块精确对齐格心；未放好的块会有 >= 吸附阈值的偏移。
    # 因此用「是否对齐格心」区分已放/未放，而不是粗暴排除棋盘区
    # （否则拖到棋盘上但没吸附的块会丢失，再也找不到）。
    centers = []
    bx, by = L["board"]["x"], L["board"]["y"]
    for r in range(L["rows"]):
        for c in range(L["cols"]):
            centers.append((bx + c * pw + pw / 2, by + r * ph + ph / 2))

    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_min <= area <= area_max):
            continue
        # 尺寸约束：白块的外接框 ≈ 格子 + 凸齿，最多 1.45 倍。
        # 照片里的大片白色区域通常尺寸不符，靠这一条过滤掉。
        bw_ = stats[i, cv2.CC_STAT_WIDTH]
        bh_ = stats[i, cv2.CC_STAT_HEIGHT]
        if not (pw * 0.90 <= bw_ <= pw * 1.30) or not (ph * 0.90 <= bh_ <= ph * 1.30):
            continue
        cx, cy = cents[i]
        if skip_aligned:
            aligned = any(abs(cx - ex) < 4 and abs(cy - ey) < 4 for ex, ey in centers)
            if aligned:
                continue                  # 已放好（精确对齐格心）
        out.append({"cx": float(cx), "cy": float(cy), "area": int(area),
                    "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                             int(bw_), int(bh_))})
    out.sort(key=lambda d: (d["cy"], d["cx"]))
    return out


# ---------------------------------------------------------------- 落子顺序

def anchor_order(assign):
    """按锚点规则排序：外圈先放，再逐层向内。只有邻居已放或在外圈的块才能放。"""
    placed = set()
    remaining = dict(assign)
    order = []
    rows = max(a["row"] for a in assign.values()) + 1
    cols = max(a["col"] for a in assign.values()) + 1

    def on_border(cell):
        r, c = cell
        return r in (0, rows - 1) or c in (0, cols - 1)

    def has_anchor(cell):
        r, c = cell
        if on_border(cell):
            return True
        return any((r + dr, c + dc) in placed
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    while remaining:
        progressed = False
        for i in sorted(remaining, key=lambda k: (remaining[k]["row"], remaining[k]["col"])):
            cell = (remaining[i]["row"], remaining[i]["col"])
            if has_anchor(cell):
                order.append(i)
                placed.add(cell)
                del remaining[i]
                progressed = True
                break
        if not progressed:
            break
    return order


# ---------------------------------------------------------------- 浏览器侧

READ_STATE_JS = "() => document.getElementById('progress-display').innerText"
CANVAS_RECT_JS = """() => { const r = document.getElementById('puzzle-canvas').getBoundingClientRect();
  return JSON.stringify({left:r.left, top:r.top, w:r.width, h:r.height}); }"""
CANVAS_PNG_JS = "() => document.getElementById('puzzle-canvas').toDataURL('image/png')"
REF_PNG_JS = """() => { const im = document.getElementById('preview-img');
  return JSON.stringify({src: im.src, w: im.naturalWidth, h: im.naturalHeight}); }"""


def drag(page, sx, sy, tx, ty, steps=12):
    """真实鼠标拖拽：按下 -> 分步移动 -> 抬起。"""
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(tx, ty, steps=steps)
    page.mouse.up()


def place_piece(page, rect, L, slot, cell, jitter=(0, 0), verbose=True):
    """把 slot 位置的块拖到 (row, col)。"""
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    start_x = rect["left"] + slot["x"] + jitter[0] + pw / 2
    start_y = rect["top"] + slot["y"] + jitter[1] + ph / 2
    target_x = rect["left"] + L["board"]["x"] + cell["col"] * pw + pw / 2
    target_y = rect["top"] + L["board"]["y"] + cell["row"] * ph + ph / 2
    if verbose:
        print(f"    拖 ({start_x:.0f},{start_y:.0f}) -> ({target_x:.0f},{target_y:.0f})")
    drag(page, start_x, start_y, target_x, target_y)
    time.sleep(0.12)

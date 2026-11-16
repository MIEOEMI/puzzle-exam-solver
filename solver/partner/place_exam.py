"""新考核站（puzzle.moyunliuyin.me）的浏览器侧接口 —— 对应旧站的 place.py。

只做浏览器操作，不含任何识别/求解逻辑：
    open_exam   开一局（Part 3 混合 · 5×5），校验块数
    export      导出画布 + 参考图 + 每块的位置
    place       按 plan（哪块 -> 哪格）把块拖过去，内部自动按锚点顺序落子
    state       读进度

与旧站 place.py 的区别
--------------------
| 项 | 旧站 | 新站 |
|---|---|---|
| 块数 | 4/9/16/25 | 25 + **5 个诱饵** = 30 |
| 吸附阈值 | 0.28×块宽 | **0.14×块宽**（更严，必须精准瞄格心） |
| 落子限制 | 无 | **锚点规则**：必须在外圈或紧邻已放块 |
| 答案暴露 | `window.app.pieces[i].correctX/Y` | **不暴露**，只能看像素 |
| 画布 | 1400×900 视口 | 919×598（棋盘 369.6×277.2） |

用法
----
    from place_exam import open_exam, export, place, state

    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=False)
        page = br.new_page(viewport={"width": 1280, "height": 800})

        info = open_exam(page)                 # 开一局 5×5 混合，返回 {n, rows, cols}
        assets = export(page, "assets")        # 画布/参考图/每块裁图
        plan = {0: {"row": 0, "col": 0},       # 块序号（槽位索引）-> 格子
                1: {"row": 0, "col": 1}}
        print(place(page, plan))               # 返回吸附成功块数
        print(state(page))
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from solver_core import compute_layout, drag, fg_mask

URL_EXAM = "https://puzzle.moyunliuyin.me/?mode=exam"

JS_RECT = """() => { const r = document.getElementById('puzzle-canvas').getBoundingClientRect();
  return JSON.stringify({left:r.left, top:r.top, w:r.width, h:r.height}); }"""
JS_PNG = "() => document.getElementById('puzzle-canvas').toDataURL('image/png')"
JS_REF = """() => { const im = document.getElementById('preview-img');
  return JSON.stringify({src: im.src, w: im.naturalWidth, h: im.naturalHeight}); }"""
JS_PROGRESS = "() => { const e = document.getElementById('progress-display'); return e ? e.innerText : ''; }"
JS_START = """([part, diff]) => {
  const p = document.querySelector('input[name="exam-part"][value="'+part+'"]');
  const d = document.querySelector('input[name="exam-difficulty"][value="'+diff+'"]');
  if (!p || !d) return 'no-radio';
  p.checked = true; p.dispatchEvent(new Event('change', {bubbles:true}));
  d.checked = true; d.dispatchEvent(new Event('change', {bubbles:true}));
  const btn = document.getElementById('btn-exam-start');
  if (!btn) return 'no-button';
  btn.click();
  return 'ok';
}"""

JS_SWITCH = """() => { const b = document.getElementById('btn-exam-switch');
  if (b) { b.click(); return 'switch'; } return 'none'; }"""


def to_bgr(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def parse_progress(txt):
    try:
        a, b = str(txt).replace(" ", "").split("/")
        return int(a), int(b)
    except Exception:
        return 0, 0


def state(page):
    """当前进度 {placed, total, won}。"""
    cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
    return {"placed": cur, "total": tot, "won": tot > 0 and cur >= tot}


def open_exam(page, diff=25, n_decoys=None, wait_s=3.0, retries=3):
    """打开考核页并开一局，确认棋盘已经换成本局的（不是上一局残留）。

    新站没有 window.app，无法直接读块数；用画布像素判断"托盘区是否有新块"：
    开局瞬间所有块都在槽位上，参考图也已加载。
    """
    rows = cols = int(round(diff ** 0.5))
    if n_decoys is None:
        n_decoys = round(diff * 0.2)
    cur = tot = 0
    for attempt in range(retries):
        page.goto(URL_EXAM, wait_until="load", timeout=60000)
        page.wait_for_timeout(1200)
        # 考核页默认是 Part 1 · 9 块，必须显式选 Part 3 + 5×5
        res = page.evaluate(JS_START, [3, diff])
        page.wait_for_timeout(int(wait_s * 1000))
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if res == "ok" and tot == diff and cur == 0:
            return {"n": diff + n_decoys, "rows": rows, "cols": cols,
                    "placed": cur, "total": tot}
        if res != "ok":                       # 没有单选框 -> 退回"切换部分"再选
            page.evaluate(JS_SWITCH)
            page.wait_for_timeout(800)
            page.evaluate(JS_START, [3, diff])
            page.wait_for_timeout(int(wait_s * 1000))
            cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
            if tot == diff and cur == 0:
                return {"n": diff + n_decoys, "rows": rows, "cols": cols,
                        "placed": cur, "total": tot}
        time.sleep(0.5)
    raise RuntimeError(f"开局失败：进度显示 {cur}/{tot}，期望 0/{diff}")


def export(page, outdir=None, rows=5, cols=5, n_decoys=5):
    """导出画布、参考图、布局几何与每块裁图。

    返回 dict：
        canvas      BGR ndarray
        ref         BGR ndarray
        layout      布局几何（棋盘/格子/槽位）
        pieces      [{index, x, y, w, h, white, img}]  槽位索引即"块序号"
        rect        画布在页面里的位置（拖拽用）
    """
    rect = json.loads(page.evaluate(JS_RECT))
    canvas = to_bgr(page.evaluate(JS_PNG))
    ref_info = json.loads(page.evaluate(JS_REF))
    ref = to_bgr(ref_info["src"]) if ref_info.get("src") else None

    L = compute_layout(canvas.shape[1], canvas.shape[0], rows, cols, n_decoys)
    pw, ph = L["piece"]["w"], L["piece"]["h"]

    pieces = []
    for i, slot in enumerate(L["slots"]):
        x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
        x1, y1 = int(round(x0 + pw)), int(round(y0 + ph))
        if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
            continue
        crop = canvas[y0:y1, x0:x1].copy()
        # 槽位比块多：棋盘外框能摆 32 个槽，实际只有 30 块，空槽位要跳过
        if fg_mask(crop).mean() < 0.5:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = crop.shape[:2]
        cc = hsv[int(h * .35):int(h * .65), int(w * .35):int(w * .65)]
        white = bool(cc[:, :, 1].mean() < 3 and cc[:, :, 2].mean() > 250)
        pieces.append({"index": i, "x": x0, "y": y0, "w": w, "h": h,
                       "white": white, "img": crop})

    out = {"canvas": canvas, "ref": ref, "layout": L, "pieces": pieces,
           "rect": rect, "rows": rows, "cols": cols}
    if outdir:
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "canvas.png"), canvas)
        if ref is not None:
            cv2.imwrite(str(d / "reference.png"), ref)
        for p in pieces:
            cv2.imwrite(str(d / f"piece_{p['index']:02d}.png"), p["img"])
        (d / "layout.json").write_text(json.dumps(
            {"board": L["board"], "piece": L["piece"], "rows": rows, "cols": cols,
             "pieces": [{"index": p["index"], "x": p["x"], "y": p["y"],
                         "white": p["white"]} for p in pieces]},
            ensure_ascii=False, indent=1))
    return out


def validate_plan(plan, pieces, rows, cols):
    """严格校验 plan：{块序号: {"row": r, "col": c}}。

    格式不对直接抛 PlanError，不做宽容转换 ——
    格式错和算法错必须能区分开，否则会把"格式写错"伪装成"算法不准"。
    """
    if not isinstance(plan, dict):
        raise PlanError(f"plan 必须是 dict，实际是 {type(plan).__name__}")
    valid_idx = {p["index"] for p in pieces}
    seen_cells = {}
    for k, v in plan.items():
        if not isinstance(k, int):
            raise PlanError(f"键必须是 int 块序号，实际是 {k!r}")
        if k not in valid_idx:
            raise PlanError(f"块序号 {k} 不在本局块列表里（有效范围 {min(valid_idx)}..{max(valid_idx)}）")
        if not isinstance(v, dict) or "row" not in v or "col" not in v:
            raise PlanError(f"块 {k} 的值必须是 {{'row': int, 'col': int}}，实际是 {v!r}")
        r, c = v["row"], v["col"]
        if not isinstance(r, int) or not isinstance(c, int):
            raise PlanError(f"块 {k} 的 row/col 必须是 int，实际是 {r!r}/{c!r}")
        if not (0 <= r < rows and 0 <= c < cols):
            raise PlanError(f"块 {k} 的格子 ({r},{c}) 越界（棋盘 {rows}×{cols}）")
        if (r, c) in seen_cells:
            raise PlanError(f"格子 ({r},{c}) 被块 {seen_cells[(r, c)]} 和块 {k} 重复占用")
        seen_cells[(r, c)] = k
    return True


class PlanError(ValueError):
    """plan 格式/取值错误。与"算法猜错"区分开。"""


def place(page, plan, pieces, layout, rect, delay=0.10, verbose=True):
    """按 plan 把块拖到格子，自动遵守锚点规则（从外圈往内）。

    返回 {"placed": n, "total": tot}。
    """
    rows, cols = layout["rows"], layout["cols"]
    validate_plan(plan, pieces, rows, cols)

    bx, by = layout["board"]["x"], layout["board"]["y"]
    pw, ph = layout["piece"]["w"], layout["piece"]["h"]
    by_index = {p["index"]: p for p in pieces}

    # 锚点顺序：外圈优先，然后逐层向内
    def on_border(cell):
        r, c = cell
        return r in (0, rows - 1) or c in (0, cols - 1)

    pending = [(k, (v["row"], v["col"])) for k, v in plan.items()]
    placed_cells = set()
    done = []

    for _ in range(len(pending) + rows * cols):
        ready = [(k, cell) for k, cell in pending
                 if cell not in placed_cells and (
                     on_border(cell) or any(
                         (cell[0] + dr, cell[1] + dc) in placed_cells
                         for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))]
        if not ready:
            break
        k, (r, c) = ready[0]
        p = by_index[k]
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        sx = rect["left"] + p["x"] + p["w"] / 2
        sy = rect["top"] + p["y"] + p["h"] / 2
        tx = rect["left"] + bx + c * pw + pw / 2
        ty = rect["top"] + by + r * ph + ph / 2
        drag(page, sx, sy, tx, ty)
        time.sleep(max(delay, 0.10))
        cur2, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if cur2 > cur:
            placed_cells.add((r, c))
            pending.remove((k, (r, c)))
            done.append(k)
            if verbose:
                print(f"  ✅ 块{k} -> ({r},{c}) {cur2}/{tot}")
        else:
            pending.remove((k, (r, c)))
            pending.append((k, (r, c)))       # 挪到队尾再试
            if verbose:
                print(f"  ❌ 块{k} -> ({r},{c}) 未吸附")
        if cur2 >= tot:
            break

    return {"placed": len(done), "total": len(plan),
            "state": state(page)}

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


from console import force_utf8   #这个是修控制台，我也不太懂

force_utf8()

from solver_core import (Analysis, anchor_order, compute_layout,
                          find_white_pieces, locate_piece, locate_piece_white)
from white_shape import board_color, cell_required_edges, white_piece_edges, white_piece_info

URL = "https://puzzle.moyunliuyin.me/?mode=exam"

# 默认考核配置：依次跑完 Part 1/2/3 × 9/16/25 共 9 关
PARTS = (1, 2, 3)         # 1=照片, 2=纯白, 3=混合
DIFFS = (9, 16, 25)       # 3×3 / 4×4 / 5×5
PAUSE = 5.0               # 每局通关后的中间停顿秒数（含通关弹窗展示与局间停顿）
START_TIMEOUT = 90.0      # 站点要下完 25+5 张图才切到 0/25，慢网络实测 10~40s

# 站点素材在仓库里有一份完全一样的副本（tools/download_catalog.py 下的），
# 用它喂给浏览器，开局就不用等 50 多张图从网上下来。
CATALOG_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"



#--------------------------------------------------------------- 素材缓存

def install_catalog_cache(context, quiet=False):
    served = {"n": 0, "miss": 0}
    mem_cache: dict[Path, bytes] = {}

    def handler(route):
        m = re.search(r"/assets/catalog/([^/?#]+)/([^/?#]+)", route.request.url)
        if not m:
            return route.continue_()
        path = CATALOG_DIR / m.group(1) / m.group(2)
        data = mem_cache.get(path)
        if data is None:
            if not path.exists():
                served["miss"] += 1
                return route.continue_()
            data = path.read_bytes()
            mem_cache[path] = data
        served["n"] += 1
        route.fulfill(body=data, content_type="image/jpeg",
                      headers={"cache-control": "public, max-age=604800"})

    context.route("**/assets/catalog/**", handler)
    if not quiet:
        have = CATALOG_DIR.exists()
        print(f"[素材] {'本地供给' if have else '本地缺失，回退联网'} {CATALOG_DIR}")
    return served






# ---------------------------------------------------------------- 页面操作

JS_CANVAS_RECT = """() => { const r = document.getElementById('puzzle-canvas').getBoundingClientRect();
  return JSON.stringify({left:r.left, top:r.top, w:r.width, h:r.height}); }"""

JS_CANVAS_PNG = "() => document.getElementById('puzzle-canvas').toDataURL('image/png')"

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

JS_START_READY = """() => {
  const b = document.getElementById('btn-exam-start');
  const panel = document.getElementById('exam-start');
  const im = document.getElementById('preview-img');
  if (!b) return JSON.stringify({ok: false, why: 'no-button'});
  const t = (b.innerText || '').trim();
  // 1) 静态 HTML 里按钮文字本来就是「开始这一部分」，但弹窗是 hidden 的；
  //    站点脚本绑定完点击事件后才会打开弹窗。
  // 2) 页面首次加载时 new JigsawApp() 会在后台异步跑一次默认 9 格的 loadTheme('animals')，
  //    完成后才会给 #preview-img 填上 data: URL。如果在这之前就点「开始这一部分」切到 16/25 格，
  //    两次 loadTheme 会并发竞态，触发 StateGuard "cells 9 != grid 16/25"。
  const open = panel ? !panel.hidden : false;
  const booted = !!(im && im.src && im.src.startsWith('data:'));
  return JSON.stringify({ok: open && booted && !b.disabled && t.includes('开始'),
                         text: t, disabled: b.disabled, open: open, booted: booted});
}"""

JS_SWITCH = """() => { const b = document.getElementById('btn-exam-switch');
  if (b) { b.click(); return 'switch'; } return 'none'; }"""

# 2026-11-16 版站点加了 StateGuard：判为篡改就把 board.locked 置 true，
# 之后所有拖拽静默失效（看起来"鼠标在动但一块都吸不上"）。开局后查一次，好在日志里说清楚。
JS_TAMPER = """() => { const d = document.documentElement.dataset.tamper;
  const el = document.getElementById('integrity-alert');
  return JSON.stringify({tamper: d || null, banner: el ? el.innerText.slice(0, 120) : null}); }"""


#---------------------------------------上方内容是JavaScript








def parse_progress(txt: str):
    """'12 / 25' -> (12, 25)"""
    try:
        a, b = txt.replace(" ", "").split("/")
        return int(a), int(b)
    except Exception:
        return 0, 0
#转换数据格式

def to_bgr(data_url: str):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 落子

def drag(page, sx, sy, tx, ty, steps=None, delay=0.0):
    if steps is None:
        steps = 12 if delay else 5
    page.mouse.move(sx, sy)
    page.mouse.down()
    if delay:
        time.sleep(delay)
    page.mouse.move(tx, ty, steps=steps)
    page.mouse.up()


def css_xy(rect, x, y):
    """画布像素坐标 -> 页面 CSS 坐标。

    为什么必须换算：canvas.width/height 不一定等于 CSS 尺寸。
    Windows 125%/150% 缩放、Retina、浏览器缩放都会让内部像素更大，
    直接拿画布像素当页面坐标去点，落点整体偏移，表现就是「一块都拼不上」。
    """
    return (rect["left"] + x * rect.get("sx", 1.0),
            rect["top"] + y * rect.get("sy", 1.0))


def drop_piece(page, rect, L, slot, cell, jitter=(0, 0), slow=0.0):
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    sx, sy = css_xy(rect, slot["x"] + jitter[0] + pw / 2,
                    slot["y"] + jitter[1] + ph / 2)
    tx, ty = css_xy(rect, L["board"]["x"] + cell[1] * pw + pw / 2,
                    L["board"]["y"] + cell[0] * ph + ph / 2)
    drag(page, sx, sy, tx, ty, delay=slow)
    time.sleep(0.015 if not slow else slow + 0.05)


# ---------------------------------------------------------------- 窗口

def screen_metrics(browser, fallback=(1600, 900, 1.0)):
    """真实屏幕可用尺寸 + 设备像素比。

    必须开一个**没有视口模拟**的临时上下文去问：一旦设了 viewport，
    Playwright 会把 screen 和 devicePixelRatio 一起模拟掉，业务页里读到的是
    假的 1280x800 / DPR 1.0（实测），按那个值放大视口等于没放大。
    """
    try:
        ctx = browser.new_context(no_viewport=True)
        pg = ctx.new_page()
        w, h = pg.evaluate("() => [screen.availWidth, screen.availHeight]")
        dpr = pg.evaluate("() => devicePixelRatio")
        ctx.close()
        return int(w), int(h), float(dpr)
    except Exception as e:
        print(f"[窗口] 读屏幕尺寸失败，用默认 {fallback}：{type(e).__name__}")
        return fallback


def maximize_window(page):
    """把浏览器窗口外框最大化（视口已经按屏幕尺寸设好）。"""
    try:
        cdp = page.context.new_cdp_session(page)
        wid = cdp.send("Browser.getWindowForTarget")["windowId"]
        cdp.send("Browser.setWindowBounds",
                 {"windowId": wid, "bounds": {"windowState": "maximized"}})
    except Exception as e:
        print(f"[窗口] 最大化窗口失败（继续用当前窗口）：{type(e).__name__}")


# ---------------------------------------------------------------- 开局

JS_SUB = ("() => { const e = document.getElementById('exam-part-sub');"
          " return e ? e.innerText : ''; }")

JS_PANEL = ("() => { const e = document.getElementById('exam-start');"
            " return e ? !e.hidden : false; }")

# 考核真正「开局成功」的判据（只看进度数字不够）：
#   考核页一打开，默认就是 Part 1 · 9 格，#progress-display 已经显示 "0 / 9"，
#   但此时「开始这一部分」弹窗还盖在画布上，考核根本没开始。旧代码看到 tot==9
#   就以为「已在运行」，于是拖拽全落在弹窗上 —— 这就是 Part 1 · 3×3 永远 0/9 的原因。
# 所以必须同时满足：
#   1. 弹窗 #exam-start 已隐藏
#   2. 横幅状态 #exam-state 是 data-state="running" 且文字为「进行中」
#      （点击后先是「准备中…」，素材加载完才变「进行中」）
#   3. 横幅徽标 #exam-part-badge 是目标部分（第 N 部分）
#   4. 进度分母 == 目标格数
JS_EXAM_STATUS = """() => {
  const q = id => document.getElementById(id);
  const panel = q('exam-start'), st = q('exam-state'), badge = q('exam-part-badge');
  const prog = q('progress-display'), btn = q('btn-exam-start');
  return JSON.stringify({
    panel_open: panel ? !panel.hidden : false,
    state: st ? (st.dataset.state || '') : '',
    label: st ? (st.textContent || '').trim() : '',
    badge: badge ? (badge.textContent || '').trim() : '',
    progress: prog ? prog.innerText : '',
    busy: btn ? btn.disabled : false,
  });
}"""


def exam_status(page):
    try:
        return json.loads(page.evaluate(JS_EXAM_STATUS))
    except Exception:
        return {}


def exam_running(s, part, diff):
    """s = exam_status() 的结果；判断目标 part/diff 是否真的在进行中。"""
    if not s:
        return False
    _, tot = parse_progress(s.get("progress", ""))
    return (not s.get("panel_open")
            and s.get("state") == "running"
            and "进行中" in s.get("label", "")
            and f"第 {part} 部分" in s.get("badge", "")
            and tot == diff)


def wait_start_ready(page, timeout=25.0):
    """等「开始这一部分」从『准备中…/禁用』变成可点。"""
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        try:
            last = json.loads(page.evaluate(JS_START_READY))
        except Exception as e:                       # 页面正在导航
            last = {"ok": False, "text": f"{type(e).__name__}"}
        if last.get("ok"):
            return last
        page.wait_for_timeout(100)
    return last


def start_exam(page, part, diff, url=URL, timeout=START_TIMEOUT, verbose=True):
    for attempt in range(1, 4):

        s = exam_status(page)
        if exam_running(s, part, diff):
            cur, tot = parse_progress(s["progress"])
            if verbose:
                print(f"[开局] 已在运行 {cur}/{tot}")
            return cur, tot


        if not s.get("panel_open"):
            try:
                page.evaluate(JS_SWITCH)
            except Exception:
                pass
            page.wait_for_timeout(200)

        ready = wait_start_ready(page, 25.0)
        if not ready.get("ok") and verbose:
            print(f"[开局 {attempt}] 按钮未就绪（{ready.get('text')}），仍在准备中，继续等…")

        r = page.evaluate(JS_START, [part, diff])
        if verbose:
            print(f"[开局 {attempt}] 点击开始 -> {r}")
        t0 = time.time()
        note = 0.0
        while time.time() - t0 < timeout:
            page.wait_for_timeout(100)
            s = exam_status(page)
            if not s:                                # 页面正在重绘/导航
                continue
            cur, tot = parse_progress(s.get("progress", ""))
            if exam_running(s, part, diff):
                if verbose:
                    print(f"[开局] 成功 {cur}/{tot}，用时 {time.time() - t0:.1f}s")
                return cur, tot
            sub = page.evaluate(JS_SUB) or ""
            if "无法准备" in sub:
                print(f"[开局 {attempt}] 站点报错：{sub}")
                break
            el = time.time() - t0
            if verbose and el >= note:
                print(f"[开局] 准备中 {el:.0f}s…（进度 {cur}/{tot}，"
                      f"等站点下完素材会自动切换）")
                note = el + 10

        # 超时：整页重载再试（重载会把卡住的 Image 请求清掉）
        if attempt < 3:
            print(f"[开局 {attempt}] {timeout:.0f}s 内没起来，整页重载重试…")
            try:
                page.goto(url, wait_until="load", timeout=60000)
                page.wait_for_timeout(2500)
            except Exception as e:
                print(f"[开局] 重载失败 {type(e).__name__}: {e}")

    raise RuntimeError(f"开局失败：{timeout:.0f}s 内进度没有变成 0/{diff}"
                       f"（part={part}）。多半是站点素材下载卡住，网络通了再跑。")


def solve_one_level(page, part, diff, classifier, cell_classifier):

    t_level = time.time()

    rows = cols = int(round(diff ** 0.5))#计算宽高
    n_decoys = round(diff * 0.2)  #诱饵数
    part_names = {1: "照片", 2: "纯白", 3: "混合"}
    print("\n" + "=" * 64)
    print(f"▶ 开始关卡：Part {part}（{part_names.get(part, '')}）· {rows}×{cols}（{diff} 格 + {n_decoys} 诱饵）")
    print("=" * 64)

    #清理class定义里的缓存，避免出现跨局使用数据倒置出错
    for clf in (classifier, cell_classifier):
        if clf is not None and hasattr(clf, "_cache"):
            clf._cache.clear()

    # 网站偶尔打不开，重试几次
    for gi in range(3):
        try:
            page.goto(URL, wait_until="load", timeout=45000)
            break
        except Exception as e:
            print(f"[goto 重试 {gi+1}] {type(e).__name__}")
            page.wait_for_timeout(1500)





    cur, tot = start_exam(page, part, diff, timeout=START_TIMEOUT)
    r = "start_exam"

    for _retry in range(1, 6):
        _t = json.loads(page.evaluate(JS_TAMPER))
        if not _t.get("tamper"):
            break
        print(f"[开局] 站点对账把这次开局判成篡改（{_t['tamper']}），刷新重开 {_retry}/5")
        page.goto(URL, wait_until="load", timeout=60000)
        cur, tot = start_exam(page, part, diff, timeout=START_TIMEOUT)
    else:
        raise RuntimeError("连续 5 次开局都撞上站点自己的对账窗口，稍后重试")

    canvas = None
    for wi in range(150):                # 最多等 15s
        page.wait_for_timeout(100)
        canvas = to_bgr(page.evaluate(JS_CANVAS_PNG))
        if float(canvas.mean()) > 5:
            break
    else:
        raise RuntimeError("画布一直空白（均值≈0），站点可能没渲染完成")
    page.wait_for_timeout(150)
    print(f"开局: part={part} diff={diff} rows={rows} decoys={n_decoys} -> {r}")
    print(f"就绪: 进度 {page.evaluate(JS_PROGRESS)}  画布均值 {canvas.mean():.1f}")

    rect = json.loads(page.evaluate(JS_CANVAS_RECT))
    canvas = to_bgr(page.evaluate(JS_CANVAS_PNG))
    rect["sx"] = rect["w"] / canvas.shape[1]
    rect["sy"] = rect["h"] / canvas.shape[0]
    if abs(rect["sx"] - 1) > 0.01 or abs(rect["sy"] - 1) > 0.01:
        print(f"[DPR] 画布 {canvas.shape[1]}x{canvas.shape[0]} -> "
              f"CSS {rect['w']:.0f}x{rect['h']:.0f}，坐标换算 sx={rect['sx']:.3f} sy={rect['sy']:.3f}")
    refinfo = json.loads(page.evaluate(JS_REF))
    ref = to_bgr(refinfo["src"])
    print(f"画布 {canvas.shape[1]}x{canvas.shape[0]}  参考图 {ref.shape[1]}x{ref.shape[0]}")


    A = Analysis(canvas, ref, rows, cols, n_decoys)
    res = A.match_photo_pieces(classifier=classifier,
                               cell_classifier=cell_classifier)
    if classifier is not None and A.photo_cells:
        print(f"YOLO 类别一致命中 {getattr(A, 'class_hits', 0)} 对（作为指派加分项）")
    print(f"白格 {len(A.white_cells)} 个: {[(c['row'], c['col']) for c in A.white_cells]}")
    print(f"白块 {len(res['white_slots'])} 个: {res['white_slots']}")
    print(f"照片映射 {len(res['assign'])} 个，诱饵 {len(res['decoys'])} 个: {res['decoys']}")


    

    L = A.L
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    slots = L["slots"]





    # ---- 按锚点顺序落照片块
    order = anchor_order(res["assign"])
    print(f"\n开始落子，锚点顺序 {len(order)} 步")
    placed = parse_progress(page.evaluate(JS_PROGRESS))[0]
    placed_cells: set[tuple[int, int]] = set()
    cell_img = {(c["row"], c["col"]): c["img"] for c in A.photo_cells}
    for k, i in enumerate(order, 1):
        a = res["assign"][i]
        cell = (a["row"], a["col"])
        before = placed
        drop_piece(page, rect, L, slots[i], cell, jitter=a["jitter"])
        after = parse_progress(page.evaluate(JS_PROGRESS))[0]

        tries = 0
        if after == before:
            # 若带 jitter 没吸上，可能是块还留在槽位或抖动估计偏了，直接退回原槽位交给后面补位阶段处理
            tx, ty = css_xy(rect, L["board"]["x"] + cell[1] * pw + pw / 2,
                            L["board"]["y"] + cell[0] * ph + ph / 2)
            px_back, py_back = css_xy(rect, slots[i]["x"] + pw / 2, slots[i]["y"] + ph / 2)
            drag(page, tx, ty, px_back, py_back, delay=0)
            time.sleep(0.02)

        ok = after > before
        if ok:
            placed_cells.add(cell)
        placed = after
        print(f"  {k:2d}/{len(order)} 块#{i:02d} -> {cell} "
              f"score={a['score']:.2f} {'✅' if ok else '❌'}"
              f"{' (重试)' if tries else ''} 进度 {after}/{diff}")

    # 若有照片格未吸上，用落选照片槽位 + 未吸附槽位按模板分从高到低逐个补试
    missed_photo = [(c["row"], c["col"]) for c in A.photo_cells
                    if (c["row"], c["col"]) not in placed_cells]
    if missed_photo:
        from solver_core import template_score
        failed_slots = [i for i in order if (res["assign"][i]["row"], res["assign"][i]["col"]) not in placed_cells]
        spare_slots = list(dict.fromkeys(list(res["decoys"]) + failed_slots))
        for mcell in list(missed_photo):
            ranked = sorted(
                spare_slots,
                key=lambda si: -template_score(res["pads"][si][0], cell_img[mcell], pw, ph)[0],
            )
            for si in ranked:
                sc_s, dx_s, dy_s = template_score(res["pads"][si][0], cell_img[mcell], pw, ph)
                before = placed
                drop_piece(page, rect, L, slots[si], mcell, jitter=(0, 0))
                after = parse_progress(page.evaluate(JS_PROGRESS))[0]
                if after > before:
                    placed = after
                    placed_cells.add(mcell)
                    spare_slots.remove(si)
                    print(f"  [补位] 块#{si:02d} -> {mcell} score={sc_s:.2f} ✅ 进度 {after}/{diff}")
                    break
                else:
                    tx, ty = css_xy(rect, L["board"]["x"] + mcell[1] * pw + pw / 2,
                                    L["board"]["y"] + mcell[0] * ph + ph / 2)
                    px_b, py_b = css_xy(rect, slots[si]["x"] + pw / 2, slots[si]["y"] + ph / 2)
                    drag(page, tx, ty, px_b, py_b, delay=0)
                    time.sleep(0.02)

    # ---- 白块求解：齿型约束（直边 + 邻接凹凸互补）+ 锚点顺序 + 缓存位置
    cur, tot = placed, diff
    if cur < tot and A.white_cells and res["white_slots"]:
        white_cells = [(c["row"], c["col"]) for c in A.white_cells
                       if (c["row"], c["col"]) not in placed_cells]
        print(f"\n白块：剩余 {tot - cur} 块，白格 {white_cells}")

        pure_white_mode = len(A.photo_cells) == 0
        wp_edges = {}
        white_pos = {}
        white_decoy = {}
        for i in res["white_slots"]:
            e, pos_xy, is_dec = white_piece_info(
                canvas, slots[i], pw, ph, pure_white_mode=pure_white_mode
            )
            wp_edges[i] = e
            white_pos[i] = pos_xy
            white_decoy[i] = is_dec
            tag = " [空心齿=诱饵]" if is_dec else ""
            print(f"  白块槽位#{i} 齿型 {e}  flat数={sum(1 for v in e.values() if v == 0)}{tag}")

        req_edges = {}
        if placed_cells:
            cv_board = to_bgr(page.evaluate(JS_CANVAS_PNG))
            bg_col = board_color(cv_board, L)
            for cell in white_cells:
                req_edges[cell] = cell_required_edges(cv_board, L, cell[0], cell[1], bg_col)

        placed_white_edges: dict[tuple[int, int], dict[str, int]] = {}
        used_white: set[int] = set()

        def has_white_anchor(cell):
            r, c = cell
            if r in (0, rows - 1) or c in (0, cols - 1):
                return True
            return any((r + dr, c + dc) in placed_cells
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

        def compat_rank(i, cell):
            e = wp_edges[i]
            b_bad = 0 if border_ok(e, cell) else 1
            r, c = cell
            mismatch = 0
            match = 0
            for dr, dc, side, opp in NEIGH:
                nb = (r + dr, c + dc)
                if nb in placed_white_edges:
                    ne = placed_white_edges[nb][opp]
                    if e[side] != 0 and ne != 0:
                        if e[side] + ne == 0:
                            match += 2
                        else:
                            mismatch += 2
                elif nb in placed_cells and cell in req_edges:
                    re_val = req_edges[cell][side]
                    if re_val != 0 and e[side] != 0:
                        if e[side] == re_val:
                            match += 1
                        else:
                            mismatch += 1
            n_border = (1 if r in (0, rows - 1) else 0) + (1 if c in (0, cols - 1) else 0)
            dec_pen = 1 if white_decoy.get(i, False) else 0
            return (dec_pen, b_bad, mismatch, -match, -n_border, i)

        def attempt_white(i, cell, refresh_loc=False):
            nonlocal cur, tot
            if refresh_loc or i not in white_pos:
                cv = to_bgr(page.evaluate(JS_CANVAS_PNG))
                _, pos_xy, _ = white_piece_info(
                    cv, slots[i], pw, ph, pure_white_mode=pure_white_mode
                )
                white_pos[i] = pos_xy
            cx, cy = white_pos[i]
            sx, sy = css_xy(rect, cx, cy)
            tx, ty = css_xy(rect, L["board"]["x"] + cell[1] * pw + pw / 2,
                            L["board"]["y"] + cell[0] * ph + ph / 2)
            before = cur
            drag(page, sx, sy, tx, ty)
            time.sleep(0.02)
            cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
            if cur > before:
                print(f"  白块#{i} -> {cell} ✅ 进度 {cur}/{tot}")
                used_white.add(i)
                white_cells.remove(cell)
                placed_cells.add(cell)
                placed_white_edges[cell] = wp_edges[i]
                return True
            print(f"  白块#{i} 试 {cell} ✗")
            park_cx, park_cy = slots[i]["x"] + pw / 2, slots[i]["y"] + ph / 2
            park_x, park_y = css_xy(rect, park_cx, park_cy)
            drag(page, tx, ty, park_x, park_y, delay=0)
            time.sleep(0.015)
            white_pos[i] = (park_cx, park_cy)
            return False

        for round_idx in range(3):
            if cur >= tot or not white_cells:
                break
            tried: set[tuple[int, tuple[int, int]]] = set()
            refresh = round_idx > 0
            if refresh:
                print(f"  白块补位第 {round_idx} 轮（剩 {len(white_cells)} 格）")
            while cur < tot and white_cells:
                anchored = [c for c in white_cells if has_white_anchor(c)] or list(white_cells)
                avail_slots = [i for i in res["white_slots"] if i not in used_white]
                cands = [
                    (compat_rank(i, cell), i, cell)
                    for cell in anchored
                    for i in avail_slots
                    if (i, cell) not in tried
                ]
                if not cands:
                    break
                cands.sort(key=lambda t: t[0])
                _, best_i, best_cell = cands[0]
                tried.add((best_i, best_cell))
                attempt_white(best_i, best_cell, refresh_loc=refresh)
                refresh = False

    cur, tot = parse_progress(page.evaluate(JS_PROGRESS))

    elapsed = round(time.time() - t_level, 1)
    _t = json.loads(page.evaluate(JS_TAMPER))
    if _t.get("tamper") and cur < tot:
        print(f"[warn] 本局被 StateGuard 判为篡改：{_t['tamper']}")
    print(f"\n=== Part {part} · {diff}格 结果 {cur}/{tot} | 用时 {elapsed}s | "
          f"{'🎉 通关' if cur == tot else '未完成'} ===")
    return cur, tot, elapsed


# ---------------------------------------------------------------- 完成证明导出与局间停顿

def default_evidence_dir(suite: str) -> Path:
    """默认按题型与时间戳生成本轮 JSON 完成证明保存目录。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent.parent / "runs" / "evidence" / f"{suite}_{stamp}"


def export_round_evidence_and_pause(
    page,
    *,
    prefix: str,
    cur: int,
    tot: int,
    evidence_dir: Path | None,
    pause: float = PAUSE,
    has_next: bool = False,
) -> Path | None:
    """单局结束后：
    1) 等通关弹窗呈现并作短暂停顿展示已完成棋盘/用时；
    2) 通过站点「导出本场证据（JSON）」按钮触发真实下载，保存本局官网 JSON 完成证明；
    3) 在切回开始面板显示已通关列表后再作局间停顿。
    """
    won = tot > 0 and cur >= tot
    victory_wait = min(pause, max(0.8, round(pause * 0.65, 2))) if pause > 0 else 0.0
    rest_wait = max(0.0, round(pause - victory_wait, 2)) if (pause > 0 and has_next) else 0.0

    if won:
        try:
            page.wait_for_selector("#victory-modal.show", timeout=3500)
        except Exception:
            pass
        if victory_wait > 0:
            print(f"[停顿] 展示通关弹窗 {victory_wait:.1f}s…")
            page.wait_for_timeout(int(victory_wait * 1000))

    saved_path: Path | None = None
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        try:
            modal_btn = page.locator("#victory-modal.show #btn-play-again")
            if modal_btn.count() > 0 and modal_btn.first.is_visible():
                modal_btn.first.click(timeout=3000)
                page.wait_for_timeout(120)

            st = exam_status(page)
            if not st.get("panel_open"):
                sw_btn = page.locator("#btn-exam-switch")
                if sw_btn.count() > 0 and sw_btn.first.is_visible():
                    sw_btn.first.click(timeout=3000)
                    page.wait_for_timeout(150)

            exp_btn = page.locator("#btn-export-evidence")
            exp_btn.wait_for(state="visible", timeout=3000)
            with page.expect_download(timeout=10000) as dl_info:
                exp_btn.click(timeout=3000)
            download = dl_info.value
            suggested = download.suggested_filename or "exam-evidence.json"
            tag = "" if won else "_incomplete"
            out_path = evidence_dir / f"{prefix}{tag}_{suggested}"
            download.save_as(str(out_path))
            saved_path = out_path

            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                n_ev = len(payload.get("events") or [])
                bld = (payload.get("meta") or {}).get("build", "")
                p_info = payload.get("part") or {}
                print(f"[证明] 已导出网站 JSON 完成证明 -> {out_path} "
                      f"(build={bld}, part={p_info.get('id')}, events={n_ev})")
            except Exception:
                print(f"[证明] 已导出网站 JSON 完成证明 -> {out_path}")
        except Exception as e:
            print(f"[warn] 网站 JSON 完成证明导出失败 ({prefix})：{type(e).__name__}: {e}")

    if rest_wait > 0:
        print(f"[停顿] 局间停顿 {rest_wait:.1f}s，准备进入下一关…")
        page.wait_for_timeout(int(rest_wait * 1000))

    return saved_path


# ---------------------------------------------------------------- 主流程

def main(auto_close: bool = False, evidence_dir: Path | None = None) -> int:
    schedule = [(pt, df) for pt in PARTS for df in DIFFS]
    ev_dir = Path(evidence_dir) if evidence_dir is not None else default_evidence_dir("basic")

    t_start = time.time()



    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False,
                                    args=["--no-sandbox", "--disable-dev-shm-usage"])

        aw, ah, scale = screen_metrics(browser)
        vw_css, vh_css = aw, max(600, ah - 8)
        print(f"[窗口] 屏幕 {aw}x{ah} · DPR {scale:g} -> 视口 {vw_css}x{vh_css}")

        ctx = browser.new_context(
            viewport={"width": vw_css, "height": vh_css},
            device_scale_factor=scale,
            accept_downloads=True,
        )
        page = ctx.new_page()
        maximize_window(page)

        page.on("dialog", lambda d: d.accept())
        install_catalog_cache(page.context)




        #这里我提前预热了一下模型
        classifier = None
        cell_classifier = None
        try:


            from yolo_check import Classifier, DET_WEIGHTS, CLS_WEIGHTS
            if DET_WEIGHTS.exists():
                cell_classifier = Classifier(weights=DET_WEIGHTS)
                print(f"YOLO 检测模型（格子）: {cell_classifier.path.name}"
                      f"（{len(cell_classifier.names)} 类）")
            if CLS_WEIGHTS.exists():
                classifier = Classifier(weights=CLS_WEIGHTS)
                print(f"YOLO 分类模型（碎片）: {classifier.path.name}"
                      f"（{len(classifier.names)} 类）")


            if classifier is None:
                classifier = cell_classifier
            if cell_classifier is None:
                cell_classifier = classifier
            #保证一下都有模型被加载，只是防范


            dummy = np.full((240, 320, 3), 128, np.uint8)
            for clf in {id(c): c for c in (classifier, cell_classifier) if c}.values():
                clf.predict(dummy)



        except Exception as e:
            print(f"[warn] YOLO 未启用，退回纯 OpenCV。错误原因：{e}")
            classifier = cell_classifier = None






        summary = []
        exported_files = []

        part_names = {1: "Part 1 · 照片", 2: "Part 2 · 纯白", 3: "Part 3 · 混合"}
        for idx, (pt, df) in enumerate(schedule):
            cur, tot, dt = solve_one_level(
                page, pt, df, classifier, cell_classifier
            )
            summary.append((pt, df, cur, tot, dt))
            proof_path = export_round_evidence_and_pause(
                page,
                prefix=f"{idx + 1:02d}_basic_p{pt}_{df}",
                cur=cur,
                tot=tot,
                evidence_dir=ev_dir,
                pause=PAUSE,
                has_next=(idx < len(schedule) - 1),
            )
            if proof_path is not None:
                exported_files.append(proof_path)




        total_dt = round(time.time() - t_start, 1)
        passed = sum(1 for _, _, c, t, _ in summary if c == t and t > 0)
        print("\n" + "=" * 64)
        print(f"全部关卡汇总：{passed}/{len(summary)} 关通关 | 总用时 {total_dt}s")
        print("-" * 64)
        for pt, df, c, t, dt in summary:
            r_sz = int(round(df ** 0.5))
            ok_s = "✅ 通关" if (c == t and t > 0) else "❌ 未完成"
            print(f"  {part_names.get(pt, f'Part {pt}'):<14} | {r_sz}×{r_sz} ({df:>2}格) | "
                  f"{c:>2}/{t:<2} | {dt:>5.1f}s | {ok_s}")
        print("-" * 64)
        print(f"网站 JSON 完成证明已导出 ({len(exported_files)}/{len(summary)} 份)：{ev_dir}")
        print("=" * 64)

        if auto_close:
            page.wait_for_timeout(1500)
        elif sys.stdin and sys.stdin.isatty():
            try:
                input("按回车关闭…")
            except (EOFError, KeyboardInterrupt):
                pass
        page.close()
        ctx.close()
        browser.close()
        all_ok = all(c == t and t > 0 for _, _, c, t, _ in summary)
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""puzzle.moyunliuyin.me 考核自动解题（Part 3 混合 · 5×5）。

用法
----
    python run_exam.py                      # 无头跑一次
    python run_exam.py --headful            # 弹窗口（录演示视频用）
    python run_exam.py --part 3 --diff 25   # 指定部分与难度
    python run_exam.py --slow 0.15          # 每步放慢，便于肉眼观察/录制

流程
----
1. 开考核局（Part/难度通过 radio 选择）
2. 从 canvas 取像素：参考图 + 拼图画布
3. 复现布局几何 -> 30 个槽位坐标
4. 参考图切 5x5：5 个纯白格 + 20 个照片格
5. 块分类：中心区纯白 -> 白块；否则照片块
6. 照片块 vs 照片格做匈牙利全局指派 -> 20 个映射，落选 5 块 = 诱饵
7. 按锚点规则（外圈 -> 逐层向内）依次拖拽落子
8. 剩下 5 个白块用「排除法 + 吸附反馈」补完
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

from solver_core import (Analysis, anchor_order, compute_layout,
                          find_white_pieces, locate_piece, locate_piece_white)
from white_shape import white_piece_edges

URL = "https://puzzle.moyunliuyin.me/?mode=exam"


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

JS_SWITCH = """() => { const b = document.getElementById('btn-exam-switch');
  if (b) { b.click(); return 'switch'; } return 'none'; }"""


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

def drag(page, sx, sy, tx, ty, steps=12, delay=0.0):
    page.mouse.move(sx, sy)
    page.mouse.down()
    if delay:
        time.sleep(delay)
    page.mouse.move(tx, ty, steps=steps)
    page.mouse.up()


def drop_piece(page, rect, L, slot, cell, jitter=(0, 0), slow=0.0):
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    sx = rect["left"] + slot["x"] + jitter[0] + pw / 2
    sy = rect["top"] + slot["y"] + jitter[1] + ph / 2
    tx = rect["left"] + L["board"]["x"] + cell[1] * pw + pw / 2
    ty = rect["top"] + L["board"]["y"] + cell[0] * ph + ph / 2
    drag(page, sx, sy, tx, ty, delay=slow)
    time.sleep(0.08 if not slow else slow + 0.05)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=3)
    ap.add_argument("--diff", type=int, default=25)
    ap.add_argument("--headful", action="store_true",default=True, help="弹窗口（演示用）")
    ap.add_argument("--slow", type=float, default=0.0, help="每步放慢秒数")
    ap.add_argument("--shots", default=None, help="截图输出目录")
    ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--video", default=None, help="录屏目录（WebM），用于演示视频")
    ap.add_argument("--no-yolo", action="store_true", help="不加载 YOLO，只用 OpenCV")
    args = ap.parse_args()


    #---------------------------------------------------------------处理传参

    rows = cols = int(round(args.diff ** 0.5))
    n_decoys = round(args.diff * 0.2) if args.part == 3 else 0
    # Part 1/2 也带诱饵，按站点规则 20%
    n_decoys = round(args.diff * 0.2)
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headful,
                                    args=["--no-sandbox", "--disable-dev-shm-usage"])

        #-----------------------------------------------------------
        if args.video:
            vdir = Path(args.video)
            vdir.mkdir(parents=True, exist_ok=True)
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 800},
                record_video_dir=str(vdir),
                record_video_size={"width": 1280, "height": 800})
            page = ctx.new_page()
        else:
            ctx = None
            page = browser.new_page(viewport={"width": 1280, "height": 800})


        #-----------------------------------------------------------



        # 站点偶尔很慢，goto 也可能超时，重试几次
        for gi in range(3):
            try:
                page.goto(URL, wait_until="load", timeout=45000)
                break
            except Exception as e:
                print(f"[goto 重试 {gi+1}] {type(e).__name__}")
                page.wait_for_timeout(2000)
        page.wait_for_timeout(2500)      # 站点脚本初始化较慢，等够再点

        # 稳健开局：点「开始这一部分」后必须看到进度变成 0/diff，否则重试。
        # 之前只等 1.2s 就点，点空了却继续跑，结果拿着上一局的 9 格棋盘去分析，
        # 表现为"一块都找不到 / 结果 0/N"。
        r = None
        cur = tot = 0
        for attempt in range(4):
            page.evaluate(JS_SWITCH)
            page.wait_for_timeout(800)
            r = page.evaluate(JS_START, [args.part, args.diff])
            for _ in range(50):                      # 最多等 10s
                page.wait_for_timeout(200)
                cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
                if tot == args.diff:
                    break
            if tot == args.diff:
                break
            print(f"[重试 {attempt+1}] 开局未生效（进度 {cur}/{tot}，期望 0/{args.diff}）")
        if tot != args.diff:
            raise RuntimeError(
                f"开局失败：进度 {cur}/{tot}，期望 0/{args.diff}"
                f"（part={args.part} diff={args.diff} 返回值 {r}）")

        # 等画布真正绘制出来再分析。
        # 实测开局后画布会有几秒钟是全黑（均值 0.0），此时分析会得到"一块都找不到"。
        canvas = None
        for wi in range(50):                 # 最多等 15s
            page.wait_for_timeout(300)
            canvas = to_bgr(page.evaluate(JS_CANVAS_PNG))
            if float(canvas.mean()) > 5:
                break
        else:
            raise RuntimeError("画布一直空白（均值≈0），站点可能没渲染完成")
        page.wait_for_timeout(700)
        canvas = to_bgr(page.evaluate(JS_CANVAS_PNG))
        print(f"开局: part={args.part} diff={args.diff} rows={rows} decoys={n_decoys} -> {r}")
        print(f"就绪: 进度 {page.evaluate(JS_PROGRESS)}  画布均值 {canvas.mean():.1f}")

        rect = json.loads(page.evaluate(JS_CANVAS_RECT))
        canvas = to_bgr(page.evaluate(JS_CANVAS_PNG))
        refinfo = json.loads(page.evaluate(JS_REF))
        ref = to_bgr(refinfo["src"])
        if shots:
            cv2.imwrite(str(shots / "01_start_canvas.png"), canvas)
        print(f"画布 {canvas.shape[1]}x{canvas.shape[0]}  参考图 {ref.shape[1]}x{ref.shape[0]}")

        # ---- YOLO（可选：模型缺失时自动降级为纯 OpenCV）
        # 分工：检测模型（YOLO11s）管**格子**——格子里是完整物体，实测 20/20 检出；
        #      分类模型管**拼图块**——块是局部碎片，检测模型会乱认。
        classifier = None          # 拼图块用
        cell_classifier = None     # 参考格用
        if not args.no_yolo:
            try:
                from yolo_check import Classifier
                from yolo_check import DET_WEIGHTS, CLS_WEIGHTS
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
            except Exception as e:
                print(f"[warn] YOLO 未启用，退回纯 OpenCV：{e}")

        # ---- 视觉分析
        A = Analysis(canvas, ref, rows, cols, n_decoys)
        res = A.match_photo_pieces(classifier=classifier,
                                   cell_classifier=cell_classifier)
        if classifier is not None:
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
        placed = 0
        cell_of = {i: (a["row"], a["col"]) for i, a in res["assign"].items()}
        cell_img = {(c["row"], c["col"]): c["img"] for c in A.photo_cells}
        for k, i in enumerate(order, 1):
            a = res["assign"][i]
            cell = (a["row"], a["col"])
            before = parse_progress(page.evaluate(JS_PROGRESS))[0]
            drop_piece(page, rect, L, slots[i], cell, jitter=a["jitter"], slow=args.slow)
            after = parse_progress(page.evaluate(JS_PROGRESS))[0]

            # 失败重试：重新在画布上按内容定位这块，再拖一次（最多 2 次）
            tries = 0
            while after == before and tries < 2:
                tries += 1
                cv_now = to_bgr(page.evaluate(JS_CANVAS_PNG))
                loc = locate_piece(cv_now, cell_img[cell], pw, ph, L)
                if not loc:
                    break
                sx = rect["left"] + loc["cx"]
                sy = rect["top"] + loc["cy"]
                tx = rect["left"] + L["board"]["x"] + cell[1] * pw + pw / 2
                ty = rect["top"] + L["board"]["y"] + cell[0] * ph + ph / 2
                drag(page, sx, sy, tx, ty, delay=args.slow)
                time.sleep(0.10)
                after = parse_progress(page.evaluate(JS_PROGRESS))[0]

            ok = after > before
            placed = after
            print(f"  {k:2d}/{len(order)} 块#{i:02d} -> {cell} "
                  f"score={a['score']:.2f} {'✅' if ok else '❌'}"
                  f"{' (重试)' if tries else ''} 进度 {after}/{args.diff}")

        if shots:
            cv2.imwrite(str(shots / "02_after_photo.png"), to_bgr(page.evaluate(JS_CANVAS_PNG)))

        # ---- 白块用排除法：每次先重新扫描白块位置（失败后块会被拖到别处，不能再用原槽位）
        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if cur < tot:
            white_cells = [(c["row"], c["col"]) for c in A.white_cells]
            print(f"\n白块：剩余 {tot - cur} 块，白格 {white_cells}")

            # 每个白块的齿型（用开局画布算，此时块都在托盘原位）
            wp_edges = {}
            for i in res["white_slots"]:
                e = white_piece_edges(canvas, slots[i], pw, ph)
                wp_edges[i] = e
                print(f"  白块槽位#{i} 齿型 {e}  flat数={sum(1 for v in e.values() if v == 0)}")

            def border_ok(edges, cell):
                """边界格必须有对应侧直边；内部侧不能有直边。"""
                r, c = cell
                need = {}
                if r == 0:
                    need["top"] = 0
                elif r == rows - 1:
                    need["bottom"] = 0
                else:
                    need["top"] = "nonflat"; need["bottom"] = "nonflat"
                # 左右同理
                sides = dict(need)
                if c == 0:
                    sides["left"] = 0
                elif c == cols - 1:
                    sides["right"] = 0
                else:
                    sides["left"] = "nonflat"; sides["right"] = "nonflat"
                for s, want in sides.items():
                    v = edges[s]
                    if want == 0 and v != 0:
                        return False
                    if want == "nonflat" and v == 0:
                        return False
                return True

            # 生成候选 (槽位, 格子)，按约束满足优先
            cands = []
            for i in res["white_slots"]:
                for cell in white_cells:
                    cands.append((0 if border_ok(wp_edges[i], cell) else 1, i, cell))
            cands.sort()

            tried = set()
            for _, i, cell in cands:
                if cur >= tot:
                    break
                if (i, cell) in tried:
                    continue
                tried.add((i, cell))
                cv = to_bgr(page.evaluate(JS_CANVAS_PNG))
                loc = locate_piece_white(cv, L, pw, ph, slots[i])
                if loc is None:
                    continue
                sx = rect["left"] + loc["cx"]
                sy = rect["top"] + loc["cy"]
                tx = rect["left"] + L["board"]["x"] + cell[1] * pw + pw / 2
                ty = rect["top"] + L["board"]["y"] + cell[0] * ph + ph / 2
                before = cur
                drag(page, sx, sy, tx, ty, delay=args.slow)
                time.sleep(0.10)
                cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
                if cur > before:
                    print(f"  白块#{i} -> {cell} ✅ 进度 {cur}/{tot}")
                    white_cells.remove(cell)
                else:
                    print(f"  白块#{i} 试 {cell} ✗")
                    # 送回原槽位，避免落到棋盘上"消失"
                    park_x = rect["left"] + slots[i]["x"] + pw / 2
                    park_y = rect["top"] + slots[i]["y"] + ph / 2
                    drag(page, tx, ty, park_x, park_y, delay=0)
                    time.sleep(0.06)
            white_cells = []

        cur, tot = parse_progress(page.evaluate(JS_PROGRESS))
        if shots:
            cv2.imwrite(str(shots / "03_final.png"), to_bgr(page.evaluate(JS_CANVAS_PNG)))
            page.screenshot(path=str(shots / "04_page.png"))

        elapsed = round(time.time() - t_start, 1)
        print(f"\n=== 结果 {cur}/{tot} | 用时 {elapsed}s | "
              f"{'🎉 通关' if cur == tot else '未完成'} ===")
        if args.headful:
            try:
                input("按回车关闭…")
            except EOFError:
                pass
        if ctx is not None:
            page.close()          # 关页面才会落盘视频
            ctx.close()
        browser.close()
        return 0 if cur == tot else 1


if __name__ == "__main__":
    sys.exit(main())

"""place_exam 的端到端自测：开局 -> 导出 -> 用求解器出 plan -> 落子 -> 读进度。"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))          # browser/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'solver'))  # solver/

from playwright.sync_api import sync_playwright
from place_exam import open_exam, export, place, state
from solver_core import Analysis

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
    page = br.new_page(viewport={"width": 1280, "height": 800})
    t0 = time.time()
    info = open_exam(page)
    print("open_exam:", info)
    assets = export(page, "/tmp/place_test", rows=info["rows"], cols=info["cols"])
    print(f"export: 画布 {assets['canvas'].shape}, 参考图 "
          f"{None if assets['ref'] is None else assets['ref'].shape}, "
          f"块 {len(assets['pieces'])} 个（白块 {sum(1 for x in assets['pieces'] if x['white'])}）")

    A = Analysis(assets["canvas"], assets["ref"], info["rows"], info["cols"], 5)
    res = A.match_photo_pieces()
    plan = {i: {"row": v["row"], "col": v["col"]} for i, v in res["assign"].items()}
    print(f"plan: {len(plan)} 步，诱饵 {len(res['decoys'])} 个，白格 {res['white_cells']}")

    out = place(page, plan, assets["pieces"], assets["layout"], assets["rect"])
    print("place:", out)
    print("耗时", round(time.time()-t0,1), "s")
    br.close()

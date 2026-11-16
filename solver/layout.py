"""复现考核站的布局几何（源码 setupBoardLayout / applyBoardGeometry / traySlots）。

这是布局数学，不含任何答案信息：棋盘矩形、格子尺寸、托盘槽位位置。
所有拼图块必然落在这些槽位之一。
"""
##这个并未参与解决，而是提前估计

def compute(cw, ch, rows, cols, n_decoys=0, overlap=False, center=False):
    padding = 30
    max_board_w = min(cw * 0.65, 750)
    max_board_h = min(ch - padding * 2, 550)

    bw = max_board_w
    bh = bw * 0.75
    if bh > max_board_h:
        bh = max_board_h
        bw = bh / 0.75

    wanted = 0 if (overlap or center) else rows * cols + n_decoys

    def geometry(scale):
        width, height = bw * scale, bh * scale
        board_x = max(padding, (cw - width) / 2)
        board_y = (ch - height) / 2
        return width, height, board_x, board_y

    def slots_for(width, height, board_x, board_y):
        pw, ph = width / cols, height / rows
        if not pw or not ph:
            return []
        step_x = pw * 1.16
        step_y = ph * 1.16
        band = max(0.0, board_x)
        cols_per_side = int(band / step_x)
        n_rows = int(ch / step_y)
        out = []
        for side in range(2):
            origin_x = 0.0 if side == 0 else board_x + width
            for col in range(cols_per_side):
                for row in range(n_rows):
                    x = origin_x + col * step_x + (step_x - pw) / 2
                    y = row * step_y + (step_y - ph) / 2
                    if x < 0 or y < 0:
                        continue
                    if x + pw > cw or y + ph > ch:
                        continue
                    out.append({"x": x, "y": y, "side": side})
        return out

    scale = 1.0
    chosen = None
    while scale >= 0.3:
        w_, h_, bx_, by_ = geometry(scale)
        if len(slots_for(w_, h_, bx_, by_)) >= wanted:
            chosen = scale
            break
        scale -= 0.04
    final = scale if chosen is None else chosen

    width, height, board_x, board_y = geometry(final)
    pw, ph = width / cols, height / rows
    return {
        "scale": final,
        "canvas": (cw, ch),
        "board": {"x": board_x, "y": board_y, "w": width, "h": height},
        "piece": {"w": pw, "h": ph},
        "rows": rows, "cols": cols,
        "slots": slots_for(width, height, board_x, board_y),
    }


if __name__ == "__main__":
    cw, ch = 919, 375
    for rows, decoys in ((5, 5), (5, 0), (4, 4), (3, 2)):
        L = compute(cw, ch, rows, rows, decoys)
        print(f"{rows}x{rows} decoys={decoys}: scale={L['scale']:.2f} "
              f"board=({L['board']['x']:.1f},{L['board']['y']:.1f}) "
              f"{L['board']['w']:.1f}x{L['board']['h']:.1f} "
              f"piece={L['piece']['w']:.1f}x{L['piece']['h']:.1f} "
              f"slots={len(L['slots'])}")
        if rows == 5 and decoys == 5:
            print("  前 6 个槽位:", [(round(s['x'],1), round(s['y'],1), s['side']) for s in L['slots'][:6]])

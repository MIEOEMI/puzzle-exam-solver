"""白块按齿型匹配（确定性，不靠试错）。

两条独立的像素证据
------------------
A. 白格需要什么齿型（从棋盘上读）
   白格四邻的照片块已经放好，边界处能直接看出：
     - 照片内容凸进白格  -> 邻居是凸齿 -> 白格这侧必须是凹槽 (+1)
     - 深色背景咬进照片  -> 邻居是凹槽 -> 白格这侧必须是凸齿 (-1)
     - 一条直线          -> 直边 (0)

B. 白块自己是什么齿型（从托盘读）
   白块是纯白填充，掩码只含自己（邻居的凸齿是照片内容，不是白色，天然排除）：
     - 格子框外有白色      -> 凸齿 (-1)
     - 格子框内边缘有黑     -> 凹槽 (+1)
     - 都不是              -> 直边 (0)
"""
import numpy as np


def white_piece_info(canvas, slot, pw, ph, pure_white_mode=False):
    """返回 (edges, (cx, cy), is_decoy)：
      - edges: {'top': -1/0/1, 'right': ..., 'bottom': ..., 'left': ...}
      - (cx, cy): 用肩部直边（0.12~0.28 与 0.72~0.88）精确对齐后的块中心坐标，
        不受邻块凸齿粘连影响，彻底解决连通域把两块粘连白块合并/漏检的问题
      - is_decoy: 纯白模式下，诱饵块从单格画布裁出，其凸齿内部是透明黑底、只有一圈
        半透明白描边（空心齿）；检测到空心齿即可 100% 判定为诱饵块。
    """
    H, W = canvas.shape[:2]
    b, g, r = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    white = ((b > 248) & (g > 248) & (r > 248)).astype(np.uint8)
    fg = (canvas.astype(np.int32).sum(axis=2) > 35).astype(np.uint8)
    gray_line = (canvas.min(axis=2) > 70).astype(np.uint8)

    tab = min(pw, ph) * 0.22
    pad = 8
    x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
    x1, y1 = int(round(slot["x"] + pw)), int(round(slot["y"] + ph))
    sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
    sx1, sy1 = min(W, x1 + pad), min(H, y1 + pad)
    sub = white[sy0:sy1, sx0:sx1]
    sh_h, sh_w = sub.shape

    cols_sh = [c for c in range(sh_w)
               if 0.12 <= (c - (x0 - sx0)) / pw <= 0.28
               or 0.72 <= (c - (x0 - sx0)) / pw <= 0.88]
    rows_sh = [r for r in range(sh_h)
               if 0.12 <= (r - (y0 - sy0)) / ph <= 0.28
               or 0.72 <= (r - (y0 - sy0)) / ph <= 0.88]

    if cols_sh and rows_sh and sub.sum() >= 50:
        by0 = sy0 + int(np.median([int(np.argmax(sub[:, c])) for c in cols_sh]))
        by1 = sy0 + int(np.median([sh_h - 1 - int(np.argmax(sub[::-1, c])) for c in cols_sh]))
        bx0 = sx0 + int(np.median([int(np.argmax(sub[r, :])) for r in rows_sh]))
        bx1 = sx0 + int(np.median([sh_w - 1 - int(np.argmax(sub[r, ::-1])) for r in rows_sh]))
    else:
        bx0, by0, bx1, by1 = x0, y0, x1, y1

    cx_f, cy_f = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    cx, cy = int(round(cx_f)), int(round(cy_f))
    hx, hy = max(3, int(pw * 0.14)), max(3, int(ph * 0.14))
    hx_w, hy_w = max(4, int(pw * 0.20)), max(4, int(ph * 0.20))
    d_in = max(4, int(round(tab * 0.75)))
    d_out = max(4, int(round(tab * 0.60)))
    dm = max(3, int(round(tab * 0.50)))

    slices = {
        "top": (
            (slice(max(0, by0 - d_out), max(0, by0 - 2)), slice(max(0, cx - hx), min(W, cx + hx))),
            (slice(min(H, by0 + 2), min(H, by0 + d_in)), slice(max(0, cx - hx), min(W, cx + hx))),
            wh_d2 := float(white[max(0, by0 - 2), max(0, cx - hx_w):min(W, cx + hx_w)].sum()) / pw,
            wh_dm := float(white[max(0, by0 - dm), max(0, cx - hx_w):min(W, cx + hx_w)].sum()) / pw,
        ),
        "bottom": (
            (slice(min(H, by1 + 3), min(H, by1 + d_out)), slice(max(0, cx - hx), min(W, cx + hx))),
            (slice(max(0, by1 - d_in), max(0, by1 - 2)), slice(max(0, cx - hx), min(W, cx + hx))),
            float(white[min(H - 1, by1 + 2), max(0, cx - hx_w):min(W, cx + hx_w)].sum()) / pw,
            float(white[min(H - 1, by1 + dm), max(0, cx - hx_w):min(W, cx + hx_w)].sum()) / pw,
        ),
        "left": (
            (slice(max(0, cy - hy), min(H, cy + hy)), slice(max(0, bx0 - d_out), max(0, bx0 - 2))),
            (slice(max(0, cy - hy), min(H, cy + hy)), slice(min(W, bx0 + 2), min(W, bx0 + d_in))),
            float(white[max(0, cy - hy_w):min(H, cy + hy_w), max(0, bx0 - 2)].sum()) / ph,
            float(white[max(0, cy - hy_w):min(H, cy + hy_w), max(0, bx0 - dm)].sum()) / ph,
        ),
        "right": (
            (slice(max(0, cy - hy), min(H, cy + hy)), slice(min(W, bx1 + 3), min(W, bx1 + d_out))),
            (slice(max(0, cy - hy), min(H, cy + hy)), slice(max(0, bx1 - d_in), max(0, bx1 - 2))),
            float(white[max(0, cy - hy_w):min(H, cy + hy_w), min(W - 1, bx1 + 2)].sum()) / ph,
            float(white[max(0, cy - hy_w):min(H, cy + hy_w), min(W - 1, bx1 + dm)].sum()) / ph,
        ),
    }

    edges = {}
    hollow = 0
    for side in ("top", "right", "bottom", "left"):
        out_sl, in_sl, w_d2, w_dm = slices[side]
        in_seg = white[in_sl]
        i_gap = float(1.0 - in_seg.mean()) if in_seg.size else 0.0
        o_wh = float(white[out_sl].mean()) if white[out_sl].size else 0.0
        o_fg = float(fg[out_sl].mean()) if fg[out_sl].size else 0.0
        o_gl = float(gray_line[out_sl].mean()) if gray_line[out_sl].size else 0.0

        if i_gap >= 0.35:
            edges[side] = 1         # 框内明显缺一块白 -> 凹槽 (+1)
        elif o_wh >= 0.35:
            # 若紧贴边线 (d=2px) 宽度偏窄且比中段更窄，说明是邻格凸齿尖伸过来，本侧其实是直边
            if w_d2 < 0.205 and w_d2 < w_dm:
                edges[side] = 0
            else:
                edges[side] = -1
        elif not pure_white_mode and o_fg >= 0.35:
            edges[side] = -1        # 混合模式：白块凸齿被裁上了邻居照片的像素
        elif pure_white_mode and o_gl >= 0.06:
            edges[side] = -1        # 纯白模式：空心半透明描边凸齿 -> 诱饵块
            hollow += 1
        else:
            edges[side] = 0
    return edges, (cx_f, cy_f), (hollow > 0)


def white_piece_edges(canvas, slot, pw, ph):
    edges, _, _ = white_piece_info(canvas, slot, pw, ph, pure_white_mode=False)
    return edges


def board_color(canvas, L):
    """取棋盘底色（#0f172a -> BGR 约 [42, 23, 15]）。"""
    return np.array([42.0, 23.0, 15.0], dtype=np.float32)


def cell_required_edges(canvas, L, row, col, bg=None):
    """从棋盘像素读出该格四边需要的齿型（邻居凸进=+1，邻居凹进=-1，边界/未知=0）。

    采样位置选在齿头区域（距边 0.35*tab ~ 0.95*tab），避开细窄的齿颈。
    """
    rows, cols = L["rows"], L["cols"]
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    bx = L["board"]["x"] + col * pw
    by = L["board"]["y"] + row * ph
    if bg is None:
        bg = board_color(canvas, L)
    dist = np.linalg.norm(canvas.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    content = (dist > 35).astype(np.uint8)

    tab = min(pw, ph) * 0.22
    d0 = max(3, int(round(tab * 0.35)))
    d1 = max(d0 + 2, int(round(tab * 0.95)))
    hx, hy = max(3, int(pw * 0.12)), max(3, int(ph * 0.12))
    cxm, cym = int(bx + pw / 2), int(by + ph / 2)

    def probe(side):
        """返回 (向格内凸进的比例, 邻居侧紧贴边界的内容比例)"""
        if side == "top":
            into = content[int(by) + d0:int(by) + d1, cxm - hx:cxm + hx]
            away = content[int(by) - d1:int(by) - d0, cxm - hx:cxm + hx]
        elif side == "bottom":
            into = content[int(by + ph) - d1:int(by + ph) - d0, cxm - hx:cxm + hx]
            away = content[int(by + ph) + d0:int(by + ph) + d1, cxm - hx:cxm + hx]
        elif side == "left":
            into = content[cym - hy:cym + hy, int(bx) + d0:int(bx) + d1]
            away = content[cym - hy:cym + hy, int(bx) - d1:int(bx) - d0]
        else:
            into = content[cym - hy:cym + hy, int(bx + pw) - d1:int(bx + pw) - d0]
            away = content[cym - hy:cym + hy, int(bx + pw) + d0:int(bx + pw) + d1]
        if into.size == 0 or away.size == 0:
            return 0.0, 0.0
        return float(into.mean()), float(away.mean())

    border_side = {
        "top": row == 0,
        "bottom": row == rows - 1,
        "left": col == 0,
        "right": col == cols - 1,
    }
    edges = {}
    for side in ("top", "right", "bottom", "left"):
        if border_side[side]:
            edges[side] = 0
            continue
        into, away = probe(side)
        if into > 0.45:
            edges[side] = 1        # 邻居凸进本格 -> 本格要凹槽 (+1)
        elif away < 0.45:
            edges[side] = -1       # 邻居侧被挖空 -> 邻居是凹槽 -> 本格要凸齿 (-1)
        else:
            edges[side] = 0
    return edges

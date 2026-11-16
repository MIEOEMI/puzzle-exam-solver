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


def white_piece_edges(canvas, slot, pw, ph):
    """白块自身四边齿型（用纯白掩码，不会被邻块污染）。"""
    b, g, r = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    white = ((b > 250) & (g > 250) & (r > 250)).astype(np.uint8)

    H, W = white.shape
    tab = min(pw, ph) * 0.22
    margin = tab * 1.5
    x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
    x1, y1 = int(round(slot["x"] + pw)), int(round(slot["y"] + ph))

    px0, py0 = max(0, int(round(x0 - margin))), max(0, int(round(y0 - margin)))
    px1, py1 = min(W, int(round(x1 + margin))), min(H, int(round(y1 + margin)))
    m = white[py0:py1, px0:px1]
    lx0, ly0 = x0 - px0, y0 - py0
    lx1, ly1 = x1 - px0, y1 - py0

    cw, chh = lx1 - lx0, ly1 - ly0
    hx, hy = max(2, int(cw * 0.18)), max(2, int(chh * 0.18))
    cxm, cym = (lx0 + lx1) // 2, (ly0 + ly1) // 2
    depth = max(3, int(round(tab * 0.8)))

    def outside(side):
        if side == "top":
            seg = m[:max(0, ly0 - 2), max(0, lx0):lx1]
        elif side == "bottom":
            seg = m[min(m.shape[0], ly1 + 2):, max(0, lx0):lx1]
        elif side == "left":
            seg = m[max(0, ly0):ly1, :max(0, lx0 - 2)]
        else:
            seg = m[max(0, ly0):ly1, min(m.shape[1], lx1 + 2):]
        return int(seg.sum()) if seg.size else 0

    def gap(side):
        if side == "top":
            seg = m[ly0:ly0 + depth, cxm - hx:cxm + hx]
        elif side == "bottom":
            seg = m[ly1 - depth:ly1, cxm - hx:cxm + hx]
        elif side == "left":
            seg = m[cym - hy:cym + hy, lx0:lx0 + depth]
        else:
            seg = m[cym - hy:cym + hy, lx1 - depth:lx1]
        return float(1 - seg.mean()) if seg.size else 0.0

    edges = {}
    for side in ("top", "right", "bottom", "left"):
        oc, gp = outside(side), gap(side)
        if oc > max(30, tab * 2):
            edges[side] = -1
        elif gp >= 0.30:
            edges[side] = 1
        else:
            edges[side] = 0
    return edges


def board_color(canvas, L):
    """取棋盘底色（空棋盘区域的中位数）。"""
    bx, by = L["board"]["x"], L["board"]["y"]
    bw, bh = L["board"]["w"], L["board"]["h"]
    patch = canvas[int(by + bh * 0.45):int(by + bh * 0.55),
                   int(bx + bw * 0.45):int(bx + bw * 0.55)]
    return np.median(patch.reshape(-1, 3), axis=0) if patch.size else np.array([0, 0, 0])


def cell_required_edges(canvas, L, row, col, bg=None):
    """从棋盘像素读出该格四边需要的齿型（邻居凸进=1，邻居凹进=-1，直线=0）。

    关键：棋盘底色是深蓝（BGR≈42,23,15），亮度和 80，用绝对亮度会把底色当内容。
    必须用「与底色的距离」判断是否被拼图内容覆盖。
    """
    pw, ph = L["piece"]["w"], L["piece"]["h"]
    bx = L["board"]["x"] + col * pw
    by = L["board"]["y"] + row * ph
    if bg is None:
        bg = board_color(canvas, L)
    dist = np.linalg.norm(canvas.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    content = (dist > 45).astype(np.uint8)

    d = max(4, int(min(pw, ph) * 0.10))
    hx, hy = int(pw * 0.18), int(ph * 0.18)
    cxm, cym = int(bx + pw / 2), int(by + ph / 2)

    def probe(side):
        """返回 (向格内凸进的比例, 向邻居内凹的比例)"""
        if side == "top":
            into = content[int(by):int(by) + d, cxm - hx:cxm + hx]
            away = content[int(by) - d:int(by), cxm - hx:cxm + hx]
        elif side == "bottom":
            into = content[int(by + ph) - d:int(by + ph), cxm - hx:cxm + hx]
            away = content[int(by + ph):int(by + ph) + d, cxm - hx:cxm + hx]
        elif side == "left":
            into = content[cym - hy:cym + hy, int(bx):int(bx) + d]
            away = content[cym - hy:cym + hy, int(bx) - d:int(bx)]
        else:
            into = content[cym - hy:cym + hy, int(bx + pw) - d:int(bx + pw)]
            away = content[cym - hy:cym + hy, int(bx + pw):int(bx + pw) + d]
        if into.size == 0 or away.size == 0:
            return 0.0, 0.0
        return float(into.mean()), float(away.mean())

    edges = {}
    for side in ("top", "right", "bottom", "left"):
        into, away = probe(side)
        if into > 0.30:
            edges[side] = 1        # 邻居凸进本格 -> 本格要凹槽
        elif away < 0.30:
            edges[side] = -1       # 邻居侧被挖空 -> 邻居是凹槽 -> 本格要凸齿
        else:
            edges[side] = 0
    return edges

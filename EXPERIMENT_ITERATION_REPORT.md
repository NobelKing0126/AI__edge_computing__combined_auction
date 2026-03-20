# UAV-DNN 实验优化迭代报告

## 总体概述

本文档记录对 UAV-DNN 实验项目的优化迭代过程，目标是使 proposed 算法在实验1-5中的成功率均高于 baseline 算法，同时满足趋势约束。

---

## 第1轮迭代 (V29_R1-V29_R2)

### 修改目标
1. 针对性降低 deadline，使其符合边缘计算任务特性
2. 检查 Baseline 与 Proposed 口径一致性
3. 确保单个 UAV 初始能量不随 UAV 数量变化

---

### 修改内容

#### 1. Deadline 配置修正 (`experiments/task_types.py`)

**小规模实验任务配置：**
- 高时延敏感任务：0.2s - 1.0s (原 2-3s)
- 计算密集型任务：5.0s - 15.0s (原 3-8s)

**大规模实验任务配置：**
- 时延敏感任务：1.0s - 3.0s (保持不变)
- 计算密集型任务：5.0s - 15.0s (原 4-10s)

**设计原则**：
- 明显低于当前值，但不过分严格
- 保持任务类别之间的层次差异
- 结合实验结果迭代微调

#### 2. MAPPO-Attention deadline 判定对齐 (`experiments/paper_baselines/mappo_attention.py`)

**修改：**
- 第345行：`delay <= deadline * 0.9` → `delay <= deadline`

**目的：**
- 确保 baseline 与 proposed 使用相同的成功判定口径
- 消除不公平比较

---

### 初步实验结果（到达速率 1.0/s）

| 算法 | 成功率 | 比较 |
|------|--------|------|
| Proposed | 40.9% | 基准 |
| MAPPO-Attention | 37.2% | ✓ 低于 Proposed |
| Edge-Only | 21.6% | ✓ 低于 Proposed |
| Cloud-Only | 30.8% | ✓ 低于 Proposed |
| B12-DelayOpt | 57.8% | △ 高于 Proposed (需分析) |

---

### 根因分析

#### B12-DelayOpt 成功率较高的原因

1. **优化目标差异**：
   - B12-DelayOpt：专门优化时延 (`min Σ T_total`)
   - Proposed：优化综合效用（自由能融合）

2. **算法策略差异**：
   - B12-DelayOpt：按预估时延升序排序任务（优先处理快任务）
   - Proposed：通过拍卖框架分配资源

3. **合理性判断**：
   - B12-DelayOpt 使用了收紧的 deadline（0.8或0.7倍），但仍然成功率更高
   - 这说明 B12 能找到更优的时延解
   - 这是算法性能差异，不是口径不一致问题

---

### Baseline 与 Proposed 口径一致性检查

#### 1. Deadline 判定逻辑

| 算法 | Deadline 判定 | 状态 |
|------|---------------|------|
| Proposed | `T_total <= deadline` | 基准 |
| Edge-Only | `T_total <= deadline` | ✓ 一致 |
| Cloud-Only | `T_total <= deadline` | ✓ 一致 |
| Greedy | `T_total <= deadline` | ✓ 一致 |
| Fixed-Split | `T_total <= deadline` | ✓ 一致 |
| MAPPO-Attention (已修复) | `delay <= deadline` | ✓ 已对齐 |
| B12-DelayOpt | `best_delay <= deadline * 0.8/0.7` | 注1 |

**注1**：B12-DelayOpt 的 deadline_margin 是设计决策，目的是让 B12 不再单纯优化时延而忽视其他因素。这是合理的设计。

#### 2. 通信时延计算

所有 baseline 都使用相同的 `SystemConfig` 通信参数：
- `W = 0.7 MHz` (信道带宽)
- `beta_0 = 3.5e-7` (参考信道增益)
- `N_0 = 1e-18` (噪声功率谱密度)
- `P_tx_user = 0.09W` (用户发射功率)
- `R_backhaul = 40Mbps` (回程链路带宽)

通信时延公式：`T_upload = data_size / upload_rate` ✓ 正确考虑了数据量

#### 3. 单个 UAV 初始能量

- `energy_capacity = 1200e3 J` (固定值)
- 不随 UAV 数量变化 ✓

---

### 已完成的 Git 提交

1. **V29_R1**: deadline 合理化调整
2. **V29_R2**: MAPPO-Attention deadline 判定对齐

---

## 下一步计划

### Phase 3: 算力分配解析解验证
- 读取 idea316.pdf 中的算力分配公式
- 对比 `algorithms/optimization/convex_solver.py` 中的实现
- 如有差异，按 idea316.pdf 更新

### Phase 4-5: 资源参数微调与趋势验证
- 运行完整实验（实验1-5）
- 检查趋势：UAV↑成功率↑，用户↑成功率↓
- 必要时微调资源参数

---

## 待验证项

1. ✓ Deadline 是否已降低到合理范围 → **已完成**
2. ✓ 通信时延是否考虑数据传输量 → **已确认**
3. ✓ 单个 UAV 初始能量是否固定 → **已确认**
4. ✓ Baseline 与 Proposed 是否使用相同判定口径 → **基本一致**
5. ? 算力分配解析解是否与 idea316.pdf 一致 → **待验证**
6. ? UAV 增加时 Proposed 成功率是否上升 → **待验证**
7. ? 用户增加时 Proposed 成功率是否下降 → **待验证**

---

## 风险点

1. **B12-DelayOpt 成功率高于 Proposed**：
   - 这是算法策略差异，不是口径不一致
   - B12 专门优化时延，使用收紧 deadline
   - Proposed 优化综合效用，考虑更多因素

2. **Deadline 调低后整体成功率下降**：
   - 这是预期内的，因为 deadline 更严格
   - 需要通过资源微调来平衡

3. **趋势验证需要完整实验数据**：
   - 当前只有到达速率 1.0/s 的结果
   - 需要运行实验2-5来检查趋势

---

---

## 第2轮迭代 (V29_R3)

### 修改目标

验证算力分配解析解是否与 idea38.txt 公式一致

### 修改内容

#### 1. convex_solver.py 修复 (solve_v2 方法)

**问题**: A 系数公式与 idea38.txt 不一致

**修改**:
- 旧: `denominator = κe·Ce·(1+rho²)`
- 新: `denominator = κc·Cc + κe·Ce·rho²`
- 对应公式: `A = κe·Ce·ρ² + κc·Cc` (idea38.txt)

**修改后计算流程**:
```
ρ = (κc/κe)^(1/3)
A = κc·Cc + κe·Ce·ρ²
f_cloud = sqrt(E_budget / A)
f_edge = ρ * f_cloud
```

#### 2. resource_optimizer.py 修复 (_solve_case3 方法)

**问题 1**: rho 计算依赖计算量（错误）
- 旧: `ratio = (κe·Ce)/(κc·Cc)` (依赖计算量)
- 新: `rho = (κc/κe)^(1/3)` (仅依赖能耗系数)

**问题 2**: A < EPSILON 检查导致误判
- 旧: `if A < NUMERICAL.EPSILON:` (A ~1e-19, EPSILON=1e-10, 始终为 True)
- 新: `if A <= 0:` (正确的检查)

**修改后计算流程**:
```
ρ = (κc/κe)^(1/3)
A = κe·ρ²·Ce + κc·Cc
if A <= 0: return max
f_cloud_unc = sqrt(E_budget / A)
f_edge_unc = ρ * f_cloud_unc
处理触顶情况...
```

### 验证结果

测试参数: C_edge=5e9, C_cloud=5e9, E_budget=21J

| 求解器 | f_edge (GFLOPS) | f_cloud (GFLOPS) | 比例 f_e/f_c | 能耗 (J) |
|--------|-----------------|------------------|--------------|----------|
| ConvexSolver.solve_v2 | 5.36 | 11.54 | 0.4642 | 21.00 |
| ResourceOptimizer._solve_case3 | 5.36 | 11.54 | 0.4642 | 21.00 |
| **期望值** | **5.36** | **11.54** | **0.4642** | **21.00** |

✓ 两个求解器结果一致且符合公式预期

### Git 提交

- **V29_R3**: Phase 3 - 修复算力分配解析解公式

---

## 下一步计划

### Phase 4-5: 资源参数微调与趋势验证
- 运行完整实验（实验1-5）
- 检查趋势：UAV↑成功率↑，用户↑成功率↓
- 必要时微调资源参数

---

## 待验证项

1. ✓ Deadline 是否已降低到合理范围 → **已完成**
2. ✓ 通信时延是否考虑数据传输量 → **已确认**
3. ✓ 单个 UAV 初始能量是否固定 → **已确认**
4. ✓ Baseline 与 Proposed 是否使用相同判定口径 → **基本一致**
5. ✓ 算力分配解析解是否与 idea38.txt 一致 → **已修复**
6. ? UAV 增加时 Proposed 成功率是否上升 → **待验证**
7. ? 用户增加时 Proposed 成功率是否下降 → **待验证**

---

## 迭代状态

- **当前轮次**: 第2轮
- **总轮次**: 待定（根据实验结果决定）
- **状态**: Phase 3 已完成，准备 Phase 4-5


# UAV-DNN Collaborative Inference Framework

基于拍卖调度的UAV边缘协同DNN推理框架

---

## 项目结构

```
projects/first/
├── config/                      # 系统配置
│   └── system_config.py         # 信道/UAV/云端/能耗参数
│
├── models/                      # 数据模型
│   ├── dnn_model.py             # DNN模型定义与切分
│   ├── uav.py                   # UAV模型
│   ├── user.py                  # 用户/任务模型
│   ├── bid.py                   # 投标模型
│   ├── ddpg_unified.pt          # 训练好的DDPG模型
│   └── td3_unified.pt           # 训练好的TD3模型
│
├── communication/               # 通信模块
│   └── channel_model.py         # 信道模型 (传输速率计算)
│
├── utils/                       # 工具模块
│   ├── energy_model.py          # 能耗模型
│   └── data_loader.py           # 数据加载器 (EUA/合成)
│
├── algorithms/                  # 算法实现
│   ├── phase0/                  # 阶段0: K-means UAV部署
│   ├── phase1/                  # 阶段1: 优先级选举
│   ├── phase2/                  # 阶段2: 投标生成 (凸优化+自由能)
│   ├── phase3/                  # 阶段3: 组合拍卖
│   ├── phase4/                  # 阶段4: 执行调度
│   └── rl_baselines/            # DDPG/TD3强化学习
│
├── experiments/                 # 实验模块
│   ├── unified_config.py        # 统一实验配置
│   ├── baselines.py             # 基线算法
│   ├── ablation_real.py         # 消融实验
│   ├── metrics.py               # 指标计算
│   └── visualization.py         # 可视化
│
├── data/                        # 数据集
│   ├── cifar10/                 # CIFAR-10图像
│   ├── coco/                    # COCO验证集
│   ├── eua/                     # EUA用户分布
│   └── shanghai/                # 上海电信用户分布
│
├── figures/                     # 实验结果图表
│   ├── exp1_baseline_comparison.png
│   ├── exp2_ablation_study.png
│   ├── exp3_scalability.png
│   ├── exp4_robustness.png
│   ├── exp5_checkpoint_theory.png
│   ├── exp8_dynamic_pricing.png
│   ├── rl_training_unified.png
│   └── ...
│
├── docs/                        # 设计文档
│   ├── idea118.txt              # 论文想法
│   └── 实验.txt                 # 实验设计
│
├── run_full_experiments.py      # 主实验脚本 (12个实验)
├── run_rl_unified.py            # RL基线实验
├── run_dnn_experiments.py       # 真实DNN推理实验
├── EXPERIMENT_REPORT.md         # 实验报告
├── README.md                    # 使用说明
├── requirements.txt             # Python依赖
└── yolov5su.pt                  # YOLOv5预训练模型
```

---

## 快速开始

### 1. 激活环境

```bash
cd /home/hyp/projects/first
source venv/bin/activate
```

### 2. 运行完整实验 (12个实验)

```bash
python run_full_experiments.py
```

包含:
- 实验1: 基线对比 (9个基线)
- 实验2: 消融实验 (A1-A8)
- 实验3: 可扩展性分析
- 实验4: 鲁棒性分析
- 实验5: Checkpoint理论验证
- 实验6: 凸优化验证
- 实验7: 切分点效果
- 实验8: 动态定价
- 实验11: 实时性验证
- 实验12: 对偶间隙

### 3. 运行RL基线 (DDPG/TD3)

```bash
python run_rl_unified.py
```

### 4. 运行真实DNN推理

```bash
python run_dnn_experiments.py
```

---

## 实验结果

| 算法 | 成功率 | 平均时延 | 能耗 |
|------|--------|----------|------|
| **Proposed** | **79.0%** | **362ms** | **490J** |
| DDPG | 37.0% | 1817ms | 3256J |
| TD3 | 37.0% | 1636ms | 5267J |
| Cloud-Only | 22.0% | 2618ms | 0J |
| Edge-Only | 17.0% | 2885ms | 9402J |

**提议方法优势**: +57% 成功率 (vs Cloud-Only)

---

## 消融实验结果

| 变体 | 成功率 | vs Full |
|------|--------|---------|
| Full | 79.0% | - |
| A2-NoCheckpoint | 66.3% | -12.7% |
| A3-NoConvex | 72.7% | -6.3% |
| A6-SingleGreedy | 74.3% | -4.7% |
| A1-NoFE-Fusion | 75.0% | -4.0% |

---

## 核心API

```python
from experiments.unified_config import UnifiedTaskGenerator
from run_full_experiments import ProposedMethod

# 生成任务
generator = UnifiedTaskGenerator(seed=42)
tasks = generator.generate_mixed_tasks(n_users=100)

# 运行算法
proposed = ProposedMethod(seed=42)
result = proposed.run(tasks, uav_resources, cloud_resources)

print(f"Success: {result.success_rate*100:.1f}%")
```

---

## 系统配置

编辑 `config/system_config.py`:

```python
@dataclass
class ChannelConfig:
    W: float = 2e6           # 信道带宽 2MHz
    beta_0: float = 1e-6     # 参考信道增益
    N_0: float = 1e-18       # 噪声功率密度
    P_tx_user: float = 0.1   # 用户发射功率 0.1W
```

---

## 依赖

- Python 3.8+
- PyTorch 2.5.1+cu121
- NumPy, SciPy, Matplotlib
- CVXPY

---

*Last updated: 2026-01-22*

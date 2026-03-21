"""
场景配置模块

定义小规模和大规模实验场景的所有参数

小规模场景：
- 区域：200m × 200m
- UAV数：3-8个
- 用户数：10-50个
- 特点：完整指标 + 竞争比

大规模场景：
- 区域：1000m × 1000m
- UAV数：10-20个
- 用户数：50-200个
- 特点：核心指标（无竞争比）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.system_config import SystemConfig

# 全局配置实例，所有场景共用
_system_config = SystemConfig()


# ============ 规模特定参数 (根据experiment_params.py设计) ============
# 设计目标：
# - 用户数增加 → 成功率下降
# - UAV数增加 → 成功率上升
# 参见 config/experiment_params.py 中的详细分析

# 小规模实验云端配置 (200m x 200m) - V30: 提升成功率~20%
# 调整：增加云端算力和并发数
_SMALL_SCALE_CLOUD = {
    'F_c': 20.0e9,            # 云端算力 20.0 GFLOPS (V30: 从14→20 GFLOPS)
    'F_per_task_max': 5.0e9,  # 单任务最大 5.0 GFLOPS (V30: 从3.5→5.0 GFLOPS)
    'T_propagation': 0.02,    # 传播延迟 20ms
    'max_concurrent_tasks': 20, # 最大并发 20 个任务 (V30: 从14→20)
}

# 小规模实验任务配置 - V22: 极大幅放宽deadline，极大幅减少图像数
_SMALL_SCALE_TASKS = {
    'latency_sensitive': {
        'min_images': 1,      # 最少图像数 (V22: 减少)
        'max_images': 4,      # 最多图像数 (V22: 减少)
        'min_deadline': 40.0, # 最小时延上限 40s (V22: 放宽)
        'max_deadline': 80.0, # 最大时延上限 80s (V22: 放宽)
    },
    'compute_intensive': {
        'min_images': 3,      # 最少图像数 (V22: 减少)
        'max_images': 8,      # 最多图像数 (V22: 减少)
        'min_deadline': 100.0, # 最小时延上限 100s (V22: 放宽)
        'max_deadline': 200.0, # 最大时延上限 200s (V22: 放宽)
    }
}

# 大规模实验云端配置 (1000m x 1000m) - V29: 提升成功率~30%
_LARGE_SCALE_CLOUD = {
    'F_c': 20.0e9,            # 云端算力 20.0 GFLOPS (V29: 从15→20 GFLOPS，提升33%)
    'F_per_task_max': 5.0e9,     # 单任务最大 5.0 GFLOPS (V29: 从4→5 GFLOPS)
    'T_propagation': 0.02,    # 传播延迟 20ms
    'max_concurrent_tasks': 20,  # 最大并发 20 个任务 (V29: 从15→20)
}

# 大规模实验任务配置 - V22: 极大幅放宽deadline，极大幅减少图像数
_LARGE_SCALE_TASKS = {
    'latency_sensitive': {
        'min_images': 1,      # 最少图像数 (V22: 减少)
        'max_images': 5,      # 最多图像数 (V22: 减少)
        'min_deadline': 50.0, # 最小时延上限 50s (V22: 放宽)
        'max_deadline': 100.0, # 最大时延上限 100s (V22: 放宽)
    },
    'compute_intensive': {
        'min_images': 4,      # 最少图像数 (V22: 减少)
        'max_images': 10,     # 最多图像数 (V22: 减少)
        'min_deadline': 120.0, # 最小时延上限 120s (V22: 放宽)
        'max_deadline': 240.0, # 最大时延上限 240s (V22: 放宽)
    }
}


class ScenarioType(Enum):
    """场景类型"""
    SMALL_SCALE = "small_scale"
    LARGE_SCALE = "large_scale"


@dataclass
class UAVConfig:
    """UAV配置"""
    n_uavs: int
    compute_capacity: float      # 计算能力 (FLOPS)
    energy_capacity: float       # 能量容量 (J)
    height: float                # 飞行高度 (m)
    cover_radius: float          # 覆盖半径 (m)
    hover_power: float           # 悬停功率 (W)
    compute_power_coeff: float   # 计算功率系数
    
    def generate_positions(self, area_size: float, seed: int = None) -> np.ndarray:
        """
        生成UAV位置（网格部署）
        
        Args:
            area_size: 区域大小
            seed: 随机种子
            
        Returns:
            位置数组 (n_uavs, 2)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 网格部署
        grid_size = int(np.ceil(np.sqrt(self.n_uavs)))
        spacing = area_size / (grid_size + 1)
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= self.n_uavs:
                    break
                x = spacing * (i + 1) + np.random.uniform(-spacing*0.1, spacing*0.1)
                y = spacing * (j + 1) + np.random.uniform(-spacing*0.1, spacing*0.1)
                positions.append([x, y])
        
        return np.array(positions[:self.n_uavs])


@dataclass
class CloudConfig:
    """云端配置"""
    compute_capacity: float      # 计算能力 (FLOPS)
    transmission_rate: float     # 边缘-云传输速率 (bps)
    compute_price: float         # 计算价格 (元/GFLOPS·s)
    # 规模特定参数
    F_per_task_max: float = 5e9    # 单任务最大分配算力
    T_propagation: float = 0.20    # 单向传播延迟 (s)
    max_concurrent_tasks: int = 5  # 最大并发任务数


@dataclass
class ChannelConfig:
    """信道配置"""
    bandwidth: float             # 带宽 (Hz)
    num_channels: int            # 子信道数
    noise_power: float           # 噪声功率 (W)
    path_loss_exp: float         # 路径损耗指数
    reference_gain: float        # 参考信道增益
    tx_power_min: float          # 最小发射功率 (W)
    tx_power_max: float          # 最大发射功率 (W)


@dataclass
class ScenarioConfig:
    """
    完整场景配置
    
    包含所有实验参数
    """
    name: str
    scenario_type: ScenarioType
    
    # 区域参数
    area_size: float             # 区域边长 (m)
    
    # 设备配置
    uav_config: UAVConfig
    cloud_config: CloudConfig
    channel_config: ChannelConfig
    
    # 用户参数
    n_users: int
    latency_ratio: float         # 延迟敏感型任务比例
    
    # 实验参数
    compute_competitive_ratio: bool  # 是否计算竞争比
    compute_robustness: bool         # 是否计算鲁棒性指标
    compute_efficiency: bool         # 是否计算算法效率指标
    
    # 有默认值的参数（必须放在最后）
    tasks_per_user: int = 10     # 每个用户提交的任务数量 (V27: 从2改为10，增加任务密度，目标成功率60-90%)
    fault_probability: float = 0.05  # 故障参数（仅小规模）
    seed: int = 42               # 随机种子
    
    def get_uav_resources(self) -> List[Dict]:
        """
        获取UAV资源列表（兼容现有代码格式）
        """
        positions = self.uav_config.generate_positions(self.area_size, self.seed)
        
        resources = []
        for i in range(self.uav_config.n_uavs):
            resources.append({
                'uav_id': i,
                'x': positions[i][0],
                'y': positions[i][1],
                'z': self.uav_config.height,
                'f_max': self.uav_config.compute_capacity,
                'E_max': self.uav_config.energy_capacity,
                'E_current': self.uav_config.energy_capacity,
                'R_cover': self.uav_config.cover_radius,
                'P_hover': self.uav_config.hover_power,
                'compute_power_coeff': self.uav_config.compute_power_coeff,
                'price': 1.0,  # 初始价格
                'utilization': 0.0
            })
        
        return resources
    
    def get_cloud_resources(self) -> Dict:
        """
        获取云端资源（兼容现有代码格式）

        包含规模特定的云端参数
        """
        return {
            'f_cloud': self.cloud_config.compute_capacity,
            'rate_edge_cloud': self.cloud_config.transmission_rate,
            'price': self.cloud_config.compute_price,
            # 规模特定参数
            'F_per_task_max': self.cloud_config.F_per_task_max,
            'T_propagation': self.cloud_config.T_propagation,
            'max_concurrent_tasks': self.cloud_config.max_concurrent_tasks
        }
    
    def get_channel_params(self) -> Dict:
        """
        获取信道参数（兼容现有代码格式）
        """
        return {
            'bandwidth': self.channel_config.bandwidth,
            'num_channels': self.channel_config.num_channels,
            'noise_power': self.channel_config.noise_power,
            'path_loss_exp': self.channel_config.path_loss_exp,
            'reference_gain': self.channel_config.reference_gain,
            'tx_power_range': (self.channel_config.tx_power_min, 
                               self.channel_config.tx_power_max)
        }


# ============ 预定义场景 ============

def create_small_scale_config(n_uavs: int = 5, n_users: int = 30,
                              tasks_per_user: int = 10) -> ScenarioConfig:  # V27: 默认10任务/用户
    """
    创建小规模场景配置

    使用专门为小规模实验优化的参数：
    - 更紧的资源约束以放大用户/UAV扩展趋势
    - 云端算力较小，并发数较少
    - 任务deadline更严格

    Args:
        n_uavs: UAV数量 (默认5)
        n_users: 用户数量 (默认30)
        tasks_per_user: 每个用户提交的任务数量 (默认5)

    Returns:
        ScenarioConfig
    """
    return ScenarioConfig(
        name=f"小规模场景 ({n_uavs}UAV, {n_users}用户, {n_users*tasks_per_user}任务)",
        scenario_type=ScenarioType.SMALL_SCALE,

        # 区域：200m × 200m
        area_size=200.0,

        # UAV配置 - V29: 提升成功率~20%，适中的单机算力保持趋势
        # 小规模场景：200m x 200m
        uav_config=UAVConfig(
            n_uavs=n_uavs,
            compute_capacity=14.0e9,              # 14.0 GFLOPS (V29: 从12→14 GFLOPS，提升成功率)
            energy_capacity=1400e3,              # 电池容量 1400kJ (V29: 从1200→1400)
            height=80.0,                          # 飞行高度 80m
            cover_radius=380.0,                   # 覆盖半径 380m (V29: 从350→380m)
            hover_power=_system_config.uav.P_hover,
            compute_power_coeff=_system_config.energy.kappa_edge
        ),

        # 云端配置 - 使用小规模专用参数
        cloud_config=CloudConfig(
            compute_capacity=_SMALL_SCALE_CLOUD['F_c'],
            transmission_rate=_system_config.channel.R_backhaul,
            compute_price=0.01,
            F_per_task_max=_SMALL_SCALE_CLOUD['F_per_task_max'],
            T_propagation=_SMALL_SCALE_CLOUD['T_propagation'],
            max_concurrent_tasks=_SMALL_SCALE_CLOUD['max_concurrent_tasks']
        ),

        # 信道配置 - 使用SystemConfig
        channel_config=ChannelConfig(
            bandwidth=_system_config.channel.W,                 # 使用SystemConfig
            num_channels=_system_config.channel.num_channels,   # 使用SystemConfig
            noise_power=_system_config.channel.N_0 * _system_config.channel.W,  # 噪声功率
            path_loss_exp=4.0,
            reference_gain=_system_config.channel.beta_0,       # 使用SystemConfig
            tx_power_min=_system_config.channel.P_tx_user * 0.8,
            tx_power_max=_system_config.channel.P_tx_user * 1.2
        ),

        # 用户参数
        n_users=n_users,
        latency_ratio=0.5,               # 50%延迟敏感型
        tasks_per_user=tasks_per_user,   # 每用户任务数

        # 小规模场景：完整指标
        compute_competitive_ratio=True,
        compute_robustness=True,
        compute_efficiency=True,

        fault_probability=0.05,
        seed=42
    )


def create_large_scale_config(n_uavs: int = 15, n_users: int = 100,
                              tasks_per_user: int = 10) -> ScenarioConfig:  # V27: 默认10任务/用户
    """
    创建大规模场景配置

    使用专门为大规模实验优化的参数：
    - 收紧资源约束以确保趋势明显
    - 增加云端传播延迟以增加云端代价

    Args:
        n_uavs: UAV数量 (默认15)
        n_users: 用户数量 (默认100)
        tasks_per_user: 每个用户提交的任务数量 (默认5)

    Returns:
        ScenarioConfig
    """
    # 大规模实验使用不同的UAV算力 (V29: 提升成功率~30%)
    large_scale_f_max = 25.0e9  # 25.0 GFLOPS (V29: 从20→25 GFLOPS，提升25%)

    return ScenarioConfig(
        name=f"大规模场景 ({n_uavs}UAV, {n_users}用户, {n_users*tasks_per_user}任务)",
        scenario_type=ScenarioType.LARGE_SCALE,

        # 区域：1000m × 1000m
        area_size=1000.0,

        # UAV配置 - V29: 提升成功率~30%
        uav_config=UAVConfig(
            n_uavs=n_uavs,
            compute_capacity=large_scale_f_max,         # 25.0 GFLOPS (V29: 提升算力)
            energy_capacity=1600e3,                      # 电池容量 1600kJ (V29: 从1200→1600)
            height=100.0,                              # 飞行高度 100m
            cover_radius=580.0,                        # 覆盖半径 580m (V29: 从550→580m)
            hover_power=_system_config.uav.P_hover,
            compute_power_coeff=_system_config.energy.kappa_edge
        ),

        # 云端配置 - 使用大规模专用参数
        cloud_config=CloudConfig(
            compute_capacity=_LARGE_SCALE_CLOUD['F_c'],
            transmission_rate=_system_config.channel.R_backhaul,
            compute_price=0.01,
            F_per_task_max=_LARGE_SCALE_CLOUD['F_per_task_max'],
            T_propagation=_LARGE_SCALE_CLOUD['T_propagation'],
            max_concurrent_tasks=_LARGE_SCALE_CLOUD['max_concurrent_tasks']
        ),

        # 信道配置 - 使用SystemConfig (更多子信道)
        channel_config=ChannelConfig(
            bandwidth=_system_config.channel.W,                 # 使用SystemConfig
            num_channels=_system_config.channel.num_channels * 2,  # 大规模场景加倍
            noise_power=_system_config.channel.N_0 * _system_config.channel.W,
            path_loss_exp=4.0,
            reference_gain=_system_config.channel.beta_0,       # 使用SystemConfig
            tx_power_min=_system_config.channel.P_tx_user * 0.8,
            tx_power_max=_system_config.channel.P_tx_user * 1.2
        ),

        # 用户参数
        n_users=n_users,
        latency_ratio=0.5,
        tasks_per_user=tasks_per_user,   # 每用户任务数

        # 大规模场景：简化指标（无竞争比、鲁棒性、效率细节）
        compute_competitive_ratio=False,
        compute_robustness=False,
        compute_efficiency=False,

        fault_probability=0.0,  # 不模拟故障
        seed=42
    )


# ============ 实验配置 ============

@dataclass
class ExperimentConfig:
    """实验配置"""
    exp_id: int
    name: str
    description: str
    scenario_type: ScenarioType
    
    # 变量设置
    fixed_param: str             # 固定参数名 ('uav' 或 'user')
    fixed_value: int             # 固定参数值
    variable_param: str          # 变化参数名
    variable_values: List[int]   # 变化参数值列表
    
    # 对比算法
    baseline_algorithms: List[str]
    
    # 是否计算竞争比
    compute_competitive_ratio: bool


# 实验1：小规模基线对比
EXP1_CONFIG = ExperimentConfig(
    exp_id=1,
    name="小规模基线对比",
    description="200m×200m, 5 UAV, 30用户, 全指标对比",
    scenario_type=ScenarioType.SMALL_SCALE,
    fixed_param="both",
    fixed_value=0,
    variable_param="none",
    variable_values=[],
    baseline_algorithms=[
        "Proposed", "Edge-Only", "Cloud-Only", "Greedy", 
        "Fixed-Split", "Random-Auction", "No-ActiveInference",
        "Heuristic-Alloc", "No-DynPricing", "B11-FixedPrice",
        "B11a-HighFixed", "B11b-LowFixed", "B12-DelayOpt"
    ],
    compute_competitive_ratio=True
)

# 实验2：小规模-固定UAV变用户数 (V26: 均匀分布用户范围)
# 与 run_real_experiments_v9.py 中的 run_exp2() 保持一致
EXP2_CONFIG = ExperimentConfig(
    exp_id=2,
    name="小规模用户扩展",
    description="200m×200m, 固定5 UAV, 用户数{10,20,30,40,50}",
    scenario_type=ScenarioType.SMALL_SCALE,
    fixed_param="uav",
    fixed_value=5,
    variable_param="user",
    variable_values=[10, 20, 30, 40, 50],
    baseline_algorithms=["Proposed", "Greedy", "Edge-Only", "Cloud-Only", "B12-DelayOpt", "MAPPO-Attention"],
    compute_competitive_ratio=True
)

# 实验3：小规模-固定用户变UAV数
# 与 run_real_experiments_v9.py 中的 run_exp3() 保持一致
EXP3_CONFIG = ExperimentConfig(
    exp_id=3,
    name="小规模UAV扩展",
    description="200m×200m, 固定50用户, UAV数{3,4,5,6,7,8}",
    scenario_type=ScenarioType.SMALL_SCALE,
    fixed_param="user",
    fixed_value=50,
    variable_param="uav",
    variable_values=[3, 4, 5, 6, 7, 8],
    baseline_algorithms=["Proposed", "Greedy", "Edge-Only", "Cloud-Only", "B12-DelayOpt", "MAPPO-Attention"],
    compute_competitive_ratio=True
)

# 实验4：大规模-固定UAV变用户数 (V26: 减少到70)
EXP4_CONFIG = ExperimentConfig(
    exp_id=4,
    name="大规模用户扩展",
    description="1000m×1000m, 固定15 UAV, 用户数{30,50,70}",
    scenario_type=ScenarioType.LARGE_SCALE,
    fixed_param="uav",
    fixed_value=15,
    variable_param="user",
    variable_values=[30, 50, 70],
    baseline_algorithms=["Proposed", "Greedy", "Edge-Only", "Cloud-Only"],
    compute_competitive_ratio=False
)

# 实验5：大规模-固定用户变UAV数 (V26: 固定35用户, UAV从10开始)
EXP5_CONFIG = ExperimentConfig(
    exp_id=5,
    name="大规模UAV扩展",
    description="1000m×1000m, 固定35用户, UAV数{10,12,14,16}",
    scenario_type=ScenarioType.LARGE_SCALE,
    fixed_param="user",
    fixed_value=35,
    variable_param="uav",
    variable_values=[10, 12, 14, 16],
    baseline_algorithms=["Proposed", "Greedy", "Edge-Only", "Cloud-Only"],
    compute_competitive_ratio=False
)

# 所有实验配置
ALL_EXPERIMENTS = [EXP1_CONFIG, EXP2_CONFIG, EXP3_CONFIG, EXP4_CONFIG, EXP5_CONFIG]


def get_scenario_for_experiment(exp_config: ExperimentConfig, 
                                 variable_value: int = None) -> ScenarioConfig:
    """
    根据实验配置获取场景配置
    
    Args:
        exp_config: 实验配置
        variable_value: 变量值（如果有）
        
    Returns:
        ScenarioConfig
    """
    if exp_config.scenario_type == ScenarioType.SMALL_SCALE:
        if exp_config.fixed_param == "uav":
            n_uavs = exp_config.fixed_value
            n_users = variable_value if variable_value else 30
        elif exp_config.fixed_param == "user":
            n_users = exp_config.fixed_value
            n_uavs = variable_value if variable_value else 5
        else:  # both fixed
            n_uavs = 5
            n_users = 30
        
        return create_small_scale_config(n_uavs, n_users)
    
    else:  # LARGE_SCALE
        if exp_config.fixed_param == "uav":
            n_uavs = exp_config.fixed_value
            n_users = variable_value if variable_value else 100
        elif exp_config.fixed_param == "user":
            n_users = exp_config.fixed_value
            n_uavs = variable_value if variable_value else 15
        else:
            n_uavs = 15
            n_users = 100
        
        return create_large_scale_config(n_uavs, n_users)


# ============ 测试 ============

def test_scenario_config():
    """测试场景配置模块"""
    print("=" * 60)
    print("测试 场景配置模块")
    print("=" * 60)
    
    # 测试小规模配置
    print("\n[Test 1] 小规模场景配置")
    small_config = create_small_scale_config(5, 30)
    print(f"  名称: {small_config.name}")
    print(f"  区域: {small_config.area_size}m × {small_config.area_size}m")
    print(f"  UAV: {small_config.uav_config.n_uavs}个, "
          f"算力{small_config.uav_config.compute_capacity/1e9:.0f}GFLOPS")
    print(f"  覆盖半径: {small_config.uav_config.cover_radius}m")
    print(f"  计算竞争比: {small_config.compute_competitive_ratio}")
    
    # 测试大规模配置
    print("\n[Test 2] 大规模场景配置")
    large_config = create_large_scale_config(15, 100)
    print(f"  名称: {large_config.name}")
    print(f"  区域: {large_config.area_size}m × {large_config.area_size}m")
    print(f"  UAV: {large_config.uav_config.n_uavs}个")
    print(f"  计算竞争比: {large_config.compute_competitive_ratio}")
    
    # 测试UAV资源生成
    print("\n[Test 3] UAV资源生成")
    uav_resources = small_config.get_uav_resources()
    print(f"  生成 {len(uav_resources)} 个UAV资源")
    for uav in uav_resources[:3]:
        print(f"    UAV-{uav['uav_id']}: 位置({uav['x']:.1f}, {uav['y']:.1f}), "
              f"算力{uav['f_max']/1e9:.0f}G")
    
    # 测试实验配置
    print("\n[Test 4] 实验配置")
    for exp in ALL_EXPERIMENTS:
        print(f"  实验{exp.exp_id}: {exp.name}")
        print(f"    - {exp.description}")
        print(f"    - 竞争比: {exp.compute_competitive_ratio}")
    
    # 测试场景获取
    print("\n[Test 5] 根据实验获取场景")
    for val in EXP2_CONFIG.variable_values[:3]:
        scenario = get_scenario_for_experiment(EXP2_CONFIG, val)
        print(f"  用户数={val}: {scenario.uav_config.n_uavs}UAV, "
              f"{scenario.n_users}用户")
    
    print("\n" + "=" * 60)
    print("场景配置模块测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_scenario_config()

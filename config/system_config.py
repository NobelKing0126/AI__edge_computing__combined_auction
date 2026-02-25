"""
M01: SystemConfig - 系统参数配置模块

功能：统一管理UAV辅助边缘计算DNN协同推理系统的所有参数
输入：可选的YAML配置文件路径
输出：包含所有参数的配置对象

关键参数来源：idea118.txt 0.3节 系统参数表
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
import os


@dataclass
class UAVConfig:
    """
    UAV（无人机）相关参数配置
    
    Attributes:
        H: 飞行高度 (m)，固定值，简化为二维部署问题
        R_cover: 覆盖半径 (m)，UAV地面覆盖范围，只有在覆盖范围内的用户才能被服务
        f_max: 最大算力 (FLOPS)，同构UAV
        E_max: 电池容量 (J)，同构UAV
        P_hover: 悬停功率 (W)
        P_fly: 飞行功率 (W)
        v_fly: 飞行速度 (m/s)
        P_rx: 接收功率 (W)
        P_tx: 发射功率 (W)
        M_max: 最大模型加载数量
    """
    H: float = 100.0  # 飞行高度 (m)
    R_cover: float = 500.0  # 覆盖半径 (m)，基于场景2km×2km和5个UAV计算
    f_max: float = 15e9  # 最大算力 15 GFLOPS = 15e9 FLOPS (优化: 从10GHz提升)
    E_max: float = 500e3  # 电池容量 500kJ = 500e3 J
    P_hover: float = 150.0  # 悬停功率 (W)
    P_fly: float = 200.0  # 飞行功率 (W)
    v_fly: float = 10.0  # 飞行速度 (m/s)
    P_rx: float = 0.1  # 接收功率 (W)
    P_tx: float = 5.0  # 发射功率 (W)
    M_max: int = 5  # 最大模型加载数量


@dataclass
class ChannelConfig:
    """
    通信信道相关参数配置
    
    公式参考 (idea118.txt 0.8节):
        信道增益: h_{i,j} = β₀ / d_{i,j}²
        传输速率: R_{i,j} = W * log₂(1 + P_tx * h_{i,j} / (N₀ * W))
    
    Attributes:
        W: 信道带宽 (Hz)
        beta_0: 参考信道增益 (1m处)
        N_0: 噪声功率谱密度 (W/Hz)
        P_tx_user: 用户发射功率 (W)
        R_backhaul: 回程链路带宽 (bps)
        num_channels: 可用信道数量
    """
    W: float = 2e6  # 信道带宽 2MHz (降低以增加挑战性)
    beta_0: float = 1e-6  # 参考信道增益 -60dB (降低以增加挑战性)
    N_0: float = 1e-18  # 噪声功率谱密度 (W/Hz)
    P_tx_user: float = 0.1  # 用户发射功率 0.1W (降低以增加挑战性)
    R_backhaul: float = 100e6  # 回程链路带宽 100Mbps = 100e6 bps
    num_channels: int = 10  # 可用信道数量


@dataclass
class CloudConfig:
    """
    云端服务器相关参数配置
    
    云端资源竞争模型说明：
        - 云端总算力 F_c 被所有同时执行的任务共享
        - 每个任务分配的算力 = F_c / max(n_concurrent_tasks, 1)
        - 这反映了真实云端多租户场景的资源竞争
    
    网络传播延迟说明：
        - UAV到云端的物理传播延迟（与带宽无关）
        - 典型值：边缘云 10-30ms，公有云 50-100ms
        - 包含：光纤传播延迟 + 路由器处理延迟 + 排队延迟
    
    Attributes:
        F_c: 云端总算力 (FLOPS)，所有任务共享
        F_per_task_max: 单任务最大分配算力 (FLOPS)，限制单个任务的云端资源占用
        kappa_cloud: 云端能耗系数
        T_propagation: UAV到云端单向传播延迟 (s)，典型值20-50ms
        max_concurrent_tasks: 云端最大并发任务数，用于计算资源竞争
    """
    F_c: float = 500e9  # 云端总算力 500 GFLOPS = 500e9 FLOPS（共享）
    F_per_task_max: float = 30e9  # 单任务最大分配 30 GFLOPS，增加资源竞争压力
    kappa_cloud: float = 1e-29  # 云端能耗系数
    # 网络传播延迟：光纤传播(~5ms/1000km) + 路由处理(~2ms/跳) + 边缘网关(~5ms)
    # 考虑实际网络波动和拥塞，取 40ms
    T_propagation: float = 0.04  # 单向传播延迟 40ms (合理范围: 20-100ms)
    # 云端资源竞争：假设云端服务多个UAV集群，平均并发任务数
    # 根据场景规模动态调整，默认值基于50用户场景的典型负载
    max_concurrent_tasks: int = 15  # 云端同时处理的最大任务数，增加竞争


@dataclass
class EnergyConfig:
    """
    能耗模型相关参数配置
    
    公式参考 (idea118.txt 2.7节):
        边缘计算能耗: E_edge = κ_edge * f² * C
        云端计算能耗: E_cloud = κ_cloud * f² * C
    
    Attributes:
        kappa_edge: 边缘计算能耗系数
        kappa_cloud: 云端计算能耗系数
        P_write: 存储写入功率 (W)，用于Checkpoint
    """
    kappa_edge: float = 1e-28  # 边缘能耗系数
    kappa_cloud: float = 1e-29  # 云端能耗系数
    P_write: float = 7.0  # 存储写入功率 (W)


@dataclass
class PriorityConfig:
    """
    任务优先级计算参数配置
    
    公式参考 (idea118.txt 1.3节):
        ω_i = α₁*ω_data + α₂*ω_comp + α₃*ω_delay + α₄*ω_level
    
    Attributes:
        alpha_data: 数据量因子权重
        alpha_comp: 计算量因子权重
        alpha_delay: 时延紧迫度因子权重（最高）
        alpha_level: 用户等级因子权重
        theta_high_percentile: 高优先级分位数阈值 (80%)
        theta_medium_percentile: 中优先级分位数阈值 (40%)
    """
    alpha_data: float = 0.15  # 数据量因子权重
    alpha_comp: float = 0.25  # 计算量因子权重
    alpha_delay: float = 0.40  # 时延紧迫度因子权重
    alpha_level: float = 0.20  # 用户等级因子权重
    theta_high_percentile: float = 80.0  # 高优先级分位数
    theta_medium_percentile: float = 40.0  # 中优先级分位数


@dataclass
class BiddingConfig:
    """
    投标生成相关参数配置
    
    公式参考 (idea118.txt 2.13节):
        效用: η = β₁*U_time + β₂*U_energy + β₃*U_reliability
    
    Attributes:
        beta_time: 时延效用权重
        beta_energy: 能效效用权重
        beta_reliability: 可靠性效用权重
        top_k: 每用户生成的候选投标数量
        F_threshold: 自由能阈值
        F_max: 自由能最大值
        gamma_cp: Checkpoint时间系数 (s/bit)
    """
    beta_time: float = 0.45  # 时延效用权重
    beta_energy: float = 0.25  # 能效效用权重
    beta_reliability: float = 0.30  # 可靠性效用权重
    top_k: int = 10  # 候选投标数量
    F_threshold: float = 30.0  # 自由能阈值
    F_max: float = 50.0  # 自由能最大值
    gamma_cp: float = 1e-7  # Checkpoint时间系数 0.1s/MB = 0.1/8e6 s/bit


@dataclass
class AuctionConfig:
    """
    组合拍卖相关参数配置
    
    公式参考 (idea118.txt 3.13节):
        步长: ε(t) = ε₀ / (t + 1)
    
    Attributes:
        epsilon_0: 次梯度法初始步长
        max_iterations: 最大迭代次数
        epsilon_tol: 收敛容差
        M_penalty: 高优先级未服务惩罚系数
    """
    epsilon_0: float = 0.5  # 初始步长
    max_iterations: int = 100  # 最大迭代次数
    epsilon_tol: float = 1e-4  # 收敛容差
    M_penalty: float = 100.0  # 惩罚系数


@dataclass
class ExecutionConfig:
    """
    执行调度相关参数配置
    
    Attributes:
        heartbeat_timeout: 心跳超时时间 (s)
        util_target: 目标利用率
        gamma_comp: 算力价格调整系数
        gamma_energy: 能量价格调整系数
        delta_pos: 重定位触发阈值
        w_priority_static: 运行时优先级中静态优先级权重
        w_priority_time: 运行时优先级中剩余时间权重
        w_priority_reliability: 运行时优先级中可靠性权重
        energy_budget_ratio: 单任务能量预算比例
        min_energy_ratio: 候选UAV最低能量比例
        constraint_tolerance: 约束可行性容差
        default_channel_quality: 默认信道质量
        free_energy_scale: 自由能缩放因子
    """
    heartbeat_timeout: float = 5.0  # 心跳超时 (s)
    util_target: float = 0.7  # 目标利用率
    gamma_comp: float = 1.0  # 算力价格调整系数
    gamma_energy: float = 0.3  # 能量价格调整系数
    delta_pos: float = 0.3  # 重定位触发阈值
    w_priority_static: float = 0.4  # 静态优先级权重
    w_priority_time: float = 0.4  # 剩余时间权重
    w_priority_reliability: float = 0.2  # 可靠性权重
    # 新增配置参数（原硬编码值）
    energy_budget_ratio: float = 0.3  # 单任务能量预算比例
    min_energy_ratio: float = 0.2  # 候选UAV最低能量比例
    constraint_tolerance: float = 0.01  # 约束可行性容差 (1%)
    default_channel_quality: float = 0.9  # 默认信道质量
    free_energy_scale: float = 10.0  # 自由能缩放因子


@dataclass
class ScenarioConfig:
    """
    场景相关参数配置
    
    Attributes:
        scene_width: 场景宽度 (m)
        scene_height: 场景高度 (m)
        num_users: 用户数量
        num_uavs: UAV数量 (若为None则自动确定)
        tau_max_range: 任务最大时延范围 (s)
        user_level_range: 用户等级范围
    """
    scene_width: float = 2000.0  # 场景宽度 2km
    scene_height: float = 2000.0  # 场景高度 2km
    num_users: int = 50  # 用户数量
    num_uavs: Optional[int] = None  # UAV数量，None则自动确定
    tau_max_range: Tuple[float, float] = (1.0, 5.0)  # 时延范围 1-5s
    user_level_range: Tuple[int, int] = (1, 5)  # 用户等级 1-5


@dataclass
class SystemConfig:
    """
    系统总配置类，整合所有子配置
    
    Usage:
        # 使用默认配置
        config = SystemConfig()
        
        # 从YAML文件加载
        config = SystemConfig.from_yaml("config.yaml")
        
        # 访问参数
        print(config.uav.f_max)  # UAV最大算力
        print(config.channel.W)  # 信道带宽
    """
    uav: UAVConfig = field(default_factory=UAVConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    bidding: BiddingConfig = field(default_factory=BiddingConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SystemConfig":
        """
        从YAML配置文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            SystemConfig: 配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # 更新各子配置
        if 'uav' in data:
            for key, value in data['uav'].items():
                if hasattr(config.uav, key):
                    setattr(config.uav, key, value)
        
        if 'channel' in data:
            for key, value in data['channel'].items():
                if hasattr(config.channel, key):
                    setattr(config.channel, key, value)
        
        if 'cloud' in data:
            for key, value in data['cloud'].items():
                if hasattr(config.cloud, key):
                    setattr(config.cloud, key, value)
        
        if 'energy' in data:
            for key, value in data['energy'].items():
                if hasattr(config.energy, key):
                    setattr(config.energy, key, value)
        
        if 'priority' in data:
            for key, value in data['priority'].items():
                if hasattr(config.priority, key):
                    setattr(config.priority, key, value)
        
        if 'bidding' in data:
            for key, value in data['bidding'].items():
                if hasattr(config.bidding, key):
                    setattr(config.bidding, key, value)
        
        if 'auction' in data:
            for key, value in data['auction'].items():
                if hasattr(config.auction, key):
                    setattr(config.auction, key, value)
        
        if 'execution' in data:
            for key, value in data['execution'].items():
                if hasattr(config.execution, key):
                    setattr(config.execution, key, value)
        
        if 'scenario' in data:
            for key, value in data['scenario'].items():
                if hasattr(config.scenario, key):
                    # 处理元组类型
                    if key in ['tau_max_range', 'user_level_range'] and isinstance(value, list):
                        value = tuple(value)
                    setattr(config.scenario, key, value)
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        将配置保存到YAML文件
        
        Args:
            yaml_path: 保存路径
        """
        data = {
            'uav': self.uav.__dict__,
            'channel': self.channel.__dict__,
            'cloud': self.cloud.__dict__,
            'energy': self.energy.__dict__,
            'priority': self.priority.__dict__,
            'bidding': self.bidding.__dict__,
            'auction': self.auction.__dict__,
            'execution': self.execution.__dict__,
            'scenario': {
                **self.scenario.__dict__,
                'tau_max_range': list(self.scenario.tau_max_range),
                'user_level_range': list(self.scenario.user_level_range)
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def get_num_uavs(self, num_users: int) -> int:
        """
        根据用户数量确定UAV数量
        
        公式参考 (idea118.txt 0.6节):
            N = 2  if M <= 20
            N = 3  if 20 < M <= 50
            N = 5  if 50 < M <= 100
            N = 8  if M > 100
        
        Args:
            num_users: 用户数量
            
        Returns:
            int: UAV数量
        """
        if self.scenario.num_uavs is not None:
            return self.scenario.num_uavs
        
        if num_users <= 20:
            return 2
        elif num_users <= 50:
            return 3
        elif num_users <= 100:
            return 5
        else:
            return 8
    
    def validate(self) -> List[str]:
        """
        验证配置参数的合法性
        
        Returns:
            List[str]: 错误信息列表，空列表表示配置有效
        """
        errors = []
        
        # 验证权重和为1
        priority_sum = (self.priority.alpha_data + self.priority.alpha_comp + 
                       self.priority.alpha_delay + self.priority.alpha_level)
        if abs(priority_sum - 1.0) > 1e-6:
            errors.append(f"优先级权重和应为1，当前为{priority_sum}")
        
        bidding_sum = (self.bidding.beta_time + self.bidding.beta_energy + 
                      self.bidding.beta_reliability)
        if abs(bidding_sum - 1.0) > 1e-6:
            errors.append(f"投标效用权重和应为1，当前为{bidding_sum}")
        
        exec_sum = (self.execution.w_priority_static + self.execution.w_priority_time + 
                   self.execution.w_priority_reliability)
        if abs(exec_sum - 1.0) > 1e-6:
            errors.append(f"运行时优先级权重和应为1，当前为{exec_sum}")
        
        # 验证正值参数
        if self.uav.f_max <= 0:
            errors.append("UAV最大算力必须为正数")
        if self.uav.E_max <= 0:
            errors.append("UAV电池容量必须为正数")
        if self.channel.W <= 0:
            errors.append("信道带宽必须为正数")
        if self.cloud.F_c <= 0:
            errors.append("云端算力必须为正数")
        
        # 验证范围参数
        if self.scenario.tau_max_range[0] >= self.scenario.tau_max_range[1]:
            errors.append("时延范围下界应小于上界")
        if self.scenario.user_level_range[0] >= self.scenario.user_level_range[1]:
            errors.append("用户等级范围下界应小于上界")
        
        return errors
    
    def summary(self) -> str:
        """
        返回配置摘要字符串
        
        Returns:
            str: 配置摘要
        """
        return f"""
========== 系统配置摘要 ==========

【UAV参数】
  飞行高度: {self.uav.H} m
  最大算力: {self.uav.f_max/1e9:.1f} GFLOPS
  电池容量: {self.uav.E_max/1e3:.1f} kJ

【通信参数】
  信道带宽: {self.channel.W/1e6:.1f} MHz
  回程带宽: {self.channel.R_backhaul/1e6:.1f} Mbps
  用户发射功率: {self.channel.P_tx_user} W

【云端参数】
  云端总算力: {self.cloud.F_c/1e9:.1f} GFLOPS (共享)
  传播延迟: {self.cloud.T_propagation*1000:.1f} ms
  最大并发任务: {self.cloud.max_concurrent_tasks}

【场景参数】
  场景尺寸: {self.scenario.scene_width/1000:.1f}km × {self.scenario.scene_height/1000:.1f}km
  用户数量: {self.scenario.num_users}
  时延范围: {self.scenario.tau_max_range[0]}-{self.scenario.tau_max_range[1]} s

【优先级权重】
  数据量: {self.priority.alpha_data}
  计算量: {self.priority.alpha_comp}
  时延: {self.priority.alpha_delay}
  等级: {self.priority.alpha_level}

【投标效用权重】
  时延: {self.bidding.beta_time}
  能效: {self.bidding.beta_energy}
  可靠性: {self.bidding.beta_reliability}

====================================
"""


# ============ 测试用例 ============

def test_system_config():
    """测试SystemConfig模块"""
    print("=" * 60)
    print("测试 M01: SystemConfig")
    print("=" * 60)
    
    # 测试1: 创建默认配置
    print("\n[Test 1] 创建默认配置...")
    config = SystemConfig()
    assert config.uav.H == 100.0, "UAV高度默认值错误"
    assert config.uav.f_max == 10e9, "UAV算力默认值错误"
    assert config.channel.W == 20e6, "信道带宽默认值错误"
    print("  ✓ 默认配置创建成功")
    
    # 测试2: 验证配置
    print("\n[Test 2] 验证配置参数...")
    errors = config.validate()
    assert len(errors) == 0, f"配置验证失败: {errors}"
    print("  ✓ 配置参数验证通过")
    
    # 测试3: UAV数量自动确定
    print("\n[Test 3] 测试UAV数量自动确定...")
    assert config.get_num_uavs(15) == 2, "M<=20时应返回2"
    assert config.get_num_uavs(30) == 3, "20<M<=50时应返回3"
    assert config.get_num_uavs(80) == 5, "50<M<=100时应返回5"
    assert config.get_num_uavs(150) == 8, "M>100时应返回8"
    print("  ✓ UAV数量规则正确")
    
    # 测试4: 保存和加载YAML
    print("\n[Test 4] 测试YAML保存和加载...")
    test_yaml_path = "/tmp/test_config.yaml"
    config.to_yaml(test_yaml_path)
    loaded_config = SystemConfig.from_yaml(test_yaml_path)
    assert loaded_config.uav.H == config.uav.H, "YAML加载后值不一致"
    assert loaded_config.channel.W == config.channel.W, "YAML加载后值不一致"
    print("  ✓ YAML保存和加载成功")
    
    # 测试5: 配置摘要
    print("\n[Test 5] 生成配置摘要...")
    summary = config.summary()
    assert "UAV参数" in summary, "摘要缺少UAV参数"
    assert "通信参数" in summary, "摘要缺少通信参数"
    print("  ✓ 配置摘要生成成功")
    print(summary)
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_system_config()

#!/usr/bin/env python3
"""
M01: SystemConfig - 系统参数配置模块

功能：统一管理UAV辅助边缘计算DNN协同推理框架的所有系统参数

输入：无（使用默认值）或YAML配置文件
输出：SystemConfig 数据类实例，包含所有系统参数

关键公式：
- 信道增益: h_{i,j} = β_0 / d_{i,j}^2  (公式见idea118.txt 0.8节)
- 传输速率: R_{i,j} = W * log2(1 + P_tx * h_{i,j} / (N_0 * W))  (香农公式)
- 能耗系数: E = κ * f^2 * C  (计算能耗模型)

参考文档：idea118.txt 阶段0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
import os


@dataclass
class UAVParams:
    """UAV相关参数
    
    Attributes:
        f_max: UAV最大算力 (FLOPS)，默认10 GFLOPS
        E_max: UAV电池容量 (J)，默认500 kJ
        H: UAV飞行高度 (m)，默认100m
        P_hover: 悬停功率 (W)，默认150W
        P_fly: 飞行功率 (W)，默认200W
        v_fly: 飞行速度 (m/s)，默认10 m/s
        P_max: 总功率上限 (W)，默认300W
        M_max: 最大同时加载模型数，默认5
    """
    f_max: float = 10e9  # 10 GFLOPS
    E_max: float = 500e3  # 500 kJ
    H: float = 100.0  # 100m
    P_hover: float = 150.0  # 150W
    P_fly: float = 200.0  # 200W
    v_fly: float = 10.0  # 10 m/s
    P_max: float = 300.0  # 300W
    M_max: int = 5  # 最大加载模型数


@dataclass
class CloudParams:
    """云端相关参数
    
    Attributes:
        F_c: 云端总算力 (FLOPS)，默认500 GFLOPS
        R_backhaul: 回程链路带宽 (bps)，默认100 Mbps
    """
    F_c: float = 500e9  # 500 GFLOPS
    R_backhaul: float = 100e6  # 100 Mbps


@dataclass
class ChannelParams:
    """通信信道相关参数
    
    Attributes:
        W: 信道带宽 (Hz)，默认20 MHz
        beta_0: 参考信道增益 (1m处)，默认1e-6
        N_0: 噪声功率谱密度 (W/Hz)，默认10^-17.4 W/Hz
        P_tx: 用户发射功率 (W)，默认0.1W
        P_rx: 接收功率 (W)，默认0.1W (UAV接收)
        P_tx_uav: UAV发射功率 (W)，默认0.5W
    """
    W: float = 20e6  # 20 MHz
    beta_0: float = 1e-6  # 参考信道增益
    N_0: float = 10**(-17.4)  # 噪声功率谱密度 W/Hz
    P_tx: float = 0.3  # 用户发射功率 0.3W
    P_rx: float = 0.1  # UAV接收功率
    P_tx_uav: float = 5.0  # UAV发射功率


@dataclass
class EnergyParams:
    """能耗相关参数
    
    Attributes:
        kappa_edge: 边缘计算能耗系数，默认1e-28
        kappa_cloud: 云端计算能耗系数，默认1e-29
        P_write: Checkpoint写入功率 (W)，默认7W
        gamma_cp: Checkpoint时间系数 (s/byte)，默认0.1 s/MB
    """
    kappa_edge: float = 1e-28  # 边缘能耗系数
    kappa_cloud: float = 1e-29  # 云端能耗系数
    P_write: float = 7.0  # Checkpoint写入功率
    gamma_cp: float = 0.1 / (1024 * 1024)  # 0.1 s/MB -> s/byte


@dataclass  
class PriorityWeights:
    """优先级计算权重 (阶段1)
    
    公式: ω_i = α_1*ω_data + α_2*ω_comp + α_3*ω_delay + α_4*ω_level
    
    Attributes:
        alpha_1: 数据量因子权重，默认0.15
        alpha_2: 计算量因子权重，默认0.25
        alpha_3: 时延紧迫度权重，默认0.40
        alpha_4: 用户等级权重，默认0.20
    """
    alpha_1: float = 0.15  # 数据量
    alpha_2: float = 0.25  # 计算量
    alpha_3: float = 0.40  # 时延紧迫度
    alpha_4: float = 0.20  # 用户等级


@dataclass
class UtilityWeights:
    """效用计算权重 (阶段2)
    
    公式: η = β_1*U_time + β_2*U_energy + β_3*U_reliability
    
    Attributes:
        beta_1: 时延效用权重，默认0.45
        beta_2: 能效效用权重，默认0.25
        beta_3: 可靠性效用权重，默认0.30
    """
    beta_1: float = 0.45  # 时延
    beta_2: float = 0.25  # 能效
    beta_3: float = 0.30  # 可靠性


@dataclass
class FreeEnergyParams:
    """自由能相关参数 (阶段2)
    
    Attributes:
        F_threshold: 自由能阈值，默认30
        F_max: 自由能上限，默认50
        sigma_R: 信道速率方差系数
        sigma_f: 边缘算力方差系数
        sigma_c: 云端算力方差系数
        beta_energy: 能量自由能惩罚系数，默认5
    """
    F_threshold: float = 30.0
    F_max: float = 50.0
    sigma_R: float = 0.1  # 信道速率方差系数
    sigma_f: float = 0.15  # 边缘算力方差系数
    sigma_c: float = 0.05  # 云端算力方差系数
    beta_energy: float = 5.0  # 能量惩罚


@dataclass
class RiskWeights:
    """风险因子权重 (阶段2)
    
    公式: γ = w_free*γ_free + w_energy*γ_energy + w_channel*γ_channel + w_compute*γ_compute
    
    Attributes:
        w_free: 自由能风险权重，默认0.35
        w_energy: 能量风险权重，默认0.25
        w_channel: 信道风险权重，默认0.25
        w_compute: 算力风险权重，默认0.15
    """
    w_free: float = 0.35
    w_energy: float = 0.25
    w_channel: float = 0.25
    w_compute: float = 0.15


@dataclass
class AuctionParams:
    """拍卖相关参数 (阶段3)
    
    Attributes:
        M_penalty: 高优先级违约惩罚系数，默认100
        epsilon_0: 次梯度步长初始值，默认0.5
        T_max: 最大迭代次数，默认100
        epsilon_tol: 收敛容差，默认1e-4
        top_k: 每用户最大投标数，默认10
    """
    M_penalty: float = 100.0
    epsilon_0: float = 0.5
    T_max: int = 100
    epsilon_tol: float = 1e-4
    top_k: int = 10


@dataclass
class ElectionWeights:
    """拍卖方选举评分权重 (阶段1)
    
    公式: S_j = w_1*(E/E_max) + w_2*(f_avail/f_max) + w_3*S_position + w_4*(1-L)
    
    Attributes:
        w_1: 能量权重，默认0.25
        w_2: 算力权重，默认0.30
        w_3: 位置权重，默认0.25
        w_4: 负载权重，默认0.20
    """
    w_1: float = 0.25  # 能量
    w_2: float = 0.30  # 算力
    w_3: float = 0.25  # 位置
    w_4: float = 0.20  # 负载


@dataclass
class DynamicPricingParams:
    """动态定价参数 (阶段4)
    
    Attributes:
        gamma_comp: 算力价格调节系数，默认1.0
        gamma_energy: 能量价格调节系数，默认0.3
        gamma_channel_up: 信道涨价系数，默认0.2
        gamma_channel_down: 信道降价系数，默认0.1
        lambda_smooth: 价格平滑系数，默认0.3
        util_target: 目标利用率，默认0.7
    """
    gamma_comp: float = 1.0
    gamma_energy: float = 0.3
    gamma_channel_up: float = 0.2
    gamma_channel_down: float = 0.1
    lambda_smooth: float = 0.3
    util_target: float = 0.7


@dataclass
class HealthParams:
    """健康度监控参数 (阶段4)
    
    Attributes:
        w_E: 能量健康度权重，默认0.5
        w_L: 负载健康度权重，默认0.3
        w_C: 通信健康度权重，默认0.2
        E_warning: 能量警告阈值比例，默认0.2
        E_critical: 能量临界阈值比例，默认0.1
        E_emergency: 能量紧急阈值比例，默认0.05
        heartbeat_timeout: 心跳超时 (s)，默认5s
    """
    w_E: float = 0.5
    w_L: float = 0.3
    w_C: float = 0.2
    E_warning: float = 0.2
    E_critical: float = 0.1
    E_emergency: float = 0.05
    heartbeat_timeout: float = 5.0


@dataclass
class KMeansParams:
    """加权K-means参数 (阶段0)
    
    Attributes:
        alpha_1: 紧迫度权重，默认0.7
        alpha_2: 数据量权重，默认0.3
        epsilon: 收敛阈值，默认1e-6
        max_iter: 最大迭代次数，默认100
    """
    alpha_1: float = 0.7
    alpha_2: float = 0.3
    epsilon: float = 1e-6
    max_iter: int = 100


@dataclass
class TimeConstraints:
    """时间约束 (各阶段)
    
    Attributes:
        T_init: 系统初始化时间 (s)，默认5-30s
        T_election: 选举时间 (s)，默认0.5s
        T_bid: 投标生成时间 (s)，默认0.2s
        T_auction: 拍卖决策时间 (s)，默认0.1s
        T_total: 端到端调度时间 (s)，默认1s
    """
    T_init: float = 30.0
    T_election: float = 0.5
    T_bid: float = 0.2
    T_auction: float = 0.1
    T_total: float = 1.0


@dataclass
class CostCoefficients:
    """成本系数 (定价)
    
    Attributes:
        c_edge: 边缘计算成本 (元/GFLOPS)，默认0.01
        c_cloud: 云端计算成本 (元/GFLOPS)，默认0.005
        c_energy: 能量成本 (元/kJ)，默认0.05
        c_trans: 传输成本 (元/MB)，默认0.001
        c_load: 模型加载成本 (元)，默认0.5
    """
    c_edge: float = 0.01
    c_cloud: float = 0.005
    c_energy: float = 0.05
    c_trans: float = 0.001
    c_load: float = 0.5


@dataclass
class SystemConfig:
    """系统配置主类 - 整合所有参数
    
    包含UAV辅助边缘计算DNN协同推理框架的全部配置参数。
    
    Attributes:
        uav: UAV相关参数
        cloud: 云端相关参数
        channel: 通信信道参数
        energy: 能耗参数
        priority_weights: 优先级权重
        utility_weights: 效用权重
        free_energy: 自由能参数
        risk_weights: 风险权重
        auction: 拍卖参数
        election_weights: 选举权重
        dynamic_pricing: 动态定价参数
        health: 健康度参数
        kmeans: K-means参数
        time_constraints: 时间约束
        cost: 成本系数
        
    Example:
        >>> config = SystemConfig()
        >>> print(config.uav.f_max)
        10000000000.0
        >>> config = SystemConfig.from_yaml('config.yaml')
    """
    uav: UAVParams = field(default_factory=UAVParams)
    cloud: CloudParams = field(default_factory=CloudParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    energy: EnergyParams = field(default_factory=EnergyParams)
    priority_weights: PriorityWeights = field(default_factory=PriorityWeights)
    utility_weights: UtilityWeights = field(default_factory=UtilityWeights)
    free_energy: FreeEnergyParams = field(default_factory=FreeEnergyParams)
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    auction: AuctionParams = field(default_factory=AuctionParams)
    election_weights: ElectionWeights = field(default_factory=ElectionWeights)
    dynamic_pricing: DynamicPricingParams = field(default_factory=DynamicPricingParams)
    health: HealthParams = field(default_factory=HealthParams)
    kmeans: KMeansParams = field(default_factory=KMeansParams)
    time_constraints: TimeConstraints = field(default_factory=TimeConstraints)
    cost: CostCoefficients = field(default_factory=CostCoefficients)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SystemConfig':
        """从YAML文件加载配置
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            SystemConfig实例
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # 递归更新各子配置
        if 'uav' in data:
            config.uav = UAVParams(**data['uav'])
        if 'cloud' in data:
            config.cloud = CloudParams(**data['cloud'])
        if 'channel' in data:
            config.channel = ChannelParams(**data['channel'])
        if 'energy' in data:
            config.energy = EnergyParams(**data['energy'])
        if 'priority_weights' in data:
            config.priority_weights = PriorityWeights(**data['priority_weights'])
        if 'utility_weights' in data:
            config.utility_weights = UtilityWeights(**data['utility_weights'])
        if 'free_energy' in data:
            config.free_energy = FreeEnergyParams(**data['free_energy'])
        if 'risk_weights' in data:
            config.risk_weights = RiskWeights(**data['risk_weights'])
        if 'auction' in data:
            config.auction = AuctionParams(**data['auction'])
        if 'election_weights' in data:
            config.election_weights = ElectionWeights(**data['election_weights'])
        if 'dynamic_pricing' in data:
            config.dynamic_pricing = DynamicPricingParams(**data['dynamic_pricing'])
        if 'health' in data:
            config.health = HealthParams(**data['health'])
        if 'kmeans' in data:
            config.kmeans = KMeansParams(**data['kmeans'])
        if 'time_constraints' in data:
            config.time_constraints = TimeConstraints(**data['time_constraints'])
        if 'cost' in data:
            config.cost = CostCoefficients(**data['cost'])
            
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """保存配置到YAML文件
        
        Args:
            yaml_path: 输出YAML文件路径
        """
        from dataclasses import asdict
        
        data = {
            'uav': asdict(self.uav),
            'cloud': asdict(self.cloud),
            'channel': asdict(self.channel),
            'energy': asdict(self.energy),
            'priority_weights': asdict(self.priority_weights),
            'utility_weights': asdict(self.utility_weights),
            'free_energy': asdict(self.free_energy),
            'risk_weights': asdict(self.risk_weights),
            'auction': asdict(self.auction),
            'election_weights': asdict(self.election_weights),
            'dynamic_pricing': asdict(self.dynamic_pricing),
            'health': asdict(self.health),
            'kmeans': asdict(self.kmeans),
            'time_constraints': asdict(self.time_constraints),
            'cost': asdict(self.cost),
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def get_uav_count_by_users(self, num_users: int) -> int:
        """根据用户数量确定UAV数量
        
        公式 (idea118.txt 0.6节):
        N = 2  if M <= 20
        N = 3  if 20 < M <= 50
        N = 5  if 50 < M <= 100
        N = 8  if M > 100
        
        Args:
            num_users: 用户数量M
            
        Returns:
            UAV数量N
        """
        if num_users <= 20:
            return 2
        elif num_users <= 50:
            return 3
        elif num_users <= 100:
            return 5
        else:
            return 8
    
    def validate(self) -> List[str]:
        """验证配置参数有效性
        
        Returns:
            错误消息列表，空列表表示配置有效
        """
        errors = []
        
        # UAV参数验证
        if self.uav.f_max <= 0:
            errors.append("UAV最大算力必须为正数")
        if self.uav.E_max <= 0:
            errors.append("UAV电池容量必须为正数")
        if self.uav.H <= 0:
            errors.append("UAV飞行高度必须为正数")
            
        # 权重和验证
        priority_sum = (self.priority_weights.alpha_1 + self.priority_weights.alpha_2 + 
                       self.priority_weights.alpha_3 + self.priority_weights.alpha_4)
        if abs(priority_sum - 1.0) > 1e-6:
            errors.append(f"优先级权重之和必须为1，当前为{priority_sum}")
            
        utility_sum = (self.utility_weights.beta_1 + self.utility_weights.beta_2 + 
                      self.utility_weights.beta_3)
        if abs(utility_sum - 1.0) > 1e-6:
            errors.append(f"效用权重之和必须为1，当前为{utility_sum}")
            
        risk_sum = (self.risk_weights.w_free + self.risk_weights.w_energy + 
                   self.risk_weights.w_channel + self.risk_weights.w_compute)
        if abs(risk_sum - 1.0) > 1e-6:
            errors.append(f"风险权重之和必须为1，当前为{risk_sum}")
            
        election_sum = (self.election_weights.w_1 + self.election_weights.w_2 + 
                       self.election_weights.w_3 + self.election_weights.w_4)
        if abs(election_sum - 1.0) > 1e-6:
            errors.append(f"选举权重之和必须为1，当前为{election_sum}")
            
        health_sum = self.health.w_E + self.health.w_L + self.health.w_C
        if abs(health_sum - 1.0) > 1e-6:
            errors.append(f"健康度权重之和必须为1，当前为{health_sum}")
            
        return errors


# 全局默认配置实例
DEFAULT_CONFIG = SystemConfig()


def get_default_config() -> SystemConfig:
    """获取默认配置实例
    
    Returns:
        默认SystemConfig实例
    """
    return SystemConfig()


# ============== 测试用例 ==============

def test_system_config():
    """测试SystemConfig模块"""
    print("=" * 50)
    print("测试 M01: SystemConfig")
    print("=" * 50)
    
    # 测试1: 创建默认配置
    print("\n[测试1] 创建默认配置...")
    config = SystemConfig()
    assert config.uav.f_max == 10e9, "UAV最大算力应为10 GFLOPS"
    assert config.cloud.F_c == 500e9, "云端算力应为500 GFLOPS"
    print("  ✓ 默认配置创建成功")
    
    # 测试2: UAV数量确定规则
    print("\n[测试2] UAV数量确定规则...")
    assert config.get_uav_count_by_users(15) == 2, "15用户应分配2个UAV"
    assert config.get_uav_count_by_users(30) == 3, "30用户应分配3个UAV"
    assert config.get_uav_count_by_users(80) == 5, "80用户应分配5个UAV"
    assert config.get_uav_count_by_users(150) == 8, "150用户应分配8个UAV"
    print("  ✓ UAV数量规则正确")
    
    # 测试3: 配置验证
    print("\n[测试3] 配置验证...")
    errors = config.validate()
    assert len(errors) == 0, f"默认配置应该有效，但发现错误: {errors}"
    print("  ✓ 默认配置验证通过")
    
    # 测试4: 保存和加载YAML
    print("\n[测试4] YAML保存/加载...")
    yaml_path = "/tmp/test_config.yaml"
    config.to_yaml(yaml_path)
    loaded_config = SystemConfig.from_yaml(yaml_path)
    assert loaded_config.uav.f_max == config.uav.f_max, "加载的配置应与原配置一致"
    print("  ✓ YAML保存/加载成功")
    
    # 测试5: 参数访问
    print("\n[测试5] 参数访问...")
    print(f"  - UAV最大算力: {config.uav.f_max/1e9:.1f} GFLOPS")
    print(f"  - 云端总算力: {config.cloud.F_c/1e9:.1f} GFLOPS")
    print(f"  - 信道带宽: {config.channel.W/1e6:.1f} MHz")
    print(f"  - 优先级权重(时延): {config.priority_weights.alpha_3}")
    print(f"  - 自由能阈值: {config.free_energy.F_threshold}")
    print("  ✓ 参数访问正常")
    
    print("\n" + "=" * 50)
    print("所有测试通过! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_system_config()

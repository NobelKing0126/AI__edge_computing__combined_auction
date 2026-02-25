"""
常量定义模块

功能：统一管理项目中的所有常量，避免硬编码
分类：
    1. 数值稳定性常量 - 用于防止除零等数值问题
    2. 自由能计算常量 - 主动推理相关参数
    3. 定价常量 - 动态定价相关参数
    4. 约束常量 - 优化约束相关参数
    5. 通信常量 - 通信模型相关参数
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalConstants:
    """
    数值稳定性常量
    
    用于防止除零、权重下界等数值稳定性问题
    """
    EPSILON: float = 1e-10          # 通用除法保护
    MIN_WEIGHT: float = 1e-6        # 最小权重值
    MIN_PROBABILITY: float = 1e-12  # 最小概率值
    MAX_VALUE: float = 1e10         # 最大数值上界
    CONVERGENCE_TOL: float = 1e-6   # 收敛判断容差


@dataclass(frozen=True)
class FreeEnergyConstants:
    """
    自由能计算常量
    
    用于主动推理框架中的自由能计算
    """
    SCALE_FACTOR: float = 5.0       # 自由能缩放因子 (降低惩罚提升成功率)
    MAX_FREE_ENERGY: float = 100.0  # 自由能上界
    ENERGY_PENALTY: float = 3.0     # 能量不足惩罚因子 (降低)
    
    # 自由能风险权重 (更重视算力和时延)
    W_FREE: float = 0.30            # 自由能权重 (降低)
    W_ENERGY: float = 0.20          # 能量风险权重
    W_CHANNEL: float = 0.20         # 信道风险权重
    W_COMPUTE: float = 0.30         # 算力风险权重 (提升)


@dataclass(frozen=True)
class PricingConstants:
    """
    定价相关常量
    
    用于动态定价机制中的价格边界
    """
    # 最小价格（防止价格归零）
    MIN_COMPUTE_PRICE: float = 1e-12    # 最小算力价格 ($/FLOPS)
    MIN_ENERGY_PRICE: float = 1e-6      # 最小能量价格 ($/J)
    MIN_CHANNEL_PRICE: float = 1e-9     # 最小信道价格 ($/bps)
    
    # 价格调整边界
    PRICE_LOWER_BOUND_RATIO: float = 0.5  # 价格下界倍率
    PRICE_UPPER_BOUND_RATIO: float = 3.0  # 价格上界倍率
    
    # 基础价格
    BASE_COMPUTE_PRICE: float = 1e-9    # 基础算力价格
    BASE_ENERGY_PRICE: float = 1e-3     # 基础能量价格
    BASE_CHANNEL_PRICE: float = 1e-6    # 基础信道价格


@dataclass(frozen=True)
class ConstraintConstants:
    """
    约束相关常量
    
    用于优化问题中的约束容差等
    """
    FEASIBILITY_TOLERANCE: float = 0.02     # 可行性容差 (2%) - 提升以放宽约束
    OPTIMALITY_TOLERANCE: float = 1e-4      # 最优性容差
    CONSTRAINT_VIOLATION_TOL: float = 0.02  # 约束违反容差 (2%)


@dataclass(frozen=True)
class CommunicationConstants:
    """
    通信相关常量
    
    用于通信模型中的传输计算
    """
    # 结果返回数据大小比例（相对于中间特征）
    RESULT_SIZE_RATIO: float = 0.1      
    # Checkpoint元数据大小 (bits)
    CHECKPOINT_METADATA_SIZE: float = 8 * 1024  # 1KB
    # 最小传输时间 (s)
    MIN_TRANSMISSION_TIME: float = 0.001  # 1ms


@dataclass(frozen=True)
class ResourceConstants:
    """
    资源相关常量
    
    用于资源分配和评估
    """
    # 能量预算比例 (提升以允许更多任务成功)
    ENERGY_BUDGET_RATIO: float = 0.5        # 单任务最大使用50%能量
    # 最低能量比例要求 (降低以允许更多UAV参与)
    MIN_ENERGY_RATIO: float = 0.1           # 候选需至少10%能量
    # 默认信道质量 (提升)
    DEFAULT_CHANNEL_QUALITY: float = 0.95    
    # 安全能量阈值 (降低)
    SAFETY_ENERGY_RATIO: float = 0.1        # 10%能量为安全阈值


# 创建全局常量实例
NUMERICAL = NumericalConstants()
FREE_ENERGY = FreeEnergyConstants()
PRICING = PricingConstants()
CONSTRAINT = ConstraintConstants()
COMMUNICATION = CommunicationConstants()
RESOURCE = ResourceConstants()


# ============ 测试用例 ============

def test_constants():
    """测试常量模块"""
    print("=" * 60)
    print("测试常量模块")
    print("=" * 60)
    
    print(f"\n数值常量:")
    print(f"  EPSILON = {NUMERICAL.EPSILON}")
    print(f"  MIN_WEIGHT = {NUMERICAL.MIN_WEIGHT}")
    
    print(f"\n自由能常量:")
    print(f"  SCALE_FACTOR = {FREE_ENERGY.SCALE_FACTOR}")
    print(f"  MAX_FREE_ENERGY = {FREE_ENERGY.MAX_FREE_ENERGY}")
    
    print(f"\n定价常量:")
    print(f"  MIN_COMPUTE_PRICE = {PRICING.MIN_COMPUTE_PRICE}")
    print(f"  BASE_COMPUTE_PRICE = {PRICING.BASE_COMPUTE_PRICE}")
    
    print(f"\n约束常量:")
    print(f"  FEASIBILITY_TOLERANCE = {CONSTRAINT.FEASIBILITY_TOLERANCE}")
    
    print(f"\n通信常量:")
    print(f"  RESULT_SIZE_RATIO = {COMMUNICATION.RESULT_SIZE_RATIO}")
    
    print(f"\n资源常量:")
    print(f"  ENERGY_BUDGET_RATIO = {RESOURCE.ENERGY_BUDGET_RATIO}")
    print(f"  MIN_ENERGY_RATIO = {RESOURCE.MIN_ENERGY_RATIO}")
    
    print("\n" + "=" * 60)
    print("常量模块测试完成! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_constants()

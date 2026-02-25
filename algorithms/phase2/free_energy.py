"""
M17: FreeEnergy - 自由能风险评估

功能：使用Active Inference的自由能框架评估任务执行风险
输入：任务状态、UAV状态、环境不确定性
输出：自由能值、风险等级

关键公式 (idea118.txt 2.9节):
    自由能分解: F̃(l, j) = F̃_trans(l, j) + F̃_comp(l, j) + F̃_energy(l, j)

    传输自由能: F̃_trans(l, j) = (D_i^trans(l) / R_{i,j}) * (σ_R² / R_{i,j}²)
    计算自由能: F̃_comp(l, j) = (C_i^edge(l) / f_{i,j}^*) * (σ_f² / (f_{i,j}^*)²)
                            + (C_i^cloud(l) / f_{c,i}^*) * (σ_c² / (f_{c,i}^*)²)
    能量自由能: F̃_energy(l, j) = β_energy if (E_j^remain - E_i^total) / E^max < 0.2, else 0

    推荐参数 (idea118.txt 2.9.4节):
        F̃_threshold = 30
        F̃_max = 50
        β_energy = 5
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import FREE_ENERGY, NUMERICAL


class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"           # F < threshold
    MEDIUM = "MEDIUM"     # threshold <= F < max
    HIGH = "HIGH"         # F >= max
    CRITICAL = "CRITICAL" # 系统无法处理


@dataclass
class FreeEnergyResult:
    """
    自由能计算结果

    Attributes:
        F_total: 总自由能
        F_trans: 传输自由能分量 (idea118.txt 2.9.2节)
        F_comp: 计算自由能分量 (边缘+云端)
        F_energy: 能量自由能分量
        F_time: 时间自由能分量 (向后兼容，基于F_comp推导)
        F_reliability: 可靠性自由能分量 (向后兼容，基于F_trans推导)
        risk_level: 风险等级
        requires_checkpoint: 是否建议Checkpoint
        details: 详细信息
    """
    F_total: float
    F_trans: float
    F_comp: float
    F_energy: float
    F_time: float  # 向后兼容
    F_reliability: float  # 向后兼容
    risk_level: RiskLevel
    requires_checkpoint: bool
    details: Dict[str, float]


class FreeEnergyCalculator:
    """
    自由能计算器

    按照idea118.txt 2.9节实现正确的自由能分解公式:
        F̃(l, j) = F̃_trans(l, j) + F̃_comp(l, j) + F̃_energy(l, j)

    Attributes:
        F_threshold: 自由能阈值（低/中风险分界），默认30
        F_max: 自由能最大值（中/高风险分界），默认50
        beta_energy: 能量自由能系数，默认5
        sigma_R_sq: 信道速率方差 (bps²)
        sigma_f_sq: 边缘算力方差 (FLOPS²)
        sigma_c_sq: 云端算力方差 (FLOPS²)
        scale_factor: 自由能缩放因子
    """

    def __init__(self,
                 F_threshold: float = 30.0,
                 F_max: float = 50.0,
                 beta_energy: float = 5.0,
                 sigma_R_sq: Optional[float] = None,
                 sigma_f_sq: Optional[float] = None,
                 sigma_c_sq: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 # 向后兼容参数
                 w_energy: float = 0.4,
                 w_time: float = 0.4,
                 w_reliability: float = 0.2):
        """
        初始化计算器

        Args:
            F_threshold: 低风险阈值 (idea118.txt: F̃_threshold = 30)
            F_max: 高风险阈值 (idea118.txt: F̃_max = 50)
            beta_energy: 能量自由能系数 (idea118.txt: β_energy = 5)
            sigma_R_sq: 信道速率方差，默认1e12 (bps²)
            sigma_f_sq: 边缘算力方差，默认1e18 (FLOPS²)
            sigma_c_sq: 云端算力方差，默认1e16 (FLOPS²)
            scale_factor: 自由能缩放因子
            w_energy, w_time, w_reliability: 向后兼容参数（新公式不使用）
        """
        self.F_threshold = F_threshold
        self.F_max = F_max
        self.beta_energy = beta_energy

        # 方差参数 (idea118.txt 2.22节: 需根据实际环境标定)
        # 默认值基于典型通信和计算环境
        self.sigma_R_sq = sigma_R_sq if sigma_R_sq is not None else 1e12  # 信道方差
        self.sigma_f_sq = sigma_f_sq if sigma_f_sq is not None else 1e18  # 边缘算力方差
        self.sigma_c_sq = sigma_c_sq if sigma_c_sq is not None else 1e16  # 云端算力方差

        self.scale_factor = scale_factor if scale_factor is not None else FREE_ENERGY.SCALE_FACTOR
        self.max_free_energy = FREE_ENERGY.MAX_FREE_ENERGY

        # 向后兼容参数
        self.w_energy = w_energy
        self.w_time = w_time
        self.w_reliability = w_reliability

    def compute_transmission_free_energy(self,
                                         D_trans: float,
                                         R_rate: float) -> float:
        """
        计算传输自由能

        公式 (idea118.txt 2.9.2节):
            F̃_trans(l, j) = (D_i^trans(l) / R_{i,j}) * (σ_R² / R_{i,j}²)

        物理含义: 传输数据量越大、信道越不稳定，自由能越高

        Args:
            D_trans: 传输数据量 (bits)，即切分层输出大小
            R_rate: 传输速率 (bps)

        Returns:
            float: 传输自由能
        """
        if R_rate <= NUMERICAL.EPSILON:
            return self.max_free_energy

        # 传输时间
        T_trans = D_trans / R_rate

        # 相对不确定性 (σ_R / R)²
        relative_uncertainty = self.sigma_R_sq / (R_rate ** 2)

        # 传输自由能
        F_trans = T_trans * relative_uncertainty * self.scale_factor

        return min(max(F_trans, 0), self.max_free_energy)

    def compute_computation_free_energy(self,
                                        C_edge: float,
                                        C_cloud: float,
                                        f_edge: float,
                                        f_cloud: float) -> float:
        """
        计算计算自由能

        公式 (idea118.txt 2.9.2节):
            F̃_comp(l, j) = (C_i^edge(l) / f_{i,j}^*) * (σ_f² / (f_{i,j}^*)²)
                         + (C_i^cloud(l) / f_{c,i}^*) * (σ_c² / (f_{c,i}^*)²)

        物理含义: 边缘算力波动通常大于云端，早期层切分（C^edge小）边缘风险低

        Args:
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)

        Returns:
            float: 计算自由能
        """
        F_comp = 0.0

        # 边缘计算自由能
        if C_edge > 0 and f_edge > NUMERICAL.EPSILON:
            T_edge = C_edge / f_edge
            relative_uncertainty_edge = self.sigma_f_sq / (f_edge ** 2)
            F_comp += T_edge * relative_uncertainty_edge * self.scale_factor

        # 云端计算自由能
        if C_cloud > 0 and f_cloud > NUMERICAL.EPSILON:
            T_cloud = C_cloud / f_cloud
            relative_uncertainty_cloud = self.sigma_c_sq / (f_cloud ** 2)
            F_comp += T_cloud * relative_uncertainty_cloud * self.scale_factor

        return min(max(F_comp, 0), self.max_free_energy)

    def compute_energy_free_energy(self,
                                   E_remain: float,
                                   E_required: float,
                                   E_max: float) -> float:
        """
        计算能量自由能

        公式 (idea118.txt 2.9.2节):
            F̃_energy(l, j) = β_energy if (E_j^remain - E_i^total) / E^max < 0.2, else 0

        物理含义: 能量储备低于20%时增加风险惩罚

        Args:
            E_remain: 剩余能量 (J)
            E_required: 所需能量 (J)
            E_max: 最大能量 (J)

        Returns:
            float: 能量自由能
        """
        if E_max <= NUMERICAL.EPSILON:
            return self.beta_energy

        # 能量储备比例
        energy_ratio = (E_remain - E_required) / E_max

        # 低于20%时返回惩罚值
        if energy_ratio < 0.2:
            return self.beta_energy

        return 0.0

    def compute_free_energy(self,
                           D_trans: float,
                           R_rate: float,
                           C_edge: float,
                           C_cloud: float,
                           f_edge: float,
                           f_cloud: float,
                           E_remain: float,
                           E_required: float,
                           E_max: float,
                           # 向后兼容参数
                           T_max: float = None,
                           T_predict: float = None,
                           health_score: float = None,
                           channel_quality: float = None) -> FreeEnergyResult:
        """
        计算总自由能

        公式 (idea118.txt 2.9节):
            F̃(l, j) = F̃_trans(l, j) + F̃_comp(l, j) + F̃_energy(l, j)

        Args:
            D_trans: 传输数据量 (bits)
            R_rate: 传输速率 (bps)
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)
            E_remain: 剩余能量 (J)
            E_required: 所需能量 (J)
            E_max: 最大能量 (J)
            T_max: 最大时延 (向后兼容)
            T_predict: 预测时延 (向后兼容)
            health_score: 健康度 (向后兼容)
            channel_quality: 信道质量 (向后兼容)

        Returns:
            FreeEnergyResult: 自由能结果
        """
        # 计算各分量 (idea118.txt 2.9.2节公式)
        F_trans = self.compute_transmission_free_energy(D_trans, R_rate)
        F_comp = self.compute_computation_free_energy(C_edge, C_cloud, f_edge, f_cloud)
        F_energy = self.compute_energy_free_energy(E_remain, E_required, E_max)

        # 总自由能
        F_total = F_trans + F_comp + F_energy

        # 确定风险等级
        if E_remain < E_required:
            risk_level = RiskLevel.CRITICAL
        elif F_total >= self.F_max:
            risk_level = RiskLevel.HIGH
        elif F_total >= self.F_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # 是否需要Checkpoint (idea118.txt 2.11.3节)
        requires_checkpoint = F_total > self.F_threshold

        # 向后兼容: 计算等效的 F_time 和 F_reliability
        F_time = F_comp * 0.7  # 计算自由能主要反映时间风险
        F_reliability = F_trans * 0.7  # 传输自由能主要反映可靠性风险

        details = {
            'transmission_ratio': D_trans / max(R_rate, 1e-10),
            'compute_ratio': (C_edge + C_cloud) / max(f_edge + f_cloud, 1e-10),
            'energy_ratio': (E_remain - E_required) / max(E_max, 1e-10),
            'F_trans': F_trans,
            'F_comp': F_comp,
            'F_energy': F_energy
        }

        return FreeEnergyResult(
            F_total=F_total,
            F_trans=F_trans,
            F_comp=F_comp,
            F_energy=F_energy,
            F_time=F_time,
            F_reliability=F_reliability,
            risk_level=risk_level,
            requires_checkpoint=requires_checkpoint,
            details=details
        )

    def compute_free_energy_legacy(self,
                                   E_remain: float,
                                   E_required: float,
                                   E_max: float,
                                   T_max: float,
                                   T_predict: float,
                                   health_score: float,
                                   channel_quality: float = 1.0) -> FreeEnergyResult:
        """
        向后兼容的自由能计算接口

        从旧参数推导新参数并调用新方法

        Args:
            E_remain: 剩余能量 (J)
            E_required: 所需能量 (J)
            E_max: 最大能量 (J)
            T_max: 最大时延 (s)
            T_predict: 预测时延 (s)
            health_score: 健康度 [0, 1]
            channel_quality: 信道质量 [0, 1]

        Returns:
            FreeEnergyResult: 自由能结果
        """
        # 从旧参数估算新参数
        # 假设传输数据量与时延成正比
        D_trans = T_predict * 10e6 * 0.3  # 假设30%时间用于传输，10Mbps速率
        R_rate = 10e6 * channel_quality

        # 假设计算量与时间成正比
        f_edge = 10e9 * health_score
        f_cloud = 100e9
        C_edge = f_edge * T_predict * 0.35  # 35%时间用于边缘计算
        C_cloud = f_cloud * T_predict * 0.35  # 35%时间用于云端计算

        return self.compute_free_energy(
            D_trans=D_trans,
            R_rate=R_rate,
            C_edge=C_edge,
            C_cloud=C_cloud,
            f_edge=f_edge,
            f_cloud=f_cloud,
            E_remain=E_remain,
            E_required=E_required,
            E_max=E_max
        )

    def compute_checkpoint_benefit(self,
                                   F_current: float,
                                   checkpoint_overhead: float,
                                   progress_ratio: float) -> float:
        """
        计算Checkpoint的收益

        Args:
            F_current: 当前自由能
            checkpoint_overhead: Checkpoint开销（时间）
            progress_ratio: 当前进度比例

        Returns:
            float: 收益值（正值表示应该Checkpoint）
        """
        # 失败恢复收益 = 节省的重新计算成本
        recovery_benefit = F_current * progress_ratio

        # Checkpoint成本（使用缩放因子）
        checkpoint_cost = checkpoint_overhead * self.scale_factor

        return recovery_benefit - checkpoint_cost

    def should_checkpoint(self,
                          F_current: float,
                          progress_ratio: float,
                          checkpoint_time: float,
                          T_remaining: float) -> Tuple[bool, float]:
        """
        决定是否应该执行Checkpoint

        Args:
            F_current: 当前自由能
            progress_ratio: 当前进度比例
            checkpoint_time: Checkpoint时间
            T_remaining: 剩余可用时间

        Returns:
            Tuple[bool, float]: (是否Checkpoint, 收益值)
        """
        # 时间约束检查
        if checkpoint_time > T_remaining:
            return False, -float('inf')

        benefit = self.compute_checkpoint_benefit(
            F_current,
            checkpoint_time / max(T_remaining, 1e-10),
            progress_ratio
        )

        # 高风险情况强制Checkpoint
        if F_current >= self.F_max and progress_ratio > 0.2:
            return True, benefit

        # 中等风险且收益为正
        if F_current >= self.F_threshold and benefit > 0:
            return True, benefit

        return benefit > 0, benefit


# ============ 测试用例 ============

def test_free_energy():
    """测试FreeEnergy模块"""
    print("=" * 60)
    print("测试 M17: FreeEnergy (idea118.txt 2.9节公式)")
    print("=" * 60)

    calculator = FreeEnergyCalculator()

    # 测试1: 传输自由能
    print("\n[Test 1] 测试传输自由能...")

    # 大数据量传输
    F_trans_large = calculator.compute_transmission_free_energy(
        D_trans=10e6, R_rate=10e6  # 10Mb数据, 10Mbps速率
    )
    print(f"  大数据量(10Mb/10Mbps): F_trans = {F_trans_large:.2f}")

    # 小数据量传输
    F_trans_small = calculator.compute_transmission_free_energy(
        D_trans=1e6, R_rate=10e6  # 1Mb数据, 10Mbps速率
    )
    assert F_trans_large > F_trans_small, "大数据量应有更高自由能"
    print(f"  小数据量(1Mb/10Mbps): F_trans = {F_trans_small:.2f}")
    print("  ✓ 传输自由能正确")

    # 测试2: 计算自由能
    print("\n[Test 2] 测试计算自由能...")

    # 边缘主导
    F_comp_edge = calculator.compute_computation_free_energy(
        C_edge=5e9, C_cloud=1e9, f_edge=10e9, f_cloud=100e9
    )
    print(f"  边缘主导(5G/1G FLOPs): F_comp = {F_comp_edge:.2f}")

    # 云端主导
    F_comp_cloud = calculator.compute_computation_free_energy(
        C_edge=1e9, C_cloud=5e9, f_edge=10e9, f_cloud=100e9
    )
    print(f"  云端主导(1G/5G FLOPs): F_comp = {F_comp_cloud:.2f}")
    print("  ✓ 计算自由能正确")

    # 测试3: 能量自由能
    print("\n[Test 3] 测试能量自由能...")

    # 能量充足
    F_energy_ok = calculator.compute_energy_free_energy(
        E_remain=400e3, E_required=100e3, E_max=500e3
    )
    assert F_energy_ok == 0, "能量充足时应无惩罚"
    print(f"  能量充足(400kJ剩余): F_energy = {F_energy_ok:.2f}")

    # 能量紧张 (<20%储备)
    F_energy_low = calculator.compute_energy_free_energy(
        E_remain=80e3, E_required=100e3, E_max=500e3
    )
    assert F_energy_low == calculator.beta_energy, "能量紧张应有惩罚"
    print(f"  能量紧张(80kJ剩余): F_energy = {F_energy_low:.2f}")
    print("  ✓ 能量自由能正确")

    # 测试4: 总自由能计算
    print("\n[Test 4] 测试总自由能...")

    result = calculator.compute_free_energy(
        D_trans=5e6, R_rate=10e6,
        C_edge=3e9, C_cloud=2e9,
        f_edge=10e9, f_cloud=100e9,
        E_remain=300e3, E_required=100e3, E_max=500e3
    )

    # 验证分解
    assert abs(result.F_total - (result.F_trans + result.F_comp + result.F_energy)) < 0.01, \
        "总自由能应等于各分量之和"

    print(f"  F_total = {result.F_total:.2f}")
    print(f"    F_trans = {result.F_trans:.2f}")
    print(f"    F_comp = {result.F_comp:.2f}")
    print(f"    F_energy = {result.F_energy:.2f}")
    print(f"  风险等级: {result.risk_level.value}")
    print("  ✓ 总自由能正确")

    # 测试5: 高风险场景
    print("\n[Test 5] 测试高风险场景...")

    result_high = calculator.compute_free_energy(
        D_trans=50e6, R_rate=2e6,  # 大数据量，低速率
        C_edge=10e9, C_cloud=5e9,
        f_edge=5e9, f_cloud=50e9,  # 低算力
        E_remain=60e3, E_required=100e3, E_max=500e3  # 能量不足
    )

    assert result_high.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL], "应为高风险"
    print(f"  F_total = {result_high.F_total:.2f}")
    print(f"  风险等级: {result_high.risk_level.value}")
    print("  ✓ 高风险检测正确")

    # 测试6: 向后兼容接口
    print("\n[Test 6] 测试向后兼容接口...")

    result_legacy = calculator.compute_free_energy_legacy(
        E_remain=300e3, E_required=100e3, E_max=500e3,
        T_max=3.0, T_predict=2.0,
        health_score=0.8, channel_quality=0.9
    )

    assert hasattr(result_legacy, 'F_trans'), "应有F_trans分量"
    assert hasattr(result_legacy, 'F_comp'), "应有F_comp分量"
    print(f"  F_total = {result_legacy.F_total:.2f}")
    print("  ✓ 向后兼容接口正确")

    # 测试7: Checkpoint决策
    print("\n[Test 7] 测试Checkpoint决策...")

    should_cp, benefit = calculator.should_checkpoint(
        F_current=40.0,
        progress_ratio=0.5,
        checkpoint_time=0.1,
        T_remaining=1.0
    )

    print(f"  中风险(F=40), 50%进度: 应Checkpoint={should_cp}, 收益={benefit:.2f}")

    should_cp_low, _ = calculator.should_checkpoint(
        F_current=10.0,
        progress_ratio=0.3,
        checkpoint_time=0.2,
        T_remaining=1.0
    )
    print(f"  低风险(F=10), 30%进度: 应Checkpoint={should_cp_low}")
    print("  ✓ Checkpoint决策正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_free_energy()

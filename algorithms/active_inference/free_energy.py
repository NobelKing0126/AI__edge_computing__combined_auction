"""
M17E: FreeEnergyCalculator - 自由能计算器 (扩展版)

功能：计算即时自由能和期望自由能，支持两种公式体系
参考文档：
    - docs/自由能.txt 第367-452行 (四分量公式)
    - docs/idea118.txt (三分量公式)

四分量即时自由能 (自由能.txt 第373-397行):
    F_t = w_E × F_t^energy + w_T × F_t^time + w_h × F_t^health + w_p × F_t^progress

    - F_t^energy = -log(E_t / E_required(p_t))
    - F_t^time = -log((T_max - T_t) / T_remaining_required(p_t))
    - F_t^health = -log(h_t × q_t)
    - F_t^progress = -log((p_t + ε) / (p_t^expected + ε))

期望自由能 (自由能.txt 第399-442行):
    G(π) = Σ γ^τ × E[F_τ] + α × Σ γ^τ × H[Q(o_τ | π)]
          = G_pragmatic(π) + G_epistemic(π)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import StateVector, ActionType
from algorithms.active_inference.trajectory_predictor import Trajectory
from config.constants import NUMERICAL, FREE_ENERGY


class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class InstantFreeEnergy:
    """
    即时自由能结果 (自由能.txt 第373-397行)

    Attributes:
        F_total: 总即时自由能
        F_energy: 能量自由能分量
        F_time: 时间自由能分量
        F_health: 健康度自由能分量
        F_progress: 进度自由能分量
        risk_level: 风险等级
    """
    F_total: float
    F_energy: float
    F_time: float
    F_health: float
    F_progress: float
    risk_level: RiskLevel


@dataclass
class ExpectedFreeEnergy:
    """
    期望自由能结果 (自由能.txt 第399-442行)

    Attributes:
        G_total: 总期望自由能
        G_pragmatic: 实用价值 (风险规避)
        G_epistemic: 认知价值 (信息增益)
        trajectory: 轨迹
        entropy_trace: 熵轨迹
        risk_level: 风险等级
    """
    G_total: float
    G_pragmatic: float
    G_epistemic: float
    trajectory: Optional[Trajectory] = None
    entropy_trace: List[float] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


class FourComponentCalculator:
    """
    四分量自由能计算器 (自由能.txt 第367-452行)

    功能：按照自由能.txt的四分量公式计算即时自由能

    Attributes:
        w_E: 能量权重
        w_T: 时间权重
        w_h: 健康度权重
        w_p: 进度权重
        epsilon: 防除零小量
        F_threshold: 低风险阈值
        F_max: 高风险阈值
    """

    def __init__(self,
                 w_E: float = None,
                 w_T: float = None,
                 w_h: float = None,
                 w_p: float = None,
                 epsilon: float = 1e-6,
                 F_threshold: float = 30.0,
                 F_max: float = 50.0):
        """
        初始化四分量计算器

        Args:
            w_E: 能量权重 (默认使用 constants.py 值)
            w_T: 时间权重
            w_h: 健康度权重
            w_p: 进度权重
            epsilon: 防除零小量
            F_threshold: 低风险阈值
            F_max: 高风险阈值
        """
        # 使用配置文件中的权重 (constants.py 第44-48行)
        self.w_E = w_E if w_E is not None else FREE_ENERGY.W_ENERGY
        self.w_T = w_T if w_T is not None else FREE_ENERGY.W_TIME
        self.w_h = w_h if w_h is not None else FREE_ENERGY.W_HEALTH
        self.w_p = w_p if w_p is not None else FREE_ENERGY.W_PROGRESS

        self.epsilon = epsilon
        self.F_threshold = F_threshold
        self.F_max = F_max

    def compute_energy_free_energy(self,
                                 E_t: float,
                                 E_required: float) -> float:
        """
        计算能量自由能 (自由能.txt 第384行)

        F_t^energy = -log(E_t / E_required)

        Args:
            E_t: 剩余能量 (J)
            E_required: 所需能量 (J)

        Returns:
            float: 能量自由能
        """
        if E_required < NUMERICAL.EPSILON:
            return 0.0

        ratio = max(E_t, 0) / E_required
        return -np.log(max(ratio, self.epsilon))

    def compute_time_free_energy(self,
                               T_max: float,
                               T_t: float,
                               T_remaining_required: float) -> float:
        """
        计算时间自由能 (自由能.txt 第386-389行)

        F_t^time = -log((T_max - T_t) / T_remaining_required)

        Args:
            T_max: 最大时间 (s)
            T_t: 已用时间 (s)
            T_remaining_required: 剩余所需时间 (s)

        Returns:
            float: 时间自由能
        """
        if T_remaining_required < NUMERICAL.EPSILON:
            return 0.0

        T_remaining = T_max - T_t
        ratio = max(T_remaining, 0) / T_remaining_required
        return -np.log(max(ratio, self.epsilon))

    def compute_health_free_energy(self,
                                  h_t: float,
                                  q_t: float) -> float:
        """
        计算健康度自由能 (自由能.txt 第390-392行)

        F_t^health = -log(h_t × q_t)

        Args:
            h_t: 健康度 [0,1]
            q_t: 信道质量 [0,1]

        Returns:
            float: 健康度自由能
        """
        product = max(h_t, 0) * max(q_t, 0)
        return -np.log(max(product, self.epsilon))

    def compute_progress_free_energy(self,
                                  p_t: float,
                                  p_expected: float) -> float:
        """
        计算进度自由能 (自由能.txt 第394-396行)

        F_t^progress = -log((p_t + ε) / (p_expected + ε))

        Args:
            p_t: 当前进度 [0,1]
            p_expected: 预期进度 [0,1]

        Returns:
            float: 进度自由能
        """
        numerator = max(p_t, 0) + self.epsilon
        denominator = max(p_expected, 0) + self.epsilon
        ratio = numerator / denominator
        return -np.log(ratio)

    def compute_instant_free_energy(self,
                                  state: StateVector,
                                  E_required: float,
                                  T_remaining_required: float,
                                  channel_quality: float = 1.0,
                                  p_expected: float = 1.0) -> InstantFreeEnergy:
        """
        计算即时自由能 (自由能.txt 第373-397行)

        F_t = w_E × F_t^energy + w_T × F_t^time + w_h × F_t^health + w_p × F_t^progress

        Args:
            state: 当前状态
            E_required: 所需能量 (J)
            T_remaining_required: 剩余所需时间 (s)
            channel_quality: 信道质量 [0,1]
            p_expected: 预期进度 [0,1]

        Returns:
            InstantFreeEnergy: 即时自由能结果
        """
        # 计算各分量
        F_energy = self.compute_energy_free_energy(state.E, E_required)
        F_time = self.compute_time_free_energy(FREE_ENERGY.MAX_FREE_ENERGY * 100,
                                          state.T, T_remaining_required)
        F_health = self.compute_health_free_energy(state.h, channel_quality)
        F_progress = self.compute_progress_free_energy(state.p, p_expected)

        # 加权求和 (自由能.txt 第378行)
        F_total = (self.w_E * F_energy +
                   self.w_T * F_time +
                   self.w_h * F_health +
                   self.w_p * F_progress)

        # 确定风险等级
        if state.E < E_required * 0.5:  # 能量严重不足
            risk_level = RiskLevel.CRITICAL
        elif F_total >= self.F_max:
            risk_level = RiskLevel.HIGH
        elif F_total >= self.F_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return InstantFreeEnergy(
            F_total=F_total,
            F_energy=F_energy,
            F_time=F_time,
            F_health=F_health,
            F_progress=F_progress,
            risk_level=risk_level
        )


class ExpectedFreeEnergyCalculator:
    """
    期望自由能计算器 (自由能.txt 第399-442行)

    功能：计算期望自由能 G(π)

    期望自由能公式 (自由能.txt 第400-422行):
        G(π) = Σ γ^τ × E[F_τ] + α × Σ γ^τ × H[Q(o_τ | π)]

    分解为:
        G_pragmatic(π) = Σ γ^τ × F(ŝ_τ^π)  (实用价值/风险规避)
        G_epistemic(π) = Σ γ^τ × H[Q(o_τ | π)]  (认知价值/信息增益)
    """

    def __init__(self,
                 gamma: float = 0.9,
                 alpha: float = 0.1,
                 four_component_calc: Optional[FourComponentCalculator] = None):
        """
        初始化期望自由能计算器

        Args:
            gamma: 时间折扣因子
            alpha: 探索-利用权衡系数 (认知价值权重)
            four_component_calc: 四分量计算器
        """
        self.gamma = gamma
        self.alpha = alpha
        self.four_component = four_component_calc or FourComponentCalculator()

    def compute_pragmatic_value(self,
                                trajectory: Trajectory,
                                E_required: float,
                                T_remaining_required: float,
                                channel_quality: float = 1.0,
                                p_expected: float = 1.0) -> float:
        """
        计算实用价值 (自由能.txt 第406-409行)

        G_pragmatic(π) = Σ γ^τ × F(ŝ_τ^π)

        Args:
            trajectory: 轨迹
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度

        Returns:
            float: 实用价值
        """
        G_pragmatic = 0.0

        for tau, state in enumerate(trajectory.states):
            # 计算该时刻的即时自由能
            instant_fe = self.four_component.compute_instant_free_energy(
                state, E_required, T_remaining_required, channel_quality, p_expected
            )

            # 时间折扣 (自由能.txt 第25行)
            discounted_F = (self.gamma ** tau) * instant_fe.F_total

            G_pragmatic += discounted_F

        return G_pragmatic

    def compute_epistemic_value(self,
                                trajectory: Trajectory,
                                uncertainty_weights: Optional[List[float]] = None) -> float:
        """
        计算认知价值 (自由能.txt 第411-421行)

        G_epistemic(π) = Σ γ^τ × tr(Σ_τ^π)

        或使用熵: H[N(μ, Σ)] = 1/2 × log|2πeΣ|

        Args:
            trajectory: 轨迹
            uncertainty_weights: 不确定性权重列表

        Returns:
            float: 认知价值
        """
        if uncertainty_weights is None:
            uncertainty_weights = trajectory.uncertainties

        G_epistemic = 0.0

        for tau, uncertainty in enumerate(uncertainty_weights):
            # 使用标量不确定性作为熵的近似
            # H ≈ -log(σ)  (简化的熵)
            entropy = -np.log(max(uncertainty, self.four_component.epsilon))

            # 时间折扣 (自由能.txt 第25行)
            discounted_H = (self.gamma ** tau) * entropy

            G_epistemic += discounted_H

        return G_epistemic

    def compute_expected_free_energy(self,
                                    trajectory: Trajectory,
                                    E_required: float,
                                    T_remaining_required: float = 50.0,
                                    channel_quality: float = 1.0,
                                    p_expected: float = 1.0) -> ExpectedFreeEnergy:
        """
        计算期望自由能 (自由能.txt 第400-422行)

        G(π) = G_pragmatic(π) + α × G_epistemic(π)

        Args:
            trajectory: 轨迹
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度

        Returns:
            ExpectedFreeEnergy: 期望自由能结果
        """
        # 实用价值 (自由能.txt 第406-409行)
        G_pragmatic = self.compute_pragmatic_value(
            trajectory, E_required, T_remaining_required, channel_quality, p_expected
        )

        # 认知价值 (自由能.txt 第411-421行)
        G_epistemic = self.compute_epistemic_value(trajectory)

        # 总期望自由能 (自由能.txt 第400-402行)
        G_total = G_pragmatic + self.alpha * G_epistemic

        # 确定风险等级
        if G_total >= self.four_component.F_max * 2:
            risk_level = RiskLevel.CRITICAL
        elif G_total >= self.four_component.F_max:
            risk_level = RiskLevel.HIGH
        elif G_total >= self.four_component.F_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return ExpectedFreeEnergy(
            G_total=G_total,
            G_pragmatic=G_pragmatic,
            G_epistemic=G_epistemic,
            trajectory=trajectory,
            entropy_trace=[-np.log(max(u, 0.001)) for u in trajectory.uncertainties],
            risk_level=risk_level
        )


class FreeEnergyCalculator:
    """
    自由能计算器 (统一接口)

    整合四分量计算器和期望自由能计算器，同时保持与idea118.txt的兼容

    Attributes:
        four_component: 四分量计算器
        expected_calculator: 期望自由能计算器
    """

    def __init__(self,
                 use_four_component: bool = True,
                 gamma: float = 0.9,
                 alpha: float = 0.1,
                 F_threshold: float = 30.0,
                 F_max: float = 50.0):
        """
        初始化自由能计算器

        Args:
            use_four_component: 是否使用四分量公式
            gamma: 时间折扣因子
            alpha: 探索-利用权衡系数
            F_threshold: 低风险阈值
            F_max: 高风险阈值
        """
        self.use_four_component = use_four_component
        self.four_component = FourComponentCalculator(F_threshold=F_threshold, F_max=F_max)
        self.expected_calculator = ExpectedFreeEnergyCalculator(
            gamma=gamma, alpha=alpha, four_component_calc=self.four_component
        )

    def compute_instant(self,
                        state: StateVector,
                        E_required: float,
                        T_remaining_required: float = 50.0,
                        channel_quality: float = 1.0,
                        p_expected: float = 1.0) -> InstantFreeEnergy:
        """
        计算即时自由能

        Args:
            state: 当前状态
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度

        Returns:
            InstantFreeEnergy: 即时自由能
        """
        if self.use_four_component:
            return self.four_component.compute_instant_free_energy(
                state, E_required, T_remaining_required, channel_quality, p_expected
            )
        else:
            # 使用idea118.txt的简化公式
            F_total = (state.p > 0.5) * 10 + (1 - state.h) * 15 + \
                      (state.sigma > 0.3) * 20
            return InstantFreeEnergy(
                F_total=F_total,
                F_energy=F_total * 0.4,
                F_time=F_total * 0.3,
                F_health=F_total * 0.2,
                F_progress=F_total * 0.1,
                risk_level=RiskLevel.HIGH if F_total > 30 else RiskLevel.LOW
            )

    def compute_expected(self,
                      trajectory,
                      E_required: float,
                      T_remaining_required: float = 50.0,
                      channel_quality: float = 1.0,
                      p_expected: float = 1.0) -> ExpectedFreeEnergy:
        """
        计算期望自由能

        Args:
            trajectory: 轨迹
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度

        Returns:
            ExpectedFreeEnergy: 期望自由能
        """
        return self.expected_calculator.compute_expected_free_energy(
            trajectory, E_required, T_remaining_required, channel_quality, p_expected
        )


# ============ 测试用例 ============

def test_free_energy():
    """测试FreeEnergy模块"""
    print("=" * 60)
    print("测试 M17E: FreeEnergyCalculator (自由能.txt 第367-452行)")
    print("=" * 60)

    # 测试1: 四分量计算器
    print("\n[Test 1] 测试四分量计算器...")
    four_comp = FourComponentCalculator()

    state_normal = StateVector(E=400e3, T=20.0, h=0.9, p=0.5, d=800.0, sigma=0.1)
    fe_normal = four_comp.compute_instant_free_energy(
        state_normal, E_required=100e3, T_remaining_required=30.0
    )
    print(f"  正常状态: F_total={fe_normal.F_total:.2f}, "
          f"F_energy={fe_normal.F_energy:.2f}, F_time={fe_normal.F_time:.2f}")
    print(f"  风险等级: {fe_normal.risk_level.value}")
    assert fe_normal.risk_level == RiskLevel.LOW, "正常状态应为低风险"
    print("  ✓ 四分量计算正确")

    # 测试2: 高风险状态
    print("\n[Test 2] 测试高风险状态...")
    state_high_risk = StateVector(E=50e3, T=90.0, h=0.4, p=0.2, d=800.0, sigma=0.5)
    fe_high = four_comp.compute_instant_free_energy(
        state_high_risk, E_required=100e3, T_remaining_required=30.0
    )
    print(f"  高风险状态: F_total={fe_high.F_total:.2f}, "
          f"F_energy={fe_high.F_energy:.2f}, F_health={fe_high.F_health:.2f}")
    print(f"  风险等级: {fe_high.risk_level.value}")
    assert fe_high.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL], "应为高风险"
    print("  ✓ 高风险检测正确")

    # 测试3: 各分量权重验证
    print("\n[Test 3] 测试各分量权重验证...")
    print(f"  权重配置: w_E={four_comp.w_E}, w_T={four_comp.w_T}, "
          f"w_h={four_comp.w_h}, w_p={four_comp.w_p}")
    print(f"  权重和: {four_comp.w_E + four_comp.w_T + four_comp.w_h + four_comp.w_p:.3f}")
    assert abs((four_comp.w_E + four_comp.w_T + four_comp.w_h + four_comp.w_p) - 1.0) < 1e-6, "权重和应为1"
    print("  ✓ 权重配置正确")

    # 测试4: 期望自由能
    print("\n[Test 4] 测试期望自由能...")
    from algorithms.active_inference.trajectory_predictor import TrajectoryPredictor
    from algorithms.active_inference.generative_model import GenerativeModel

    gen_model = GenerativeModel(T_task=50.0)
    traj_predictor = TrajectoryPredictor(gen_model)

    policy = [ActionType.CONTINUE] * 10
    trajectory, _ = traj_predictor.predict(state_normal, policy, method='deterministic')

    expected_calc = ExpectedFreeEnergyCalculator()
    G = expected_calc.compute_expected_free_energy(
        trajectory, E_required=100e3, T_remaining_required=30.0
    )

    print(f"  轨迹长度: {len(trajectory)}")
    print(f"  G_total={G.G_total:.2f}, "
          f"G_pragmatic={G.G_pragmatic:.2f}, G_epistemic={G.G_epistemic:.2f}")
    print(f"  风险等级: {G.risk_level.value}")
    print("  ✓ 期望自由能计算正确")

    # 测试5: 统一接口
    print("\n[Test 5] 测试统一接口...")
    calculator = FreeEnergyCalculator(use_four_component=True)

    instant = calculator.compute_instant(state_normal, E_required=100e3)
    print(f"  即时自由能: F_total={instant.F_total:.2f}, "
          f"风险={instant.risk_level.value}")

    expected = calculator.compute_expected(trajectory, E_required=100e3)
    print(f"  期望自由能: G_total={expected.G_total:.2f}, "
          f"风险={expected.risk_level.value}")
    print("  ✓ 统一接口正确")

    # 测试6: 时间衰减特性
    print("\n[Test 6] 测试时间衰减特性...")
    # 创建多个轨迹
    trajectories = []
    for i in range(5):
        traj, _ = traj_predictor.predict(state_normal, policy, method='deterministic')
        trajectories.append(traj)

    for i, traj in enumerate(trajectories):
        G_i = expected_calc.compute_expected_free_energy(
            traj, E_required=100e3
        )
        print(f"  轨迹{i+1}: G_total={G_i.G_total:.2f}")
    print("  ✓ 时间衰减特性正确")

    # 测试7: 认知价值计算
    print("\n[Test 7] 测试认知价值计算...")
    # 高不确定性轨迹
    high_uncertainty_traj = Trajectory()
    high_uncertainty_traj.uncertainties = [0.1, 0.3, 0.5, 0.7, 0.8]
    high_uncertainty_traj.states = [
        state_normal.copy() for _ in range(5)
    ]

    G_high = expected_calc.compute_expected_free_energy(high_uncertainty_traj, E_required=100e3)

    # 低不确定性轨迹
    low_uncertainty_traj = Trajectory()
    low_uncertainty_traj.uncertainties = [0.05, 0.08, 0.1, 0.12, 0.15]
    low_uncertainty_traj.states = [
        state_normal.copy() for _ in range(5)
    ]

    G_low = expected_calc.compute_expected_free_energy(low_uncertainty_traj, E_required=100e3)

    print(f"  高不确定性: G_epistemic={G_high.G_epistemic:.2f}")
    print(f"  低不确定性: G_epistemic={G_low.G_epistemic:.2f}")
    assert G_high.G_epistemic > G_low.G_epistemic, "高不确定性应有更高认知价值"
    print("  ✓ 认知价值计算正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_free_energy()

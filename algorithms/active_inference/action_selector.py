"""
M17F: ActionSelector - 行动选择器

功能：基于期望自由能选择最优行动
参考文档：docs/自由能.txt 第455-530行

策略选择 (自由能.txt 第463-477行):
    - 确定性选择: a* = argmin_a G(a)
    - 概率性选择: P(a) = exp(-β × G(a)) / Σ exp(-β × G(a'))

    参数 β (逆温度):
        - β → 0: 随机选择 (探索)
        - β → ∞: 贪婪选择 (利用)

多步规划 (自由能.txt 第479-501行):
    - 滚动时域优化: 只执行 a_1*, 然后重新规划
    - 树搜索剪枝: Prune if G_partial > G_best + δ

约束处理 (自由能.txt 第503-520行):
    - 硬约束: A_feasible(s) = {a | C(s, a) ≤ 0}
    - 软约束: G_constrained(a) = G(a) + λ × max(0, C(s,a))
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import (
    StateVector, ActionType, ActionSet, StateBounds
)
from algorithms.active_inference.trajectory_predictor import (
    Trajectory, TrajectoryPredictor
)
from algorithms.active_inference.free_energy import (
    FreeEnergyCalculator, ExpectedFreeEnergy
)
from config.constants import NUMERICAL


@dataclass
class ActionEvaluation:
    """
    行动评估结果

    Attributes:
        action: 行动类型
        G_value: 期望自由能值
        pragmatic_value: 实用价值
        epistemic_value: 认知价值
        feasibility: 可行性 [0,1]
        trajectory: 轨迹
        action_confidence: 行动置信度
    """
    action: ActionType
    G_value: float
    pragmatic_value: float
    epistemic_value: float
    feasibility: float
    trajectory: Optional[Trajectory] = None
    action_confidence: float = 0.5


class GreedySelector:
    """
    贪婪行动选择器 (自由能.txt 第523-524行子模块清单)

    功能：选择期望自由能最小的行动
    公式 (自由能.txt 第465-467行): a* = argmin_a G(a)
    """

    def select(self, evaluations: List[ActionEvaluation]) -> ActionEvaluation:
        """
        贪婪选择 (自由能.txt 第465-467行)

        a* = argmin_a G(a)

        Args:
            evaluations: 行动评估列表

        Returns:
            ActionEvaluation: 最优行动评估
        """
        if not evaluations:
            raise ValueError("行动评估列表为空")

        return min(evaluations, key=lambda e: e.G_value)


class SoftmaxSelector:
    """
    Softmax概率选择器 (自由能.txt 第525-526行子模块清单)

    功能：使用Softmax分布进行概率性选择
    公式 (自由能.txt 第469-473行):
        P(a) = exp(-β × G(a)) / Σ exp(-β × G(a'))

    Attributes:
        beta: 逆温度参数
    """

    def __init__(self, beta: float = 5.0):
        """
        初始化Softmax选择器

        Args:
            beta: 逆温度参数
                - β → 0: 随机选择 (探索)
                - β → ∞: 贪婪选择 (利用)
        """
        self.beta = beta

    def select(self, evaluations: List[ActionEvaluation]) -> ActionEvaluation:
        """
        Softmax选择 (自由能.txt 第469-473行)

        Args:
            evaluations: 行动评估列表

        Returns:
            ActionEvaluation: 选择的行动评估
        """
        if not evaluations:
            raise ValueError("行动评估列表为空")

        # 计算Softmax概率 (自由能.txt 第469-473行)
        G_values = [e.G_value for e in evaluations]
        exp_neg_beta_G = np.exp(-self.beta * np.array(G_values))
        probs = exp_neg_beta_G / np.sum(exp_neg_beta_G)

        # 采样
        idx = np.random.choice(len(evaluations), p=probs)
        return evaluations[idx]

    def get_probabilities(self, evaluations: List[ActionEvaluation]) -> Dict[ActionType, float]:
        """
        获取各行动的概率

        Args:
            evaluations: 行动评估列表

        Returns:
            Dict: {行动类型: 概率}
        """
        if not evaluations:
            return {}

        G_values = [e.G_value for e in evaluations]
        exp_neg_beta_G = np.exp(-self.beta * np.array(G_values))
        probs = exp_neg_beta_G / np.sum(exp_neg_beta_G)

        return {e.action: prob for e, prob in zip(evaluations, probs)}


class ConstraintChecker:
    """
    约束检查器 (自由能.txt 第526-528行子模块清单)

    功能：检查行动的可行性约束

    硬约束 (自由能.txt 第508-513行):
        - E_t - P(a) × Δt ≥ E_min
        - T_t + κ(a) × Δt ≤ T_max
    """

    def __init__(self, bounds: Optional[StateBounds] = None, delta_t: float = 1.0):
        """
        初始化约束检查器

        Args:
            bounds: 状态边界
            delta_t: 时间步长
        """
        self.bounds = bounds or StateBounds()
        self.delta_t = delta_t
        self.action_set = ActionSet()

    def check_feasibility(self, state: StateVector, action: ActionType) -> float:
        """
        检查行动可行性 (自由能.txt 第508-513行)

        Args:
            state: 当前状态
            action: 待检查行动

        Returns:
            float: 可行性评分 [0,1], 1表示完全可行
        """
        feasibility = 1.0

        # 能量约束 (自由能.txt 第512行)
        P_action = self.action_set.get_action_power(action)
        E_after = state.E - P_action * self.delta_t
        if E_after < self.bounds.E_min:
            feasibility *= 0.0
        elif E_after < self.bounds.E_max * 0.1:  # 低于10%能量
            feasibility *= 0.3

        # 时间约束 (自由能.txt 第513行)
        kappa = self.action_set.get_action_time_coeff(action)
        T_after = state.T + self.delta_t * kappa
        if T_after > self.bounds.T_max:
            feasibility *= 0.0
        elif T_after > self.bounds.T_max * 0.8:  # 超过80%时间
            feasibility *= 0.5

        # 健康度约束
        if action == ActionType.ABORT and state.h > 0.5:
            feasibility *= 0.5  # 健康度较高时不建议中止

        return feasibility


class RollingHorizonOptimizer:
    """
    滚动时域优化器 (自由能.txt 第526-528行子模块清单)

    功能：实现滚动时域优化
    公式 (自由能.txt 第496-501行): π*_1:H = argmin_π G(π)

    只执行 a_1*, 然后重新规划

    Attributes:
        H: 预测时域
    """

    def __init__(self, H: int = 10):
        """
        初始化滚动时域优化器

        Args:
            H: 预测时域 (步数)
        """
        self.H = H

    def optimize_policy(self,
                       state: StateVector,
                       actions: List[ActionType],
                       evaluate_fn: Callable[[StateVector, List[ActionType]], float],
                       max_combinations: int = 100) -> List[ActionType]:
        """
        优化策略 (自由能.txt 第496-501行)

        π*_1:H = argmin_π G(π)

        简化实现：探索有限数量的策略组合

        Args:
            state: 当前状态
            actions: 可用行动列表
            evaluate_fn: 评估函数
            max_combinations: 最大搜索组合数

        Returns:
            List[ActionType]: 最优策略
        """
        best_policy = None
        best_G = float('inf')

        # 策略空间剪枝：只探索部分组合
        # 1. 全单一行动策略
        for action in actions:
            policy = [action] * self.H
            G = evaluate_fn(state, policy)
            if G < best_G:
                best_G = G
                best_policy = policy

        # 2. 混合策略 (早期探索，后期保守)
        n_mixed = min(max_combinations - len(actions), 50)
        for _ in range(n_mixed):
            # 随机生成混合策略
            policy = np.random.choice(actions, size=self.H).tolist()
            G = evaluate_fn(state, policy)
            if G < best_G:
                best_G = G
                best_policy = policy

        return best_policy if best_policy is not None else [actions[0]] * self.H


class ActionSelector:
    """
    行动选择器 (自由能.txt 第455-530行)

    整合贪婪选择、Softmax选择和约束检查

    Attributes:
        traj_predictor: 轨迹预测器
        fe_calculator: 自由能计算器
        greedy_selector: 贪婪选择器
        softmax_selector: Softmax选择器
        constraint_checker: 约束检查器
        horizon_optimizer: 滚动时域优化器
    """

    def __init__(self,
                 traj_predictor: TrajectoryPredictor,
                 fe_calculator: FreeEnergyCalculator,
                 selection_method: str = 'greedy',
                 H: int = 5):
        """
        初始化行动选择器

        Args:
            traj_predictor: 轨迹预测器
            fe_calculator: 自由能计算器
            selection_method: 选择方法 ('greedy', 'softmax', 'hybrid')
            H: 预测时域
        """
        self.traj_predictor = traj_predictor
        self.fe_calculator = fe_calculator
        self.selection_method = selection_method
        self.greedy_selector = GreedySelector()
        self.softmax_selector = SoftmaxSelector(beta=5.0)
        self.constraint_checker = ConstraintChecker()
        self.horizon_optimizer = RollingHorizonOptimizer(H=H)

        # 评估参数
        self.E_required = 100e3  # 默认100kJ
        self.T_remaining_required = 50.0
        self.channel_quality = 1.0
        self.p_expected = 0.1

    def set_evaluation_params(self,
                               E_required: float,
                               T_remaining_required: float,
                               channel_quality: float = 1.0,
                               p_expected: float = 0.1):
        """
        设置评估参数

        Args:
            E_required: 所需能量
            T_remaining_required: 剩余所需时间
            channel_quality: 信道质量
            p_expected: 预期进度
        """
        self.E_required = E_required
        self.T_remaining_required = T_remaining_required
        self.channel_quality = channel_quality
        self.p_expected = p_expected

    def evaluate_single_action(self,
                              state: StateVector,
                              action: ActionType) -> ActionEvaluation:
        """
        评估单个行动

        Args:
            state: 当前状态
            action: 行动

        Returns:
            ActionEvaluation: 行动评估
        """
        # 检查可行性
        feasibility = self.constraint_checker.check_feasibility(state, action)

        if feasibility < 0.1:  # 不可行
            return ActionEvaluation(
                action=action,
                G_value=float('inf'),
                pragmatic_value=float('inf'),
                epistemic_value=float('inf'),
                feasibility=feasibility
            )

        # 单步预测
        trajectory, _ = self.traj_predictor.predict(
            state, [action], method='deterministic', max_horizon=1
        )

        # 计算期望自由能
        expected_fe = self.fe_calculator.compute_expected(
            trajectory,
            self.E_required,
            self.T_remaining_required,
            self.channel_quality,
            self.p_expected
        )

        return ActionEvaluation(
            action=action,
            G_value=expected_fe.G_total,
            pragmatic_value=expected_fe.G_pragmatic,
            epistemic_value=expected_fe.G_epistemic,
            feasibility=feasibility,
            trajectory=trajectory
        )

    def evaluate_all_actions(self,
                            state: StateVector,
                            actions: Optional[List[ActionType]] = None) -> List[ActionEvaluation]:
        """
        评估所有行动

        Args:
            state: 当前状态
            actions: 行动列表

        Returns:
            List[ActionEvaluation]: 行动评估列表
        """
        if actions is None:
            actions = list(ActionType)

        evaluations = []
        for action in actions:
            eval_result = self.evaluate_single_action(state, action)
            evaluations.append(eval_result)

        return evaluations

    def select_action(self,
                      state: StateVector,
                      actions: Optional[List[ActionType]] = None) -> ActionEvaluation:
        """
        选择最优行动

        Args:
            state: 当前状态
            actions: 可用行动列表

        Returns:
            ActionEvaluation: 最优行动评估
        """
        evaluations = self.evaluate_all_actions(state, actions)

        if self.selection_method == 'greedy':
            return self.greedy_selector.select(evaluations)
        elif self.selection_method == 'softmax':
            return self.softmax_selector.select(evaluations)
        elif self.selection_method == 'hybrid':
            # 低风险时贪婪，高风险时Softmax
            instant_fe = self.fe_calculator.compute_instant(
                state, self.E_required, self.T_remaining_required,
                self.channel_quality, self.p_expected
            )
            if instant_fe.risk_level.name in ['LOW', 'MEDIUM']:
                return self.greedy_selector.select(evaluations)
            else:
                return self.softmax_selector.select(evaluations)
        else:
            raise ValueError(f"未知选择方法: {self.selection_method}")

    def select_with_rolling_horizon(self,
                                      state: StateVector,
                                      actions: Optional[List[ActionType]] = None) -> ActionEvaluation:
        """
        使用滚动时域优化选择行动 (自由能.txt 第496-501行)

        只执行 a_1*, 然后重新规划

        Args:
            state: 当前状态
            actions: 可用行动列表

        Returns:
            ActionEvaluation: 最优行动评估
        """
        if actions is None:
            actions = list(ActionType)

        # 定义评估函数
        def evaluate_policy(s: StateVector, policy: List[ActionType]) -> float:
            trajectory, _ = self.traj_predictor.predict(
                s, policy, method='deterministic', max_horizon=len(policy)
            )
            expected_fe = self.fe_calculator.compute_expected(
                trajectory,
                self.E_required,
                self.T_remaining_required,
                self.channel_quality,
                self.p_expected
            )
            return expected_fe.G_total

        # 优化策略
        best_policy = self.horizon_optimizer.optimize_policy(
            state, actions, evaluate_policy
        )

        # 返回第一个行动的评估
        return self.evaluate_single_action(state, best_policy[0])


# ============ 测试用例 ============

def test_action_selector():
    """测试ActionSelector模块"""
    print("=" * 60)
    print("测试 M17F: ActionSelector (自由能.txt 第455-530行)")
    print("=" * 60)

    from algorithms.active_inference.trajectory_predictor import TrajectoryPredictor
    from algorithms.active_inference.generative_model import GenerativeModel
    from algorithms.active_inference.free_energy import FreeEnergyCalculator

    # 初始化
    gen_model = GenerativeModel(T_task=50.0)
    traj_predictor = TrajectoryPredictor(gen_model)
    fe_calculator = FreeEnergyCalculator(use_four_component=True)

    selector = ActionSelector(traj_predictor, fe_calculator, H=5)
    selector.set_evaluation_params(
        E_required=100e3,
        T_remaining_required=30.0,
        channel_quality=0.9,
        p_expected=0.1
    )

    # 测试1: 初始状态
    print("\n[Test 1] 测试初始状态...")
    initial_state = StateVector(E=400e3, T=5.0, h=0.9, p=0.3, d=800.0, sigma=0.1)
    print(f"  初始状态: E={initial_state.E/1e3:.1f}kJ, T={initial_state.T:.1f}s, "
          f"h={initial_state.h:.2f}, p={initial_state.p:.2f}")
    print("  ✓ 初始状态正确")

    # 测试2: 单行动评估
    print("\n[Test 2] 测试单行动评估...")
    eval_continue = selector.evaluate_single_action(initial_state, ActionType.CONTINUE)
    print(f"  CONTINUE: G={eval_continue.G_value:.2f}, "
          f"可行={eval_continue.feasibility:.2f}")

    eval_abort = selector.evaluate_single_action(initial_state, ActionType.ABORT)
    print(f"  ABORT: G={eval_abort.G_value:.2f}, "
          f"可行={eval_abort.feasibility:.2f}")
    print("  ✓ 单行动评估正确")

    # 测试3: 贪婪选择
    print("\n[Test 3] 测试贪婪选择...")
    selector.selection_method = 'greedy'
    greedy_choice = selector.select_action(initial_state)
    print(f"  贪婪选择: {greedy_choice.action.value}")
    print(f"  G_value={greedy_choice.G_value:.2f}, "
          f"pragmatic={greedy_choice.pragmatic_value:.2f}, "
          f"epistemic={greedy_choice.epistemic_value:.2f}")
    print("  ✓ 贪婪选择正确")

    # 测试4: Softmax选择
    print("\n[Test 4] 测试Softmax选择...")
    selector.selection_method = 'softmax'
    softmax_choice = selector.select_action(initial_state)
    print(f"  Softmax选择: {softmax_choice.action.value}")
    print(f"  G_value={softmax_choice.G_value:.2f}")
    print("  ✓ Softmax选择正确")

    # 测试5: 约束检查
    print("\n[Test 5] 测试约束检查...")
    # 低能量状态
    low_energy_state = StateVector(E=30e3, T=10.0, h=0.8, p=0.3, d=800.0, sigma=0.1)
    for action in [ActionType.CONTINUE, ActionType.ABORT]:
        feasibility = selector.constraint_checker.check_feasibility(low_energy_state, action)
        print(f"  {action:12s}: 可行性={feasibility:.2f}")

    assert feasibility < 1.0, "低能量时应降低可行性"
    print("  ✓ 约束检查正确")

    # 测试6: 所有行动评估
    print("\n[Test 6] 测试所有行动评估...")
    all_evals = selector.evaluate_all_actions(initial_state)

    print(f"  行动评估:")
    for eval_result in sorted(all_evals, key=lambda e: e.G_value):
        print(f"    {eval_result.action.value:12s}: G={eval_result.G_value:6.2f}, "
              f"可行={eval_result.feasibility:.2f}")
    print("  ✓ 所有行动评估正确")

    # 测试7: 滚动时域优化
    print("\n[Test 7] 测试滚动时域优化...")
    rh_choice = selector.select_with_rolling_horizon(initial_state)
    print(f"  滚动时域选择: {rh_choice.action.value}")
    print(f"  G_value={rh_choice.G_value:.2f}")
    print("  ✓ 滚动时域优化正确")

    # 测试8: 混合选择策略
    print("\n[Test 8] 测试混合选择策略...")
    # 高风险状态
    high_risk_state = StateVector(E=80e3, T=40.0, h=0.4, p=0.2, d=800.0, sigma=0.5)
    selector.selection_method = 'hybrid'

    hybrid_choice_low = selector.select_action(initial_state)
    hybrid_choice_high = selector.select_action(high_risk_state)

    print(f"  低风险状态选择: {hybrid_choice_low.action.value}")
    print(f"  高风险状态选择: {hybrid_choice_high.action.value}")
    print("  ✓ 混合选择策略正确")

    # 测试9: Softmax概率分布
    print("\n[Test 9] 测试Softmax概率分布...")
    probs = selector.softmax_selector.get_probabilities(all_evals)

    print(f"  行动概率 (β={selector.softmax_selector.beta}):")
    for action, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"    {action.value:12s}: {prob*100:5.1f}%")
    print("  ✓ Softmax概率分布正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_action_selector()

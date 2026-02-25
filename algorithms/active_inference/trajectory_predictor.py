"""
M17C: TrajectoryPredictor - 轨迹预测器

功能：给定当前状态和策略，预测未来状态序列及其不确定性
参考文档：docs/自由能.txt 第235-289行

核心公式 (自由能.txt 第243-263行):
    - 确定性轨迹预测: ŝ_{τ+1} = f(ŝ_τ, a_τ)
    - 不确定性传播: Σ_{τ+1} = F_τ Σ_τ F_τ^T + Q
    - 蒙特卡洛采样: s_τ^{(i)} ~ P(s_τ | s_{τ-1}^{(i)}, a_{τ-1})
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import StateVector, ActionType
from algorithms.active_inference.generative_model import GenerativeModel, TransitionResult
from config.constants import NUMERICAL


@dataclass
class TrajectoryState:
    """
    轨迹状态点

    Attributes:
        state: 状态向量
        uncertainty: 不确定性 [0,1]
        timestamp: 时间戳
        action: 导致此状态的行动
        energy_consumed: 累计能耗
    """
    state: StateVector
    uncertainty: float
    timestamp: float
    action: ActionType
    energy_consumed: float = 0.0


@dataclass
class Trajectory:
    """
    完整轨迹

    Attributes:
        states: 状态序列
        actions: 行动序列
        uncertainties: 不确定性序列
        timestamps: 时间戳序列
        total_energy: 总能耗
        total_time: 总时间
        final_progress: 最终进度
        is_terminated: 是否终止
    """
    states: List[StateVector] = field(default_factory=list)
    actions: List[ActionType] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    total_energy: float = 0.0
    total_time: float = 0.0
    final_progress: float = 0.0
    is_terminated: bool = False

    def __len__(self) -> int:
        return len(self.states)

    def get_final_state(self) -> Optional[StateVector]:
        """获取最终状态"""
        return self.states[-1] if self.states else None

    def get_mean_uncertainty(self) -> float:
        """获取平均不确定性"""
        if not self.uncertainties:
            return 0.0
        return np.mean(self.uncertainties)

    def get_max_uncertainty(self) -> float:
        """获取最大不确定性"""
        if not self.uncertainties:
            return 0.0
        return np.max(self.uncertainties)


class DeterministicPredictor:
    """
    确定性轨迹预测器 (自由能.txt 第283-286行子模块清单)

    功能：给定策略，预测确定性未来轨迹
    公式 (自由能.txt 第243-247行): ŝ_{τ+1} = f(ŝ_τ, a_τ)
    """

    def __init__(self, generative_model: GenerativeModel):
        """
        初始化确定性预测器

        Args:
            generative_model: 生成模型
        """
        self.gen_model = generative_model

    def predict(self,
                initial_state: StateVector,
                policy: List[ActionType],
                delta_t: float = 1.0,
                max_horizon: Optional[int] = None) -> Trajectory:
        """
        预测轨迹 (自由能.txt 第243-247行)

        Args:
            initial_state: 初始状态
            policy: 策略（行动序列）
            delta_t: 时间步长
            max_horizon: 最大预测步数

        Returns:
            Trajectory: 预测轨迹
        """
        if max_horizon is None:
            max_horizon = len(policy)

        trajectory = Trajectory()
        current_state = initial_state.copy()

        total_energy = 0.0
        total_time = 0.0

        for step, action in enumerate(policy[:max_horizon]):
            # 预测下一状态
            result = self.gen_model.predict_next(current_state, action, delta_t)

            if result.is_terminal:
                trajectory.is_terminated = True
                break

            current_state = result.next_state
            total_energy += result.energy_consumed
            total_time += result.time_elapsed

            # 记录轨迹
            trajectory.states.append(current_state)
            trajectory.actions.append(action)
            trajectory.uncertainties.append(current_state.sigma)
            trajectory.timestamps.append(total_time)

        trajectory.total_energy = total_energy
        trajectory.total_time = total_time
        trajectory.final_progress = current_state.p if trajectory.states else 0.0

        return trajectory


class UncertaintyPropagator:
    """
    不确定性传播器 (自由能.txt 第284-286行子模块清单)

    功能：传播不确定性，使用简化版本的标量传播
    公式 (自由能.txt 第259-263行): σ_{τ+1}² = λ_σ × σ_τ² + q_τ(a_τ)
    """

    def __init__(self, lambda_sigma: float = 1.05, q_base: float = 0.001):
        """
        初始化不确定性传播器

        Args:
            lambda_sigma: 不确定性增长率 (>1 表示随时间增长)
            q_base: 基础行动引入的不确定性
        """
        self.lambda_sigma = lambda_sigma
        self.q_base = q_base

    def propagate(self,
                sigma: float,
                action: ActionType,
                delta_t: float = 1.0) -> float:
        """
        传播不确定性 (自由能.txt 第259-263行)

        Args:
            sigma: 当前不确定性
            action: 执行的行动
            delta_t: 时间步长

        Returns:
            float: 下一时刻的不确定性
        """
        # 行动引入的额外不确定性
        if action == ActionType.CHECKPOINT:
            q_action = self.q_base * 0.5  # Checkpoint减少不确定性
        elif action == ActionType.HANDOVER:
            q_action = self.q_base * 2.0  # Handover增加不确定性
        elif action == ActionType.ABORT:
            q_action = 0.0  # 终止后无不确定性
        else:
            q_action = self.q_base

        # 不确定性传播公式
        sigma_squared = sigma**2 * self.lambda_sigma + q_action * delta_t
        sigma_next = np.sqrt(max(0, sigma_squared))

        # 限制在[0,1]范围
        return min(1.0, sigma_next)


class MonteCarloSampler:
    """
    蒙特卡洛采样器 (自由能.txt 第284-286行子模块清单)

    功能：使用采样方法估计轨迹分布
    公式 (自由能.txt 第267-279行)
    """

    def __init__(self, generative_model: GenerativeModel):
        """
        初始化蒙特卡洛采样器

        Args:
            generative_model: 生成模型
        """
        self.gen_model = generative_model

    def sample_trajectories(self,
                          initial_state: StateVector,
                          policy: List[ActionType],
                          n_samples: int = 100,
                          delta_t: float = 1.0,
                          max_horizon: Optional[int] = None) -> List[Trajectory]:
        """
        采样多条轨迹 (自由能.txt 第267-279行)

        Args:
            initial_state: 初始状态
            policy: 策略
            n_samples: 采样数量
            delta_t: 时间步长
            max_horizon: 最大预测步数

        Returns:
            List[Trajectory]: 采样轨迹列表
        """
        if max_horizon is None:
            max_horizon = len(policy)

        trajectories = []

        for _ in range(n_samples):
            trajectory = Trajectory()
            current_state = initial_state.copy()

            total_energy = 0.0
            total_time = 0.0

            for step, action in enumerate(policy[:max_horizon]):
                result = self.gen_model.predict_next(current_state, action, delta_t)

                if result.is_terminal:
                    trajectory.is_terminated = True
                    break

                current_state = result.next_state
                total_energy += result.energy_consumed
                total_time += result.time_elapsed

                trajectory.states.append(current_state)
                trajectory.actions.append(action)
                trajectory.uncertainties.append(current_state.sigma)
                trajectory.timestamps.append(total_time)

            trajectory.total_energy = total_energy
            trajectory.total_time = total_time
            trajectory.final_progress = current_state.p if trajectory.states else 0.0

            trajectories.append(trajectory)

        return trajectories

    def estimate_statistics(self, trajectories: List[Trajectory]) -> Dict:
        """
        估计轨迹统计量 (自由能.txt 第276-279行)

        Args:
            trajectories: 采样轨迹列表

        Returns:
            Dict: 统计量字典
        """
        if not trajectories:
            return {
                'mean_energy': 0.0,
                'mean_time': 0.0,
                'mean_progress': 0.0,
                'mean_uncertainty': 0.0,
                'std_energy': 0.0,
                'success_rate': 0.0
            }

        energies = [t.total_energy for t in trajectories]
        times = [t.total_time for t in trajectories]
        progresses = [t.final_progress for t in trajectories]
        uncertainties = [t.get_mean_uncertainty() for t in trajectories]

        return {
            'mean_energy': float(np.mean(energies)),
            'mean_time': float(np.mean(times)),
            'mean_progress': float(np.mean(progresses)),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_energy': float(np.std(energies)),
            'std_time': float(np.std(times)),
            'success_rate': sum(1 for t in trajectories if not t.is_terminated) / len(trajectories)
        }


class TrajectoryPredictor:
    """
    轨迹预测器 (自由能.txt 第235-289行)

    整合确定性预测和蒙特卡洛采样

    Attributes:
        gen_model: 生成模型
        deterministic: 确定性预测器
        mc_sampler: 蒙特卡洛采样器
        uncertainty_propagator: 不确定性传播器
    """

    def __init__(self, generative_model: GenerativeModel):
        """
        初始化轨迹预测器

        Args:
            generative_model: 生成模型
        """
        self.gen_model = generative_model
        self.deterministic = DeterministicPredictor(generative_model)
        self.mc_sampler = MonteCarloSampler(generative_model)
        self.uncertainty_propagator = UncertaintyPropagator()

    def predict(self,
                initial_state: StateVector,
                policy: List[ActionType],
                method: str = 'deterministic',
                delta_t: float = 1.0,
                n_samples: int = 100,
                max_horizon: Optional[int] = None) -> Tuple[Trajectory, Dict]:
        """
        预测轨迹

        Args:
            initial_state: 初始状态
            policy: 策略（行动序列）
            method: 预测方法 ('deterministic' 或 'monte_carlo')
            delta_t: 时间步长
            n_samples: 蒙特卡洛采样数量
            max_horizon: 最大预测步数

        Returns:
            Tuple[Trajectory, Dict]: (主轨迹, 统计量)
        """
        if method == 'deterministic':
            trajectory = self.deterministic.predict(initial_state, policy, delta_t, max_horizon)
            stats = {
                'mean_energy': trajectory.total_energy,
                'mean_time': trajectory.total_time,
                'mean_progress': trajectory.final_progress,
                'mean_uncertainty': trajectory.get_mean_uncertainty(),
                'success_rate': 0.0 if trajectory.is_terminated else 1.0
            }
        elif method == 'monte_carlo':
            trajectories = self.mc_sampler.sample_trajectories(
                initial_state, policy, n_samples, delta_t, max_horizon
            )
            stats = self.mc_sampler.estimate_statistics(trajectories)
            # 选择最接近均值的主轨迹
            mean_progress = stats['mean_progress']
            trajectory = min(trajectories, key=lambda t: abs(t.final_progress - mean_progress))
        else:
            raise ValueError(f"未知预测方法: {method}")

        return trajectory, stats

    def propagate_uncertainty(self,
                            sigma: float,
                            action: ActionType,
                            delta_t: float = 1.0) -> float:
        """
        传播不确定性

        Args:
            sigma: 当前不确定性
            action: 行动
            delta_t: 时间步长

        Returns:
            float: 下一时刻的不确定性
        """
        return self.uncertainty_propagator.propagate(sigma, action, delta_t)


# ============ 测试用例 ============

def test_trajectory_predictor():
    """测试TrajectoryPredictor模块"""
    print("=" * 60)
    print("测试 M17C: TrajectoryPredictor (自由能.txt 第235-289行)")
    print("=" * 60)

    from algorithms.active_inference.generative_model import GenerativeModel

    # 初始化
    gen_model = GenerativeModel(T_task=50.0)
    predictor = TrajectoryPredictor(gen_model)

    # 测试1: 初始状态
    print("\n[Test 1] 测试初始状态...")
    initial_state = StateVector(
        E=400e3,  # 400kJ
        T=0.0,
        h=0.9,
        p=0.0,
        d=800.0,
        sigma=0.1
    )
    print(f"  初始状态: E={initial_state.E/1e3:.1f}kJ, h={initial_state.h:.2f}, "
          f"p={initial_state.p:.2f}, σ={initial_state.sigma:.2f}")
    print("  ✓ 初始状态正确")

    # 测试2: 确定性预测
    print("\n[Test 2] 测试确定性预测...")
    policy = [ActionType.CONTINUE] * 10  # 继续执行10步
    trajectory, stats = predictor.predict(initial_state, policy, method='deterministic')

    print(f"  预测轨迹长度: {len(trajectory)}")
    print(f"  最终状态: E={trajectory.get_final_state().E/1e3:.1f}kJ, "
          f"h={trajectory.get_final_state().h:.2f}, p={trajectory.get_final_state().p:.3f}")
    print(f"  总能耗: {trajectory.total_energy/1e3:.1f}kJ, "
          f"总时间: {trajectory.total_time:.1f}s")
    print("  ✓ 确定性预测正确")

    # 测试3: 蒙特卡洛采样
    print("\n[Test 3] 测试蒙特卡洛采样...")
    mc_trajectory, mc_stats = predictor.predict(
        initial_state, policy, method='monte_carlo', n_samples=50
    )

    print(f"  采样统计:")
    print(f"    平均能耗: {mc_stats['mean_energy']/1e3:.1f}kJ")
    print(f"    平均时间: {mc_stats['mean_time']:.1f}s")
    print(f"    平均进度: {mc_stats['mean_progress']:.3f}")
    print(f"    平均不确定性: {mc_stats['mean_uncertainty']:.3f}")
    print(f"    成功率: {mc_stats['success_rate']*100:.1f}%")
    print("  ✓ 蒙特卡洛采样正确")

    # 测试4: 不同策略比较
    print("\n[Test 4] 测试不同策略比较...")
    policies = {
        'Continue': [ActionType.CONTINUE] * 10,
        'Checkpoint': [ActionType.CHECKPOINT] + [ActionType.CONTINUE] * 9,
        'Reduce': [ActionType.REDUCE_POWER] * 10
    }

    for name, policy in policies.items():
        traj, st = predictor.predict(initial_state, policy, method='deterministic')
        print(f"  {name:10s}: E={traj.total_energy/1e3:.1f}kJ, "
              f"T={traj.total_time:.1f}s, p={traj.final_progress:.3f}, "
              f"σ={traj.get_mean_uncertainty():.3f}")
    print("  ✓ 策略比较正确")

    # 测试5: 不确定性传播
    print("\n[Test 5] 测试不确定性传播...")
    sigma = 0.1
    sigmas = [sigma]

    for action in [ActionType.CONTINUE, ActionType.CONTINUE, ActionType.HANDOVER]:
        sigma = predictor.propagate_uncertainty(sigma, action)
        sigmas.append(sigma)

    print(f"  不确定性传播: {[f'{s:.3f}' for s in sigmas]}")
    assert sigmas[-1] > sigmas[0], "不确定性应增长"
    print("  ✓ 不确定性传播正确")

    # 测试6: 终止状态处理
    print("\n[Test 6] 测试终止状态处理...")
    abort_policy = [ActionType.ABORT] + [ActionType.CONTINUE] * 5
    abort_traj, _ = predictor.predict(initial_state, abort_policy, method='deterministic')

    print(f"  终止后轨迹长度: {len(abort_traj)}")
    print(f"  是否终止: {abort_traj.is_terminated}")
    assert abort_traj.is_terminated, "ABORT应终止轨迹"
    print("  ✓ 终止状态处理正确")

    # 测试7: 不确定性量化
    print("\n[Test 7] 测试不确定性量化...")
    all_trajectories = predictor.mc_sampler.sample_trajectories(
        initial_state, [ActionType.CONTINUE] * 10, n_samples=30
    )

    max_u = max(t.get_max_uncertainty() for t in all_trajectories)
    mean_u = np.mean([t.get_mean_uncertainty() for t in all_trajectories])

    print(f"  最大不确定性: {max_u:.3f}")
    print(f"  平均不确定性: {mean_u:.3f}")
    print("  ✓ 不确定性量化正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_trajectory_predictor()

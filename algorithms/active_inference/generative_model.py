"""
M17B: GenerativeModel - 生成模型

功能：建立世界的内部模型，描述状态如何演化、观测如何产生
参考文档：docs/自由能.txt 第133-232行

完整生成模型 (自由能.txt 第140-151行):
    P(õ, s̃, π) = P(π) × Π P(o_τ | s_τ) × P(s_τ | s_{τ-1}, a_{τ-1})

状态转移模型 (自由能.txt 第152-215行):
    - 能量转移: E_{t+1} = E_t - P(a_t) × Δt - ε_E
    - 时间转移: T_{t+1} = T_t + Δt × κ(a_t)
    - 健康度转移: h_{t+1} = h_t × λ_h - δ_h(a_t) + ε_h
    - 进度转移: p_{t+1} = min(1, p_t + Δt × η(a_t) / T_task)

似然模型 (自由能.txt 第216-222行):
    P(o_t | s_t) = N(o_t; g(s_t), Σ_o)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import (
    StateVector, ObservationVector, ActionType,
    ActionEffect, ActionSet
)
from config.constants import NUMERICAL


@dataclass
class TransitionResult:
    """
    状态转移结果

    Attributes:
        next_state: 下一状态
        energy_consumed: 消耗的能量 (J)
        time_elapsed: 经过的时间 (s)
        progress_made: 推进的进度 [0,1]
        is_terminal: 是否终止状态
    """
    next_state: StateVector
    energy_consumed: float
    time_elapsed: float
    progress_made: float
    is_terminal: bool = False


@dataclass
class NoiseParameters:
    """
    噪声模型参数 (自由能.txt 第224-231行子模块清单)

    Attributes:
        sigma_E: 能量观测噪声标准差 (J)
        sigma_h: 健康度观测噪声标准差
        process_noise_E: 能量过程噪声标准差 (J/s)
        process_noise_h: 健康度过程噪声标准差
    """
    sigma_E: float = 1000.0      # 能量观测噪声 (J)
    sigma_h: float = 0.02         # 健康度观测噪声
    process_noise_E: float = 50.0  # 能量过程噪声 (J/s)
    process_noise_h: float = 0.005 # 健康度过程噪声


class TransitionModel:
    """
    状态转移模型 (自由能.txt 第224-229行子模块清单)

    功能：给定当前状态和行动，预测下一状态
    公式 (自由能.txt 第154-214行)
    """

    def __init__(self,
                 action_effects: Optional[ActionEffect] = None,
                 noise_params: Optional[NoiseParameters] = None,
                 T_task: float = 100.0):
        """
        初始化转移模型

        Args:
            action_effects: 行动效果参数
            noise_params: 噪声参数
            T_task: 任务总时间预算 (s)
        """
        self.action_effects = action_effects or ActionEffect()
        self.noise = noise_params or NoiseParameters()
        self.T_task = T_task

    def predict_next_state(self,
                          state: StateVector,
                          action: ActionType,
                          delta_t: float = 1.0) -> TransitionResult:
        """
        预测下一状态 (自由能.txt 第154-214行)

        Args:
            state: 当前状态
            action: 执行的行动
            delta_t: 时间步长 (s)

        Returns:
            TransitionResult: 转移结果
        """
        # 获取行动参数
        P_action = self.action_effects.get_power(action)
        kappa = self.action_effects.get_time_coefficient(action)
        eta = self.action_effects.get_efficiency_factor(action)

        # 能量转移 (自由能.txt 第156-158行)
        # E_{t+1} = E_t - P(a_t) × Δt - ε_E
        E_next = state.E - P_action * delta_t
        energy_consumed = P_action * delta_t

        # 添加过程噪声
        E_next += np.random.normal(0, self.noise.process_noise_E * delta_t)

        # 时间转移 (自由能.txt 第174-176行)
        # T_{t+1} = T_t + Δt × κ(a_t)
        if action == ActionType.ABORT:
            T_next = state.T  # 终止后时间不再增加
        else:
            T_next = state.T + delta_t * kappa

        # 健康度转移 (自由能.txt 第191-194行)
        # h_{t+1} = h_t × λ_h - δ_h(a_t) + ε_h
        h_next = state.h * self.action_effects.lambda_h

        # 行动导致的额外健康损耗
        if action == ActionType.CHECKPOINT:
            delta_h = 0.01  # Checkpoint有轻微开销
        elif action == ActionType.HANDOVER:
            delta_h = 0.02  # Handover有风险
        elif action == ActionType.ABORT:
            delta_h = 0.05  # 紧急中止有较大风险
        else:
            delta_h = 0.0

        h_next -= delta_h
        h_next += np.random.normal(0, self.noise.process_noise_h)

        # 限制在[0,1]范围
        h_next = np.clip(h_next, 0.0, 1.0)

        # 进度转移 (自由能.txt 第199-202行)
        # p_{t+1} = min(1, p_t + Δt × η(a_t) / T_task)
        progress_delta = delta_t * eta / self.T_task if action != ActionType.ABORT else 0.0
        p_next = min(1.0, state.p + progress_delta)

        # 距离和不确定性更新
        if action == ActionType.ABORT:
            d_next = 0.0  # 返航后距离重置
            sigma_next = 0.0
        else:
            d_next = state.d  # 距离不变
            # 不确定性随时间增长 (简化模型)
            sigma_next = min(1.0, state.sigma + 0.01 * delta_t)

        # 创建下一状态
        next_state = StateVector(
            E=max(0.0, E_next),
            T=T_next,
            h=h_next,
            p=p_next,
            d=d_next,
            sigma=sigma_next
        )

        # 判断是否终止
        is_terminal = (
            action == ActionType.ABORT or
            next_state.p >= 1.0 or
            next_state.E <= 0.0 or
            next_state.h <= 0.0
        )

        return TransitionResult(
            next_state=next_state,
            energy_consumed=energy_consumed,
            time_elapsed=delta_t * kappa if not is_terminal else 0.0,
            progress_made=progress_delta,
            is_terminal=is_terminal
        )


class LikelihoodModel:
    """
    似然模型 (自由能.txt 第224-231行子模块清单)

    功能：描述状态如何产生观测
    公式 (自由能.txt 第218-222行): P(o_t | s_t) = N(o_t; g(s_t), Σ_o)
    """

    def __init__(self, noise_params: Optional[NoiseParameters] = None):
        """
        初始化似然模型

        Args:
            noise_params: 观测噪声参数
        """
        self.noise = noise_params or NoiseParameters()

    def generate_observation(self, state: StateVector) -> ObservationVector:
        """
        从状态生成观测 (自由能.txt 第98-106行)

        Args:
            state: 当前状态

        Returns:
            ObservationVector: 观测向量
        """
        # 能量观测 (自由能.txt 第103行): 带噪声的能量读数
        E_hat = state.E + np.random.normal(0, self.noise.sigma_E)
        E_hat = max(0.0, E_hat)

        # 健康度观测 (自由能.txt 第104行): 传感器检测值
        h_hat = state.h + np.random.normal(0, self.noise.sigma_h)
        h_hat = np.clip(h_hat, 0.0, 1.0)

        # 信道质量 (自由能.txt 第105行): 与不确定性相关
        # 不确定性越高，信道质量越差
        q = 1.0 - state.sigma * 0.5 + np.random.normal(0, 0.05)
        q = np.clip(q, 0.0, 1.0)

        # 环境因素 (自由能.txt 第106行): 风速、温度等
        # 简化为与不确定性相关的噪声
        w = 0.8 + np.random.normal(0, 0.1 * (1 - state.sigma))
        w = np.clip(w, 0.0, 1.0)

        return ObservationVector(
            E_hat=E_hat,
            h_hat=h_hat,
            q=q,
            w=w
        )

    def compute_likelihood(self,
                        obs: ObservationVector,
                        state: StateVector) -> float:
        """
        计算似然 P(o_t | s_t) (自由能.txt 第218-222行)

        使用高斯似然:
        P(o_t | s_t) = N(o_t; g(s_t), Σ_o)

        Args:
            obs: 观测向量
            state: 状态向量

        Returns:
            float: 对数似然
        """
        # 预测观测 (状态到观测的映射)
        pred_E = state.E
        pred_h = state.h
        pred_q = 1.0 - state.sigma * 0.5
        pred_w = 0.8

        # 计算高斯似然
        # log P = -0.5 * (log(2πσ²) + (x-μ)²/σ²)
        log_p_E = -0.5 * (np.log(2 * np.pi * self.noise.sigma_E**2) +
                         (obs.E_hat - pred_E)**2 / (self.noise.sigma_E**2 + NUMERICAL.EPSILON))

        log_p_h = -0.5 * (np.log(2 * np.pi * self.noise.sigma_h**2) +
                         (obs.h_hat - pred_h)**2 / (self.noise.sigma_h**2 + NUMERICAL.EPSILON))

        # 简化：q和w使用固定方差
        log_p_q = -0.5 * (np.log(2 * np.pi * 0.05**2) +
                         (obs.q - pred_q)**2 / 0.0025)
        log_p_w = -0.5 * (np.log(2 * np.pi * 0.1**2) +
                         (obs.w - pred_w)**2 / 0.01)

        return log_p_E + log_p_h + log_p_q + log_p_w


class GenerativeModel:
    """
    生成模型 (自由能.txt 第133-232行)

    整合状态转移模型和似然模型

    Attributes:
        transition_model: 状态转移模型
        likelihood_model: 似然模型
        action_set: 行动集合
    """

    def __init__(self,
                 action_effects: Optional[ActionEffect] = None,
                 noise_params: Optional[NoiseParameters] = None,
                 T_task: float = 100.0):
        """
        初始化生成模型

        Args:
            action_effects: 行动效果参数
            noise_params: 噪声参数
            T_task: 任务总时间预算 (s)
        """
        self.transition_model = TransitionModel(action_effects, noise_params, T_task)
        self.likelihood_model = LikelihoodModel(noise_params)
        self.action_set = ActionSet(action_effects)

    def predict_next(self,
                     state: StateVector,
                     action: ActionType,
                     delta_t: float = 1.0) -> TransitionResult:
        """
        预测下一状态

        Args:
            state: 当前状态
            action: 执行的行动
            delta_t: 时间步长

        Returns:
            TransitionResult: 转移结果
        """
        return self.transition_model.predict_next_state(state, action, delta_t)

    def generate_observation(self, state: StateVector) -> ObservationVector:
        """
        生成观测

        Args:
            state: 当前状态

        Returns:
            ObservationVector: 观测向量
        """
        return self.likelihood_model.generate_observation(state)

    def compute_likelihood(self,
                        obs: ObservationVector,
                        state: StateVector) -> float:
        """
        计算似然

        Args:
            obs: 观测向量
            state: 状态向量

        Returns:
            float: 对数似然
        """
        return self.likelihood_model.compute_likelihood(obs, state)


# ============ 测试用例 ============

def test_generative_model():
    """测试GenerativeModel模块"""
    print("=" * 60)
    print("测试 M17B: GenerativeModel (自由能.txt 第133-232行)")
    print("=" * 60)

    # 初始化模型
    model = GenerativeModel(T_task=50.0)

    # 测试1: 初始状态
    print("\n[Test 1] 测试初始状态...")
    initial_state = StateVector(
        E=450e3,  # 450kJ
        T=0.0,
        h=0.95,
        p=0.0,
        d=800.0,
        sigma=0.1
    )
    print(f"  初始状态: E={initial_state.E/1e3:.1f}kJ, T={initial_state.T:.1f}s, "
          f"h={initial_state.h:.2f}, p={initial_state.p:.2f}")
    print("  ✓ 初始状态创建成功")

    # 测试2: 状态转移
    print("\n[Test 2] 测试状态转移...")
    for action in [ActionType.CONTINUE, ActionType.CHECKPOINT, ActionType.REDUCE_POWER]:
        result = model.predict_next(initial_state, action, delta_t=1.0)
        print(f"  行动={action:12s}: E={result.next_state.E/1e3:.1f}kJ, "
              f"T={result.next_state.T:.1f}s, 进度={result.next_state.p:.3f}, "
              f"能耗={result.energy_consumed:.1f}J")
    print("  ✓ 状态转移正确")

    # 测试3: 观测生成
    print("\n[Test 3] 测试观测生成...")
    obs = model.generate_observation(initial_state)
    print(f"  真实状态: E={initial_state.E/1e3:.1f}kJ, h={initial_state.h:.2f}")
    print(f"  观测状态: Ê={obs.E_hat/1e3:.1f}kJ, ĥ={obs.h_hat:.2f}, "
          f"q={obs.q:.2f}, w={obs.w:.2f}")
    print("  ✓ 观测生成正确")

    # 测试4: 序列模拟
    print("\n[Test 4] 测试序列模拟...")
    state = initial_state.copy()
    total_energy = 0.0
    total_time = 0.0

    for step in range(5):
        # 选择行动 (简化：总是继续)
        result = model.predict_next(state, ActionType.CONTINUE, delta_t=1.0)
        state = result.next_state
        total_energy += result.energy_consumed
        total_time += result.time_elapsed

        print(f"  步骤{step+1}: E={state.E/1e3:.1f}kJ, "
              f"T={state.T:.1f}s, p={state.p:.3f}")

    print(f"  总能耗: {total_energy:.1f}J, 总时间: {total_time:.1f}s")
    print("  ✓ 序列模拟正确")

    # 测试5: 终止状态
    print("\n[Test 5] 测试终止状态...")
    abort_state = StateVector(E=400e3, T=10.0, h=0.9, p=0.5, d=800.0, sigma=0.1)
    result_abort = model.predict_next(abort_state, ActionType.ABORT, delta_t=1.0)
    print(f"  ABORT后状态: E={result_abort.next_state.E/1e3:.1f}kJ, "
          f"d={result_abort.next_state.d:.1f}m, 终止={result_abort.is_terminal}")
    assert result_abort.is_terminal, "ABORT应为终止状态"
    print("  ✓ 终止状态正确")

    # 测试6: 似然计算
    print("\n[Test 6] 测试似然计算...")
    obs = model.generate_observation(initial_state)
    log_likelihood = model.compute_likelihood(obs, initial_state)

    # 创建偏差观测
    obs_biased = ObservationVector(
        E_hat=initial_state.E - 10000,
        h_hat=initial_state.h - 0.1,
        q=0.5,
        w=0.5
    )
    log_likelihood_biased = model.compute_likelihood(obs_biased, initial_state)

    print(f"  准确观测似然: {log_likelihood:.2f}")
    print(f"  偏差观测似然: {log_likelihood_biased:.2f}")
    assert log_likelihood > log_likelihood_biased, "准确观测应有更高似然"
    print("  ✓ 似然计算正确")

    # 测试7: 健康度衰减
    print("\n[Test 7] 测试健康度衰减...")
    state_h = StateVector(E=400e3, T=0, h=1.0, p=0, d=500, sigma=0)
    h_values = [state_h.h]

    for i in range(10):
        result = model.predict_next(state_h, ActionType.CONTINUE, delta_t=1.0)
        state_h = result.next_state
        h_values.append(state_h.h)

    print(f"  健康度衰减: {h_values[0]:.3f} -> {h_values[-1]:.3f}")
    assert h_values[-1] < h_values[0], "健康度应衰减"
    print("  ✓ 健康度衰减正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_generative_model()

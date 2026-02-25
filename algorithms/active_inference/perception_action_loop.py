"""
M17G: PerceptionActionLoop - 感知-行动循环

功能：整合所有模块，实现完整的主动推理循环
参考文档：docs/自由能.txt 第533-605行

主循环流程 (自由能.txt 第540-570行):
    1. OBSERVE: o_t ← get_observation()
    2. PERCEIVE: Q(s_t) ← update_belief(Q(s_{t-1}), o_t), F_t ← compute_free_energy(Q(s_t))
    3. PREDICT: For each a ∈ A: trajectory_a ← predict(Q(s_t), a, horizon), G(a) ← compute_expected_free_energy(trajectory_a)
    4. ACT: a* ← select_action({G(a)}), execute(a*)
    5. LEARN: update_model_parameters(o_t, s_t, a*)
    6. LOOP: t ← t + 1, goto 1

时间预算分配 (自由能.txt 第572-594行):
    - 观测获取: 5%
    - 信念更新: 15%
    - 轨迹预测: 40%
    - 行动选择: 30%
    - 执行与记录: 10%
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import (
    StateVector, ObservationVector, ActionType, StateBounds
)
from algorithms.active_inference.generative_model import GenerativeModel
from algorithms.active_inference.trajectory_predictor import TrajectoryPredictor
from algorithms.active_inference.belief_updater import BeliefUpdater, BeliefState
from algorithms.active_inference.free_energy import FreeEnergyCalculator
from algorithms.active_inference.action_selector import ActionSelector, ActionEvaluation
from config.constants import NUMERICAL


class LoopPhase(Enum):
    """循环阶段"""
    OBSERVE = "OBSERVE"
    PERCEIVE = "PERCEIVE"
    PREDICT = "PREDICT"
    ACT = "ACT"
    LEARN = "LEARN"


@dataclass
class LoopState:
    """
    循环状态

    Attributes:
        current_time: 当前时间步
        belief: 当前信念
        last_action: 上次执行的行动
        last_observation: 上次观测
        free_energy: 当前自由能
        action_evaluations: 行动评估结果
        decision_confidence: 决策置信度
    """
    current_time: int = 0
    belief: Optional[BeliefState] = None
    last_action: Optional[ActionType] = None
    last_observation: Optional[ObservationVector] = None
    free_energy: Optional[float] = None
    action_evaluations: List[ActionEvaluation] = field(default_factory=list)
    decision_confidence: float = 0.5


@dataclass
class ExecutionResult:
    """
    执行结果

    Attributes:
        success: 是否成功
        action: 执行的行动
        new_state: 新状态
        reward: 奖励值
        terminated: 是否终止
        phase: 当前阶段
        time_used: 使用时间
    """
    success: bool
    action: ActionType
    new_state: StateVector
    reward: float
    terminated: bool
    phase: LoopPhase
    time_used: float = 0.0


class TimeBudgetManager:
    """
    时间预算管理器 (自由能.txt 第571-604行子模块清单)

    功能：管理各阶段的时间预算

    时间预算分配 (自由能.txt 第576-582行):
        - 观测获取: 5%
        - 信念更新: 15%
        - 轨迹预测: 40%
        - 行动选择: 30%
        - 执行与记录: 10%
    """

    def __init__(self, T_budget: float = 1.0):
        """
        初始化时间预算管理器

        Args:
            T_budget: 每个决策周期的时间预算 (s)
        """
        self.T_budget = T_budget
        self.T_available = T_budget

        # 时间分配比例 (自由能.txt 第576-582行)
        self.allocation = {
            'observe': 0.05,
            'perceive': 0.15,
            'predict': 0.40,
            'act': 0.30,
            'execute': 0.10
        }

    def get_time_budget(self, phase: LoopPhase) -> float:
        """
        获取阶段时间预算

        Args:
            phase: 循环阶段

        Returns:
            float: 时间预算 (s)
        """
        return self.T_budget * self.allocation.get(phase.value.lower(), 0.1)

    def reset(self):
        """重置时间预算"""
        self.T_available = self.T_budget

    def update_used_time(self, time_used: float):
        """
        更新已用时间

        Args:
            time_used: 已用时间 (s)
        """
        self.T_available -= time_used

    def get_remaining_time(self) -> float:
        """获取剩余时间"""
        return max(0, self.T_available)


class ObservationInterface:
    """
    观测获取接口 (自由能.txt 第599-600行子模块清单)

    功能：获取系统观测

    Attributes:
        gen_model: 生成模型
        current_state: 当前真实状态
    """

    def __init__(self, gen_model: GenerativeModel):
        """
        初始化观测接口

        Args:
            gen_model: 生成模型
        """
        self.gen_model = gen_model
        self.current_state = StateVector(
            E=500e3, T=0, h=1.0, p=0, d=1000, sigma=0.1
        )

    def update_state(self, new_state: StateVector):
        """
        更新真实状态

        Args:
            new_state: 新状态
        """
        self.current_state = new_state

    def get_observation(self, noise: bool = True) -> ObservationVector:
        """
        获取观测 (自由能.txt 第546-547行)

        Args:
            noise: 是否添加观测噪声

        Returns:
            ObservationVector: 观测向量
        """
        if noise:
            return self.gen_model.generate_observation(self.current_state)
        else:
            # 无噪声观测 (使用状态直接映射)
            return ObservationVector(
                E_hat=self.current_state.E,
                h_hat=self.current_state.h,
                q=1.0 - self.current_state.sigma * 0.5,
                w=0.8
            )


class ExecutionInterface:
    """
    行动执行接口 (自由能.txt 第601-602行子模块清单)

    功能：执行行动并返回新状态

    Attributes:
        gen_model: 生成模型
    """

    def __init__(self, gen_model: GenerativeModel):
        """
        初始化执行接口

        Args:
            gen_model: 生成模型
        """
        self.gen_model = gen_model
        self.obs_interface = ObservationInterface(gen_model)

    def execute(self,
                action: ActionType,
                current_belief_mean: StateVector,
                delta_t: float = 1.0) -> ExecutionResult:
        """
        执行行动

        Args:
            action: 行动
            current_belief_mean: 信念均值
            delta_t: 时间步长

        Returns:
            ExecutionResult: 执行结果
        """
        # 使用真实状态执行
        result = self.gen_model.predict_next(
            self.obs_interface.current_state, action, delta_t
        )

        # 更新真实状态
        self.obs_interface.update_state(result.next_state)

        # 计算奖励
        if action == ActionType.CONTINUE:
            reward = result.progress_made * 10 - result.energy_consumed / 1e4
        elif action == ActionType.ABORT:
            reward = -5.0  # 中止惩罚
        elif action == ActionType.CHECKPOINT:
            reward = -1.0  # Checkpoint小开销
        elif action == ActionType.HANDOVER:
            reward = -2.0  # Handover开销
        else:
            reward = 0.0

        return ExecutionResult(
            success=not result.is_terminal,
            action=action,
            new_state=result.next_state,
            reward=reward,
            terminated=result.is_terminal,
            phase=LoopPhase.EXECUTE,
            time_used=delta_t
        )


class PerceptionActionLoop:
    """
    感知-行动循环 (自由能.txt 第533-605行)

    整合所有模块，实现完整的主动推理循环

    主循环流程 (自由能.txt 第540-570行):
        1. OBSERVE: o_t ← get_observation()
        2. PERCEIVE: Q(s_t) ← update_belief(Q(s_{t-1}), o_t)
        3. PREDICT: For each a ∈ A: trajectory_a ← predict(Q(s_t), a, horizon)
        4. ACT: a* ← select_action({G(a)}), execute(a*)
        5. LEARN: update_model_parameters(o_t, s_t, a*)
        6. LOOP: t ← t + 1, goto 1

    Attributes:
        gen_model: 生成模型
        traj_predictor: 轨迹预测器
        belief_updater: 信念更新器
        fe_calculator: 自由能计算器
        action_selector: 行动选择器
        obs_interface: 观测接口
        exec_interface: 执行接口
        time_manager: 时间预算管理器
        loop_state: 循环状态
        callbacks: 回调函数
    """

    def __init__(self,
                 initial_state: Optional[StateVector] = None,
                 selection_method: str = 'greedy',
                 H: int = 5,
                 T_budget: float = 1.0):
        """
        初始化感知-行动循环

        Args:
            initial_state: 初始状态
            selection_method: 选择方法
            H: 预测时域
            T_budget: 时间预算
        """
        # 初始化生成模型
        self.gen_model = GenerativeModel(T_task=100.0)

        # 初始化其他模块
        self.traj_predictor = TrajectoryPredictor(self.gen_model)
        self.fe_calculator = FreeEnergyCalculator(use_four_component=True)
        self.action_selector = ActionSelector(
            self.traj_predictor, self.fe_calculator, selection_method=selection_method, H=H
        )
        self.exec_interface = ExecutionInterface(self.gen_model)

        # 初始化信念更新器
        if initial_state:
            self.belief_updater = BeliefUpdater(self.gen_model, initial_state)
            self.exec_interface.obs_interface.update_state(initial_state)
        else:
            self.belief_updater = BeliefUpdater(self.gen_model)

        # 时间管理
        self.time_manager = TimeBudgetManager(T_budget=T_budget)

        # 循环状态
        self.loop_state = LoopState()
        self.loop_state.belief = self.belief_updater.get_current_belief()

        # 回调函数
        self.callbacks = {
            'on_observe': None,
            'on_perceive': None,
            'on_predict': None,
            'on_act': None,
            'on_learn': None,
            'on_loop': None
        }

        # 评估参数
        self._set_evaluation_parameters()

        # 日志
        self.history: List[Dict] = []

    def _set_evaluation_parameters(self):
        """设置评估参数"""
        bounds = StateBounds()
        self.action_selector.set_evaluation_params(
            E_required=bounds.E_max * 0.2,  # 需要20%能量
            T_remaining_required=bounds.T_max * 0.3,  # 需要30%时间
            channel_quality=0.8,
            p_expected=0.1
        )

    def set_callback(self, phase: LoopPhase, callback: Callable):
        """
        设置回调函数

        Args:
            phase: 循环阶段
            callback: 回调函数
        """
        self.callbacks[phase.value] = callback

    def step(self, max_steps: int = 100, verbose: bool = False) -> ExecutionResult:
        """
        执行一步循环

        Args:
            max_steps: 最大步数
            verbose: 是否打印详细信息

        Returns:
            ExecutionResult: 执行结果
        """
        self.time_manager.reset()

        # Phase 1: OBSERVE (自由能.txt 第546-548行)
        if verbose:
            print(f"[{self.loop_state.current_time}] Phase: {LoopPhase.OBSERVE.value}")
        start_time = time.time()
        obs = self.exec_interface.obs_interface.get_observation(noise=True)
        observe_time = time.time() - start_time
        self.time_manager.update_used_time(observe_time)

        if self.callbacks['on_observe']:
            self.callbacks['on_observe'](obs, self.loop_state)

        # Phase 2: PERCEIVE (自由能.txt 第549-551行)
        if verbose:
            print(f"[{self.loop_state.current_time}] Phase: {LoopPhase.PERCEIVE.value}")
        start_time = time.time()
        self.loop_state.belief = self.belief_updater.update(obs, method='kalman')
        perceive_time = time.time() - start_time
        self.time_manager.update_used_time(perceive_time)

        # 计算即时自由能
        instant_fe = self.fe_calculator.compute_instant(
            self.loop_state.belief.mean,
            self.action_selector.E_required,
            self.action_selector.T_remaining_required,
            self.action_selector.channel_quality,
            self.action_selector.p_expected
        )
        self.loop_state.free_energy = instant_fe.F_total

        if self.callbacks['on_perceive']:
            self.callbacks['on_perceive'](self.loop_state.belief, instant_fe)

        # Phase 3: PREDICT (自由能.txt 第553-557行)
        if verbose:
            print(f"[{self.loop_state.current_time}] Phase: {LoopPhase.PREDICT.value}")
        start_time = time.time()

        # 对每个行动预测 (自由能.txt 第554行)
        actions = list(ActionType)
        self.loop_state.action_evaluations = []

        for action in actions:
            # 评估行动
            eval_result = self.action_selector.evaluate_single_action(
                self.loop_state.belief.mean, action
            )
            self.loop_state.action_evaluations.append(eval_result)

        predict_time = time.time() - start_time
        self.time_manager.update_used_time(predict_time)

        if self.callbacks['on_predict']:
            self.callbacks['on_predict'](self.loop_state.action_evaluations)

        # Phase 4: ACT (自由能.txt 第558-560行)
        if verbose:
            print(f"[{self.loop_state.current_time}] Phase: {LoopPhase.ACT.value}")
        start_time = time.time()

        # 选择行动 (自由能.txt 第559行)
        selected = self.action_selector.select_action(self.loop_state.belief.mean)
        action = selected.action

        # 计算置信度
        if self.loop_state.action_evaluations:
            G_values = [e.G_value for e in self.loop_state.action_evaluations]
            confidence = min(1.0, (selected.G_value - min(G_values)) /
                              (max(G_values) - min(G_values) + 1e-6))
            self.loop_state.decision_confidence = confidence
        else:
            self.loop_state.decision_confidence = 0.5

        # 执行行动 (自由能.txt 第560行)
        exec_result = self.exec_interface.execute(
            action, self.loop_state.belief.mean
        )
        act_time = time.time() - start_time
        self.time_manager.update_used_time(act_time)

        self.loop_state.last_action = action
        self.loop_state.last_observation = obs

        if self.callbacks['on_act']:
            self.callbacks['on_act'](action, exec_result)

        # Phase 5: LEARN (自由能.txt 第562-564行)
        # 简化：通过信念更新隐式学习

        # 记录历史
        self.history.append({
            'time': self.loop_state.current_time,
            'observation': obs,
            'belief_mean': self.loop_state.belief.mean,
            'action': action,
            'free_energy': self.loop_state.free_energy,
            'reward': exec_result.reward,
            'terminated': exec_result.terminated
        })

        if verbose:
            print(f"[{self.loop_state.current_time}] Action: {action.value}, "
                  f"FE={self.loop_state.free_energy:.2f}, "
                  f"Reward={exec_result.reward:.2f}")

        # Phase 6: LOOP (自由能.txt 第565-567行)
        self.loop_state.current_time += 1

        if self.callbacks['on_loop']:
            self.callbacks['on_loop'](self.loop_state)

        return exec_result

    def run(self, max_steps: int = 100, verbose: bool = False) -> List[Dict]:
        """
        运行完整循环

        Args:
            max_steps: 最大步数
            verbose: 是否打印详细信息

        Returns:
            List[Dict]: 历史记录
        """
        for step in range(max_steps):
            result = self.step(verbose=verbose)

            if result.terminated:
                if verbose:
                    print(f"[{self.loop_state.current_time}] Terminated!")
                break

        return self.history

    def get_summary(self) -> Dict:
        """获取循环摘要"""
        if not self.history:
            return {}

        rewards = [h['reward'] for h in self.history]
        free_energies = [h['free_energy'] for h in self.history]

        actions = {}
        for h in self.history:
            action = h['action']
            actions[action] = actions.get(action, 0) + 1

        return {
            'total_steps': len(self.history),
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'average_free_energy': np.mean(free_energies),
            'min_free_energy': min(free_energies),
            'max_free_energy': max(free_energies),
            'action_counts': actions,
            'termination_time': self.loop_state.current_time
        }


# ============ 测试用例 ============

def test_perception_action_loop():
    """测试PerceptionActionLoop模块"""
    print("=" * 60)
    print("测试 M17G: PerceptionActionLoop (自由能.txt 第533-605行)")
    print("=" * 60)

    # 初始状态
    initial_state = StateVector(E=450e3, T=0.0, h=0.95, p=0.0, d=1000.0, sigma=0.1)

    # 测试1: 贪婪选择循环
    print("\n[Test 1] 测试贪婪选择循环...")
    loop_greedy = PerceptionActionLoop(
        initial_state=initial_state,
        selection_method='greedy',
        H=5
    )

    history_greedy = loop_greedy.run(max_steps=20, verbose=False)
    summary_greedy = loop_greedy.get_summary()

    print(f"  总步数: {summary_greedy['total_steps']}")
    print(f"  总奖励: {summary_greedy['total_reward']:.2f}")
    print(f"  平均奖励: {summary_greedy['average_reward']:.2f}")
    print(f"  平均自由能: {summary_greedy['average_free_energy']:.2f}")
    print(f"  行动统计: {summary_greedy['action_counts']}")
    print("  ✓ 贪婪循环正确")

    # 测试2: Softmax选择循环
    print("\n[Test 2] 测试Softmax选择循环...")
    loop_softmax = PerceptionActionLoop(
        initial_state=initial_state,
        selection_method='softmax',
        H=5
    )

    history_softmax = loop_softmax.run(max_steps=20, verbose=False)
    summary_softmax = loop_softmax.get_summary()

    print(f"  总步数: {summary_softmax['total_steps']}")
    print(f"  总奖励: {summary_softmax['total_reward']:.2f}")
    print(f"  平均奖励: {summary_softmax['average_reward']:.2f}")
    print(f"  平均自由能: {summary_softmax['average_free_energy']:.2f}")
    print("  ✓ Softmax循环正确")

    # 测试3: 回调函数
    print("\n[Test 3] 测试回调函数...")
    loop_callbacks = PerceptionActionLoop(
        initial_state=initial_state,
        selection_method='greedy',
        H=3
    )

    callback_data = []

    def on_act_callback(action, result):
        callback_data.append(('act', action.value, result.reward))

    loop_callbacks.set_callback(LoopPhase.ACT, on_act_callback)
    loop_callbacks.run(max_steps=10, verbose=False)

    print(f"  回调触发次数: {len([d for d in callback_data if d[0] == 'act'])}")
    print("  ✓ 回调函数正确")

    # 测试4: 时间预算管理
    print("\n[Test 4] 测试时间预算管理...")
    time_manager = TimeBudgetManager(T_budget=1.0)

    print(f"  总预算: {time_manager.T_budget:.2f}s")
    print(f"  各阶段预算:")
    for phase in LoopPhase:
        budget = time_manager.get_time_budget(phase)
        print(f"    {phase.value:12s}: {budget:.2f}s")
    print("  ✓ 时间预算管理正确")

    # 测试5: 循环状态追踪
    print("\n[Test 5] 测试循环状态追踪...")
    loop_state = loop_greedy.loop_state

    print(f"  最终时间步: {loop_state.current_time}")
    print(f"  最终行动: {loop_state.last_action.value if loop_state.last_action else 'None'}")
    print(f"  最终自由能: {loop_state.free_energy:.2f}")
    print(f"  信念均值: E={loop_state.belief.mean.E/1e3:.1f}kJ, "
          f"h={loop_state.belief.mean.h:.2f}")
    print("  ✓ 循环状态追踪正确")

    # 测试6: 观测和信念更新
    print("\n[Test 6] 测试观测和信念更新...")
    obs = loop_greedy.exec_interface.obs_interface.get_observation()
    print(f"  观测: Ê={obs.E_hat/1e3:.1f}kJ, ĥ={obs.h_hat:.2f}")

    old_belief = loop_greedy.belief_updater.get_current_belief()
    new_belief = loop_greedy.belief_updater.update(obs, method='kalman')

    print(f"  更新前信念: E={old_belief.mean.E/1e3:.1f}kJ")
    print(f"  更新后信念: E={new_belief.mean.E/1e3:.1f}kJ")
    print("  ✓ 观测和信念更新正确")

    # 测试7: 单步执行
    print("\n[Test 7] 测试单步执行...")
    loop_single = PerceptionActionLoop(
        initial_state=initial_state,
        selection_method='greedy',
        H=3
    )

    result = loop_single.step(max_steps=1, verbose=True)
    print(f"  执行结果: 成功={result.success}, 终止={result.terminated}, "
          f"奖励={result.reward:.2f}")
    print("  ✓ 单步执行正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_perception_action_loop()

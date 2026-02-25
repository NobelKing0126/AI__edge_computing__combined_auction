"""
M17A: StateSpace - 状态空间定义

功能：定义系统的状态空间、观测空间和行动空间
参考文档：docs/自由能.txt 第74-131行

状态向量定义 (自由能.txt 第82-93行):
    s_t = [E_t, T_t, h_t, p_t, d_t, σ_t]
    - E_t: 剩余能量 (J)
    - T_t: 已用时间 (s)
    - h_t: 健康度 [0,1]
    - p_t: 任务进度 [0,1]
    - d_t: 距离目标/基站距离 (m)
    - σ_t: 环境不确定性 [0,1]

观测向量定义 (自由能.txt 第95-107行):
    o_t = [Ê_t, ĥ_t, q_t, w_t]
    - Ê_t: 能量观测 (带噪声)
    - ĥ_t: 健康度观测
    - q_t: 信道质量
    - w_t: 环境因素

行动空间定义 (自由能.txt 第108-121行):
    A = {a_1, a_2, a_3, a_4, a_5}
    - a_continue: 继续执行
    - a_checkpoint: 保存检查点
    - a_reduce: 降低功率
    - a_handover: 请求转移
    - a_abort: 中止返航
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import NUMERICAL


class ActionType(Enum):
    """行动类型 (自由能.txt 第110-121行)"""
    CONTINUE = "continue"       # 继续执行
    CHECKPOINT = "checkpoint"   # 保存检查点
    REDUCE_POWER = "reduce"     # 降低功率
    HANDOVER = "handover"       # 请求转移
    ABORT = "abort"            # 中止返航

    def __str__(self) -> str:
        return self.value


@dataclass
class StateVector:
    """
    状态向量 (自由能.txt 第82-93行)

    Attributes:
        E: 剩余能量 (J)
        T: 已用时间 (s)
        h: 健康度 [0,1]
        p: 任务进度 [0,1]
        d: 距离目标/基站距离 (m)
        sigma: 环境不确定性 [0,1]
    """
    E: float      # 剩余能量 (J)
    T: float      # 已用时间 (s)
    h: float      # 健康度 [0,1]
    p: float      # 任务进度 [0,1]
    d: float      # 距离目标 (m)
    sigma: float  # 不确定性 [0,1]

    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.E, self.T, self.h, self.p, self.d, self.sigma], dtype=float)

    def copy(self) -> 'StateVector':
        """创建副本"""
        return StateVector(
            E=self.E,
            T=self.T,
            h=self.h,
            p=self.p,
            d=self.d,
            sigma=self.sigma
        )


@dataclass
class ObservationVector:
    """
    观测向量 (自由能.txt 第95-107行)

    Attributes:
        E_hat: 能量观测 (带噪声) (J)
        h_hat: 健康度观测 [0,1]
        q: 信道质量 [0,1]
        w: 环境因素 [0,1]
    """
    E_hat: float  # 能量观测 (J)
    h_hat: float  # 健康度观测 [0,1]
    q: float      # 信道质量 [0,1]
    w: float      # 环境因素 [0,1]

    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.E_hat, self.h_hat, self.q, self.w], dtype=float)


@dataclass
class ActionEffect:
    """
    行动效果参数 (自由能.txt 第159-188行)

    Attributes:
        P_base: 基础功率 (W)
        P_compute: 计算功率 (W)
        P_checkpoint: Checkpoint额外功率 (W)
        P_comm: 通信功率 (W)
        P_flight: 返航飞行功率 (W)
        kappa_cp: Checkpoint时间系数
        eta_reduce: 功率降低效率比
        lambda_h: 健康度自然衰减率
    """
    P_base: float = 150.0       # 基础功率 (W)
    P_compute: float = 50.0      # 计算功率 (W)
    P_checkpoint: float = 30.0   # Checkpoint额外功率 (W)
    P_comm: float = 20.0        # 通信功率 (W)
    P_flight: float = 200.0      # 返航飞行功率 (W)
    kappa_cp: float = 0.1        # Checkpoint时间系数
    eta_reduce: float = 0.6      # 功率降低效率比
    lambda_h: float = 0.99       # 健康度自然衰减率

    def get_power(self, action: ActionType) -> float:
        """
        获取行动对应的功率 (自由能.txt 第163-169行)

        Args:
            action: 行动类型

        Returns:
            float: 功率 (W)
        """
        if action == ActionType.CONTINUE:
            return self.P_base + self.P_compute
        elif action == ActionType.CHECKPOINT:
            return self.P_base + self.P_checkpoint
        elif action == ActionType.REDUCE_POWER:
            return self.P_base * self.eta_reduce
        elif action == ActionType.HANDOVER:
            return self.P_base + self.P_comm
        elif action == ActionType.ABORT:
            return self.P_flight
        else:
            return self.P_base

    def get_time_coefficient(self, action: ActionType) -> float:
        """
        获取行动的时间系数 (自由能.txt 第181-187行)

        Args:
            action: 行动类型

        Returns:
            float: 时间系数
        """
        if action == ActionType.CONTINUE:
            return 1.0
        elif action == ActionType.CHECKPOINT:
            return 1.0 + self.kappa_cp
        elif action == ActionType.REDUCE_POWER:
            return 1.0 / self.eta_reduce
        elif action == ActionType.HANDOVER:
            return 1.0 + 0.2  # 通信延迟
        elif action == ActionType.ABORT:
            return float('inf')  # 任务终止
        else:
            return 1.0

    def get_efficiency_factor(self, action: ActionType) -> float:
        """
        获取行动的效率因子 (自由能.txt 第207-213行)

        Args:
            action: 行动类型

        Returns:
            float: 效率因子 [0,1]
        """
        if action == ActionType.CONTINUE:
            return 1.0
        elif action == ActionType.CHECKPOINT:
            return 0.0  # Checkpoint不推进进度
        elif action == ActionType.REDUCE_POWER:
            return self.eta_reduce
        elif action == ActionType.HANDOVER:
            return 0.0  # Handover不推进进度
        elif action == ActionType.ABORT:
            return 0.0
        else:
            return 1.0


@dataclass
class StateBounds:
    """
    状态边界 (自由能.txt 第86-93行)

    Attributes:
        E_max: 最大能量 (J)
        E_min: 最小能量 (J)
        T_max: 最大时间 (s)
        d_max: 最大距离 (m)
    """
    E_max: float = 500e3      # 最大能量 500kJ
    E_min: float = 0.0         # 最小能量
    T_max: float = 300.0       # 最大时间 5分钟
    d_max: float = 2000.0      # 最大距离 2km


class StateNormalizer:
    """
    状态归一化工具 (自由能.txt 第122-129行子模块清单)

    功能：将状态向量归一化到[0,1]范围，便于神经网络处理
    """

    def __init__(self, bounds: StateBounds):
        """
        初始化归一化器

        Args:
            bounds: 状态边界
        """
        self.bounds = bounds

    def normalize(self, state: StateVector) -> np.ndarray:
        """
        归一化状态向量

        Args:
            state: 原始状态

        Returns:
            np.ndarray: 归一化后的状态向量 [0,1]
        """
        return np.array([
            state.E / max(self.bounds.E_max, NUMERICAL.EPSILON),
            state.T / max(self.bounds.T_max, NUMERICAL.EPSILON),
            state.h,
            state.p,
            state.d / max(self.bounds.d_max, NUMERICAL.EPSILON),
            state.sigma
        ], dtype=float)

    def denormalize(self, normalized: np.ndarray) -> StateVector:
        """
        反归一化状态向量

        Args:
            normalized: 归一化后的状态向量

        Returns:
            StateVector: 原始状态
        """
        return StateVector(
            E=normalized[0] * self.bounds.E_max,
            T=normalized[1] * self.bounds.T_max,
            h=np.clip(normalized[2], 0.0, 1.0),
            p=np.clip(normalized[3], 0.0, 1.0),
            d=normalized[4] * self.bounds.d_max,
            sigma=np.clip(normalized[5], 0.0, 1.0)
        )


class ActionSet:
    """
    行动集合 (自由能.txt 第111-121行子模块清单)

    功能：管理可用的行动类型和行动参数
    """

    def __init__(self, effects: Optional[ActionEffect] = None):
        """
        初始化行动集合

        Args:
            effects: 行动效果参数
        """
        self.effects = effects or ActionEffect()
        self.actions = list(ActionType)

    def get_all_actions(self) -> List[ActionType]:
        """获取所有可用行动"""
        return self.actions

    def get_action(self, name: str) -> Optional[ActionType]:
        """根据名称获取行动"""
        for action in self.actions:
            if action.value == name or str(action) == name:
                return action
        return None

    def get_action_power(self, action: ActionType) -> float:
        """获取行动的功率消耗"""
        return self.effects.get_power(action)

    def get_action_time_coeff(self, action: ActionType) -> float:
        """获取行动的时间系数"""
        return self.effects.get_time_coefficient(action)

    def get_action_efficiency(self, action: ActionType) -> float:
        """获取行动的效率因子"""
        return self.effects.get_efficiency_factor(action)


# ============ 测试用例 ============

def test_state_space():
    """测试StateSpace模块"""
    print("=" * 60)
    print("测试 M17A: StateSpace (自由能.txt 第74-131行)")
    print("=" * 60)

    # 测试1: 状态向量
    print("\n[Test 1] 测试状态向量...")
    state = StateVector(
        E=400e3,  # 400kJ
        T=10.0,   # 10s
        h=0.9,     # 90%健康度
        p=0.5,     # 50%进度
        d=500.0,   # 500m
        sigma=0.1   # 10%不确定性
    )
    print(f"  状态向量: E={state.E/1e3:.1f}kJ, T={state.T:.1f}s, "
          f"h={state.h:.2f}, p={state.p:.2f}, d={state.d:.0f}m, σ={state.sigma:.2f}")
    print("  ✓ 状态向量创建成功")

    # 测试2: 观测向量
    print("\n[Test 2] 测试观测向量...")
    obs = ObservationVector(
        E_hat=395e3,  # 带噪声的能量观测
        h_hat=0.88,    # 带噪声的健康度
        q=0.9,         # 信道质量
        w=0.7          # 环境因素
    )
    print(f"  观测向量: Ê={obs.E_hat/1e3:.1f}kJ, ĥ={obs.h_hat:.2f}, "
          f"q={obs.q:.2f}, w={obs.w:.2f}")
    print("  ✓ 观测向量创建成功")

    # 测试3: 行动效果
    print("\n[Test 3] 测试行动效果...")
    effects = ActionEffect()

    for action in ActionType:
        power = effects.get_power(action)
        time_coeff = effects.get_time_coeff(action)
        efficiency = effects.get_efficiency_factor(action)
        print(f"  {action:12s}: 功率={power:6.1f}W, "
              f"时间系数={time_coeff:.2f}, 效率={efficiency:.2f}")
    print("  ✓ 行动效果计算正确")

    # 测试4: 状态归一化
    print("\n[Test 4] 测试状态归一化...")
    bounds = StateBounds(E_max=500e3, T_max=300.0, d_max=2000.0)
    normalizer = StateNormalizer(bounds)

    norm_state = normalizer.normalize(state)
    print(f"  原始状态: E={state.E/1e3:.1f}kJ, T={state.T:.1f}s")
    print(f"  归一化状态: E={norm_state[0]:.3f}, T={norm_state[1]:.3f}")

    denorm_state = normalizer.denormalize(norm_state)
    print(f"  反归一化: E={denorm_state.E/1e3:.1f}kJ, T={denorm_state.T:.1f}s")

    assert abs(denorm_state.E - state.E) < 1.0, "能量反归一化错误"
    assert abs(denorm_state.T - state.T) < 0.1, "时间反归一化错误"
    print("  ✓ 归一化计算正确")

    # 测试5: 行动集合
    print("\n[Test 5] 测试行动集合...")
    action_set = ActionSet(effects)

    all_actions = action_set.get_all_actions()
    print(f"  可用行动数量: {len(all_actions)}")
    for action in all_actions:
        print(f"    - {action}")

    continue_action = action_set.get_action("continue")
    assert continue_action == ActionType.CONTINUE, "行动查找错误"
    print("  ✓ 行动集合正确")

    # 测试6: 状态复制
    print("\n[Test 6] 测试状态复制...")
    state_copy = state.copy()
    state_copy.E = 100e3  # 修改副本

    assert state.E == 400e3, "原状态不应被修改"
    assert state_copy.E == 100e3, "副本状态应被修改"
    print("  ✓ 状态复制正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_state_space()

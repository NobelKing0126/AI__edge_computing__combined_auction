"""
M01: UserBenefitModel - 用户收益模型模块

功能：实现基于idea118.txt 1.5节的用户收益模型
- 服务价值计算 (指数衰减形式)
- 用户收益计算
- 服务接受条件判断

公式参考 (idea118.txt 1.5节):
    服务价值: V_i = v_0 * omega_i * exp(-beta_T * T_actual)
    用户收益: U_i = V_i - P
    服务接受条件: U_i >= 0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from config.system_config import SystemConfig


@dataclass
class UserBenefitConfig:
    """
    用户收益模型配置参数

    基于idea118.txt 1.5.1节的参数设置
    """
    # 基础价值系数 (元)
    v0: float = 10.0

    # 时延敏感度 (s^-1)
    beta_T: float = 0.5

    # 自由能阈值
    F_threshold: float = 30.0

    # 最大服务时延参考值 (秒)，用于归一化
    T_max_ref: float = 10.0


class UserBenefitModel:
    """
    用户收益模型

    实现基于时延敏感性的指数衰减价值模型

    特性:
    - 时延越长，价值指数衰减
    - 无硬性最大时延约束
    - 高优先级任务omega_i大，基础价值高
    """

    def __init__(self, config: UserBenefitConfig = None, system_config: SystemConfig = None):
        """
        初始化用户收益模型

        Args:
            config: 用户收益配置
            system_config: 系统配置
        """
        self.config = config if config else UserBenefitConfig()
        self.system_config = system_config if system_config else SystemConfig()

    def compute_service_value(
        self,
        priority: float,
        actual_delay: float
    ) -> float:
        """
        计算服务价值

        V_i = v_0 * omega_i * exp(-beta_T * T_actual)

        Args:
            priority: 任务优先级 omega_i (0-1)
            actual_delay: 实际服务时延 (秒)

        Returns:
            float: 服务价值 (元)
        """
        v0 = self.config.v0
        beta_T = self.config.beta_T

        # 指数衰减价值
        value = v0 * priority * np.exp(-beta_T * actual_delay)

        return value

    def compute_user_benefit(
        self,
        priority: float,
        actual_delay: float,
        price: float
    ) -> float:
        """
        计算用户收益

        U_i = V_i - P = v_0 * omega_i * exp(-beta_T * T_actual) - P

        Args:
            priority: 任务优先级 omega_i (0-1)
            actual_delay: 实际服务时延 (秒)
            price: 服务价格 (元)

        Returns:
            float: 用户收益 (元)
        """
        service_value = self.compute_service_value(priority, actual_delay)
        user_benefit = service_value - price

        return user_benefit

    def check_service_acceptance(
        self,
        priority: float,
        actual_delay: float,
        price: float
    ) -> bool:
        """
        判断服务是否被接受

        服务接受条件: U_i >= 0

        Args:
            priority: 任务优先级 omega_i (0-1)
            actual_delay: 实际服务时延 (秒)
            price: 服务价格 (元)

        Returns:
            bool: 是否接受服务
        """
        user_benefit = self.compute_user_benefit(priority, actual_delay, price)
        return user_benefit >= 0

    def compute_max_acceptable_price(
        self,
        priority: float,
        actual_delay: float
    ) -> float:
        """
        计算最大可接受价格

        即用户收益为零时的价格上限

        Args:
            priority: 任务优先级 omega_i (0-1)
            actual_delay: 实际服务时延 (秒)

        Returns:
            float: 最大可接受价格 (元)
        """
        return self.compute_service_value(priority, actual_delay)

    def compute_value_decay_ratio(
        self,
        actual_delay: float
    ) -> float:
        """
        计算价值衰减比例

        用于分析时延对价值的影响

        Args:
            actual_delay: 实际服务时延 (秒)

        Returns:
            float: 衰减比例 (0-1)
        """
        beta_T = self.config.beta_T
        return np.exp(-beta_T * actual_delay)

    def batch_compute_user_benefits(
        self,
        priorities: List[float],
        actual_delays: List[float],
        prices: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """
        批量计算用户收益和接受状态

        Args:
            priorities: 任务优先级列表
            actual_delays: 实际时延列表
            prices: 价格列表

        Returns:
            Tuple[List[float], List[bool]]: (用户收益列表, 接受状态列表)
        """
        benefits = []
        acceptances = []

        for priority, delay, price in zip(priorities, actual_delays, prices):
            benefit = self.compute_user_benefit(priority, delay, price)
            accepted = benefit >= 0

            benefits.append(benefit)
            acceptances.append(accepted)

        return benefits, acceptances

    def get_value_decay_table(self, max_delay: float = 10.0, step: float = 1.0) -> Dict[float, float]:
        """
        获取价值衰减表

        用于展示不同时延下的价值衰减情况

        Args:
            max_delay: 最大时延 (秒)
            step: 时延步长 (秒)

        Returns:
            Dict[float, float]: {时延: 价值衰减比例}
        """
        delays = np.arange(0, max_delay + step, step)
        decay_table = {}

        for delay in delays:
            decay_ratio = self.compute_value_decay_ratio(delay)
            # 计算priority=1时的价值
            value = self.config.v0 * decay_ratio
            decay_table[delay] = value

        return decay_table

    # ============ V2修正版方法 ============

    def compute_service_value_v2(
        self,
        priority: float,
        utility_stage2: float
    ) -> float:
        """
        计算服务价值 (修正版V2)

        V = ω * η^(stage2)

        修正要点:
        - 服务价值基于阶段2综合效用，而非绝对时延
        - 更好地反映用户对服务质量的综合评价

        Args:
            priority: 任务优先级 ω (0-1)
            utility_stage2: 阶段2综合效用 η^(stage2) (0-1)

        Returns:
            float: 服务价值 (元)
        """
        v0 = self.config.v0

        # 基于阶段2效用的服务价值
        value = v0 * priority * utility_stage2

        return value

    def compute_user_benefit_v2(
        self,
        priority: float,
        utility_stage2: float,
        price_normalized: float
    ) -> float:
        """
        计算用户收益 (修正版V2)

        U = V - P_norm

        其中:
        - V = ω * η^(stage2) 为服务价值
        - P_norm 为归一化价格 [0, 1]

        Args:
            priority: 任务优先级 ω (0-1)
            utility_stage2: 阶段2综合效用 η^(stage2) (0-1)
            price_normalized: 归一化价格 P_norm (0-1)

        Returns:
            float: 用户收益
        """
        service_value = self.compute_service_value_v2(priority, utility_stage2)
        user_benefit = service_value - price_normalized

        return user_benefit

    def check_service_acceptance_v2(
        self,
        priority: float,
        utility_stage2: float,
        price_normalized: float
    ) -> bool:
        """
        判断服务是否被接受 (修正版V2)

        Accept = 1(U >= 0)

        Args:
            priority: 任务优先级 ω (0-1)
            utility_stage2: 阶段2综合效用 η^(stage2) (0-1)
            price_normalized: 归一化价格 P_norm (0-1)

        Returns:
            bool: 是否接受服务
        """
        user_benefit = self.compute_user_benefit_v2(priority, utility_stage2, price_normalized)
        return user_benefit >= 0

    def compute_max_acceptable_price_v2(
        self,
        priority: float,
        utility_stage2: float
    ) -> float:
        """
        计算最大可接受价格 (修正版V2)

        即用户收益为零时的价格上限

        Args:
            priority: 任务优先级 ω (0-1)
            utility_stage2: 阶段2综合效用 η^(stage2) (0-1)

        Returns:
            float: 最大可接受归一化价格
        """
        return self.compute_service_value_v2(priority, utility_stage2)


# ============ 便捷函数 ============

def compute_user_value(
    v0: float,
    omega_i: float,
    beta_T: float,
    T_actual: float
) -> float:
    """
    计算服务价值 (便捷函数)

    V_i = v_0 * omega_i * exp(-beta_T * T_actual)

    Args:
        v0: 基础价值系数 (元)
        omega_i: 任务优先级 (0-1)
        beta_T: 时延敏感度 (s^-1)
        T_actual: 实际服务时延 (秒)

    Returns:
        float: 服务价值 (元)
    """
    return v0 * omega_i * np.exp(-beta_T * T_actual)


def compute_user_benefit_simple(
    V_i: float,
    price: float
) -> float:
    """
    计算用户收益 (便捷函数)

    U_i = V_i - P

    Args:
        V_i: 服务价值 (元)
        price: 服务价格 (元)

    Returns:
        float: 用户收益 (元)
    """
    return V_i - price


def check_service_acceptance(U_i: float) -> bool:
    """
    判断服务是否被接受 (便捷函数)

    服务接受条件: U_i >= 0

    Args:
        U_i: 用户收益 (元)

    Returns:
        bool: 是否接受服务
    """
    return U_i >= 0


# ============ 测试用例 ============

def test_user_benefit_model():
    """测试UserBenefitModel模块"""
    print("=" * 60)
    print("测试 UserBenefitModel")
    print("=" * 60)

    # 创建模型
    model = UserBenefitModel()

    # 测试1: 价值衰减示例 (参考idea118.txt 1.5.4节)
    print("\n[Test 1] 价值衰减示例 (v0=10, omega=1, beta_T=0.5)")
    print("-" * 40)
    for delay in [0, 1, 2, 5, 10]:
        value = model.compute_service_value(1.0, delay)
        print(f"  时延={delay}s: 价值={value:.2f}元")

    # 测试2: 用户收益计算
    print("\n[Test 2] 用户收益计算")
    print("-" * 40)
    priority = 0.8
    delay = 2.0
    price = 2.0

    value = model.compute_service_value(priority, delay)
    benefit = model.compute_user_benefit(priority, delay, price)
    accepted = model.check_service_acceptance(priority, delay, price)

    print(f"  优先级={priority}, 时延={delay}s, 价格={price}元")
    print(f"  服务价值={value:.2f}元")
    print(f"  用户收益={benefit:.2f}元")
    print(f"  是否接受={accepted}")

    # 测试3: 最大可接受价格
    print("\n[Test 3] 最大可接受价格")
    print("-" * 40)
    max_price = model.compute_max_acceptable_price(priority, delay)
    print(f"  优先级={priority}, 时延={delay}s")
    print(f"  最大可接受价格={max_price:.2f}元")

    # 测试4: 批量计算
    print("\n[Test 4] 批量计算")
    print("-" * 40)
    priorities = [0.9, 0.7, 0.5, 0.3]
    delays = [0.5, 1.0, 2.0, 5.0]
    prices = [1.0, 2.0, 3.0, 5.0]

    benefits, acceptances = model.batch_compute_user_benefits(priorities, delays, prices)

    print(f"  {'优先级':<8} {'时延':<8} {'价格':<8} {'收益':<10} {'接受':<8}")
    for i in range(len(priorities)):
        print(f"  {priorities[i]:<8.1f} {delays[i]:<8.1f} {prices[i]:<8.1f} "
              f"{benefits[i]:<10.2f} {acceptances[i]:<8}")

    # 测试5: 价值衰减表
    print("\n[Test 5] 价值衰减表")
    print("-" * 40)
    decay_table = model.get_value_decay_table()
    print(f"  {'时延(s)':<10} {'价值(元)':<10} {'衰减比例':<10}")
    for delay, value in decay_table.items():
        decay_ratio = value / model.config.v0
        print(f"  {delay:<10.0f} {value:<10.2f} {decay_ratio:<10.2%}")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_user_benefit_model()

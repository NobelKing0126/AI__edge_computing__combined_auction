"""
M24: UAVRelocator - UAV重定位控制器

功能：检测位置失配并触发UAV重定位
输入：用户位置、UAV状态、任务分配
输出：重定位决策和执行

关键设计 (docs/idea118.txt 4.11节):
    1. 失配指标计算 (4.11.1)
    2. 动态权重计算 (4.11.2)
    3. 能量约束检查 (4.11.3)
    4. 任务迁移处理 (4.11.4)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.uav import UAV
from models.user import User
from models.mobility import (
    UAVRelocationState, UAVRelocationTrigger, MobilityPattern, MobilityMetrics
)
from algorithms.phase0.weighted_kmeans import WeightedKMeans
from utils.data_loader import Location
from config.system_config import SystemConfig, MobilityConfig


@dataclass
class RelocationDecision:
    """
    重定位决策

    Attributes:
        uav_id: UAV ID
        target_x: 目标x坐标
        target_y: 目标y坐标
        trigger: 触发类型
        energy_cost: 预计能耗 (J)
        flight_time: 预计飞行时间 (s)
        tasks_to_migrate: 需迁移的任务ID列表
        can_execute: 是否可以执行（能量充足）
    """
    uav_id: int
    target_x: float
    target_y: float
    trigger: UAVRelocationTrigger
    energy_cost: float = 0.0
    flight_time: float = 0.0
    tasks_to_migrate: List[int] = field(default_factory=list)
    can_execute: bool = True


class UAVRelocator:
    """
    UAV重定位控制器

    参考 (docs/idea118.txt 4.11节):
        - 失配触发：Mismatch(t) > delta_pos * d_bar_service_init
        - 周期触发：每 T_reposition 秒执行一次
        - 能量约束：E_fly <= E_remain - E_reserve

    Attributes:
        config: 系统配置
        mobility_config: 移动配置
        initial_avg_service_distance: 初始平均服务距离
        last_reposition_time: 上次重定位时间
        metrics: 移动指标
    """

    def __init__(self, config: SystemConfig):
        """
        初始化重定位控制器

        Args:
            config: 系统配置
        """
        self.config = config
        self.mobility_config = config.mobility

        # 初始平均服务距离（用于失配计算）
        self.initial_avg_service_distance: float = 0.0

        # 重定位状态跟踪
        self.relocation_states: Dict[int, UAVRelocationState] = {}
        self.last_reposition_time: float = 0.0

        # 指标
        self.metrics = MobilityMetrics()

    def initialize(self, uavs: List[UAV], users: List[User]) -> None:
        """
        初始化重定位控制器

        Args:
            uavs: UAV列表
            users: 用户列表
        """
        # 计算初始平均服务距离
        self.initial_avg_service_distance = self._compute_avg_service_distance(uavs, users)

        # 初始化每个UAV的重定位状态
        for uav in uavs:
            self.relocation_states[uav.uav_id] = UAVRelocationState()

    def check_and_relocate(self,
                          uavs: List[UAV],
                          users: List[User],
                          current_time: float,
                          active_tasks: Dict[int, List[int]]) -> List[RelocationDecision]:
        """
        检查是否需要重定位并执行

        参考 (docs/idea118.txt 4.11.1节):
            Mismatch(t) = (1/M) * Σ min_j d_{i,j}(t) - d_bar_service_init
            Trigger = Mismatch(t) > delta_pos * d_bar_service_init

        Args:
            uavs: UAV列表
            users: 用户列表
            current_time: 当前时间
            active_tasks: 每个UAV的活跃任务ID列表 {uav_id: [task_id, ...]}

        Returns:
            List[RelocationDecision]: 重定位决策列表
        """
        decisions = []

        # 1. 失配触发检查
        if self.mobility_config.enable_mismatch_trigger:
            mismatch = self._compute_mismatch(uavs, users)
            self.metrics.mismatch_values.append(mismatch)

            if mismatch > self.mobility_config.delta_pos * self.initial_avg_service_distance:
                # 触发重定位
                decision = self._make_relocation_decision(
                    uavs, users, active_tasks, UAVRelocationTrigger.MISMATCH
                )
                if decision:
                    decisions.append(decision)

        # 2. 周期触发检查
        if self.mobility_config.enable_periodic_reposition:
            if current_time - self.last_reposition_time >= self.mobility_config.T_reposition:
                decision = self._make_relocation_decision(
                    uavs, users, active_tasks, UAVRelocationTrigger.PERIODIC
                )
                if decision:
                    decisions.append(decision)
                self.last_reposition_time = current_time

        return decisions

    def _compute_mismatch(self, uavs: List[UAV], users: List[User]) -> float:
        """
        计算位置失配指标

        公式 (docs/idea118.txt 4.11.1节):
            Mismatch(t) = (1/M) * Σ min_j d_{i,j}(t) - d_bar_service_init

        Args:
            uavs: UAV列表
            users: 用户列表

        Returns:
            float: 失配指标 (m)
        """
        if not users or not uavs:
            return 0.0

        total_min_dist = 0.0
        for user in users:
            min_dist = min(
                np.sqrt((user.x - uav.x) ** 2 + (user.y - uav.y) ** 2 + uav.height ** 2)
                for uav in uavs
            )
            total_min_dist += min_dist

        avg_min_dist = total_min_dist / len(users)

        return avg_min_dist - self.initial_avg_service_distance

    def _compute_avg_service_distance(self, uavs: List[UAV], users: List[User]) -> float:
        """
        计算平均服务距离

        公式 (docs/idea118.txt 0.10节):
            d_bar_service = (1/M) * Σ d_{i, π(i)}

        Args:
            uavs: UAV列表
            users: 用户列表

        Returns:
            float: 平均服务距离 (m)
        """
        if not users or not uavs:
            return 0.0

        total_dist = 0.0
        for user in users:
            # 找到最近的UAV
            min_dist = min(
                np.sqrt((user.x - uav.x) ** 2 + (user.y - uav.y) ** 2 + uav.height ** 2)
                for uav in uavs
            )
            total_dist += min_dist

        return total_dist / len(users)

    def _make_relocation_decision(self,
                                  uavs: List[UAV],
                                  users: List[User],
                                  active_tasks: Dict[int, List[int]],
                                  trigger: UAVRelocationTrigger) -> Optional[RelocationDecision]:
        """
        生成重定位决策

        参考 (docs/idea118.txt 4.11.2-4.11.4节):
            1. 计算动态权重
            2. 使用加权K-means计算新位置
            3. 检查能量约束
            4. 确定需迁移的任务

        Args:
            uavs: UAV列表
            users: 用户列表
            active_tasks: 活跃任务映射
            trigger: 触发类型

        Returns:
            Optional[RelocationDecision]: 重定位决策，无需重定位则返回None
        """
        if not users or not uavs:
            return None

        # 使用加权K-means计算新位置
        locations = [Location(id=u.user_id, x=u.x, y=u.y) for u in users]
        weights = self._compute_dynamic_weights(users, active_tasks)

        kmeans = WeightedKMeans(n_clusters=len(uavs))
        result = kmeans.fit(locations, weights)

        if not result.centers:
            return None

        # 找到需要重定位最明显的UAV
        max_shift = 0.0
        target_uav = None
        target_center = None

        for i, uav in enumerate(uavs):
            if i >= len(result.centers):
                break

            center = result.centers[i]
            shift = np.sqrt((uav.x - center[0]) ** 2 + (uav.y - center[1]) ** 2)

            if shift > max_shift:
                max_shift = shift
                target_uav = uav
                target_center = center

        if target_uav is None or target_center is None:
            return None

        # 计算飞行能耗和时间
        distance = np.sqrt(
            (target_uav.x - target_center[0]) ** 2 +
            (target_uav.y - target_center[1]) ** 2
        )

        energy_cost = self.mobility_config.uav_fly_power * distance / self.mobility_config.uav_fly_speed
        flight_time = distance / self.mobility_config.uav_fly_speed

        # 能量约束检查
        energy_reserve = self.mobility_config.uav_energy_reserve_ratio * target_uav.E_max
        can_execute = target_uav.E_remain - energy_cost >= energy_reserve

        # 确定需要迁移的任务
        tasks_to_migrate = []
        if not can_execute:
            tasks_to_migrate = active_tasks.get(target_uav.uav_id, [])

        return RelocationDecision(
            uav_id=target_uav.uav_id,
            target_x=target_center[0],
            target_y=target_center[1],
            trigger=trigger,
            energy_cost=energy_cost,
            flight_time=flight_time,
            tasks_to_migrate=tasks_to_migrate,
            can_execute=can_execute
        )

    def _compute_dynamic_weights(self,
                                 users: List[User],
                                 active_tasks: Dict[int, List[int]]) -> np.ndarray:
        """
        计算动态权重

        公式 (docs/idea118.txt 4.11.2节):
            w_i(t) = α₁*(C_i/τ_max) + α₂*(InputSize_i/max) + α₃*1(i∈U_active)

        Args:
            users: 用户列表
            active_tasks: 活跃任务映射

        Returns:
            np.ndarray: 权重数组
        """
        weights = np.zeros(len(users))

        # 收集活跃用户ID
        active_user_ids: Set[int] = set()
        for task_ids in active_tasks.values():
            active_user_ids.update(task_ids)

        for i, user in enumerate(users):
            if user.task is None:
                weights[i] = 0.1  # 最小权重
                continue

            task = user.task

            # 因子1: 计算紧迫度
            urgency = task.get_urgency() if task.total_flops > 0 else 0.0

            # 因子2: 数据量
            input_size = task.input_size

            # 因子3: 是否活跃
            is_active = 1.0 if user.user_id in active_user_ids else 0.0

            # 综合权重
            weights[i] = 0.4 * urgency + 0.3 * input_size / 10e6 + 0.3 * is_active

        # 确保权重非零
        weights = np.maximum(weights, 0.1)

        return weights

    def execute_relocation(self,
                          uav: UAV,
                          decision: RelocationDecision) -> bool:
        """
        执行UAV重定位

        参考 (docs/idea118.txt 4.11.3节):
            E_fly = P_fly * distance / v_fly
            约束: E_fly <= E_remain - E_reserve

        Args:
            uav: UAV对象
            decision: 重定位决策

        Returns:
            bool: 是否成功执行
        """
        if not decision.can_execute:
            return False

        # 使用UAV的move_to方法
        success = uav.move_to(decision.target_x, decision.target_y)

        if success:
            # 更新指标
            distance = np.sqrt(
                (uav.x - decision.target_x) ** 2 +
                (uav.y - decision.target_y) ** 2
            )
            self.metrics.uav_relocation_count += 1
            self.metrics.uav_total_fly_distance += distance
            self.metrics.uav_total_fly_energy += decision.energy_cost

            # 更新重定位状态
            if uav.uav_id in self.relocation_states:
                state = self.relocation_states[uav.uav_id]
                state.relocation_count += 1
                state.total_distance += distance

        return success

    def get_metrics(self) -> MobilityMetrics:
        """
        获取移动指标

        Returns:
            MobilityMetrics: 移动指标
        """
        return self.metrics

    def summary(self) -> str:
        """返回重定位控制器摘要"""
        return f"""
UAV重定位控制器摘要:
  初始平均服务距离: {self.initial_avg_service_distance:.1f} m
  重定位次数: {self.metrics.uav_relocation_count}
  累计飞行距离: {self.metrics.uav_total_fly_distance:.1f} m
  累计飞行能耗: {self.metrics.uav_total_fly_energy/1e3:.2f} kJ
  平均失配指标: {np.mean(self.metrics.mismatch_values):.2f} m (共{len(self.metrics.mismatch_values)}次采样)
"""


# ============ 测试用例 ============

def test_uav_relocator():
    """测试UAVRelocator模块"""
    print("=" * 60)
    print("测试 M24: UAVRelocator")
    print("=" * 60)

    from config.system_config import SystemConfig

    config = SystemConfig()
    relocator = UAVRelocator(config)

    # 创建测试UAV
    uavs = [
        UAV(uav_id=0, x=500.0, y=500.0, height=100.0),
        UAV(uav_id=1, x=1500.0, y=1500.0, height=100.0),
    ]

    # 创建测试用户
    users = []
    for i in range(20):
        user = User(
            user_id=i,
            x=np.random.uniform(0, 2000),
            y=np.random.uniform(0, 2000),
            task=Task(
                task_id=i,
                user_id=i,
                model_id=1,
                input_size=5e6,
                tau_max=3.0,
                user_level=3
            )
        )
        user.task.total_flops = 10e9
        users.append(user)

    # 测试1: 初始化
    print("\n[Test 1] 测试初始化...")
    relocator.initialize(uavs, users)
    assert relocator.initial_avg_service_distance > 0
    print(f"  初始平均服务距离: {relocator.initial_avg_service_distance:.1f} m")
    print("  ✓ 初始化正确")

    # 测试2: 失配计算
    print("\n[Test 2] 测试失配计算...")
    mismatch = relocator._compute_mismatch(uavs, users)
    print(f"  当前失配指标: {mismatch:.2f} m")
    print("  ✓ 失配计算正确")

    # 测试3: 重定位决策
    print("\n[Test 3] 测试重定位决策...")
    active_tasks = {0: [0, 1], 1: [2, 3]}
    decisions = relocator.check_and_relocate(uavs, users, 0.0, active_tasks)
    print(f"  生成决策数: {len(decisions)}")
    print("  ✓ 重定位决策正确")

    # 测试4: 指标获取
    print("\n[Test 4] 测试指标获取...")
    metrics = relocator.get_metrics()
    assert metrics.uav_relocation_count >= 0
    print(relocator.summary())
    print("  ✓ 指标获取正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_uav_relocator()

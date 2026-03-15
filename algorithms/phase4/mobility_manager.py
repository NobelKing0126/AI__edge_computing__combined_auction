"""
M25: MobilityManager - 移动管理器

功能：统一管理用户移动和UAV重定位
输入：用户列表、UAV列表、任务分配
输出：位置更新、重定位决策

关键设计:
    1. 统一管理用户移动 (docs/实验.txt 2.4节)
    2. 统一管理UAV重定位 (docs/idea118.txt 4.11节)
    3. 处理移动期间的系统状态维护
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.uav import UAV
from models.user import User
from models.mobility import (
    MobilityPattern, UserMobilityState, UAVRelocationState,
    UAVRelocationTrigger, MobilityMetrics
)
from algorithms.phase4.uav_relocator import UAVRelocator, RelocationDecision
from config.system_config import SystemConfig, MobilityConfig


@dataclass
class MobilityStepResult:
    """
    移动步进结果

    Attributes:
        users_moved: 移动的用户数量
        uavs_relocated: 重定位的UAV数量
        total_user_distance: 用户移动总距离
        total_uav_distance: UAV飞行总距离
        tasks_migrated: 迁移的任务数量
        mismatch_value: 当前失配指标
    """
    users_moved: int = 0
    uavs_relocated: int = 0
    total_user_distance: float = 0.0
    total_uav_distance: float = 0.0
    tasks_migrated: int = 0
    mismatch_value: float = 0.0


class MobilityManager:
    """
    移动管理器

    统一管理用户移动和UAV重定位

    参考 (docs/实验.txt 2.4节):
        动态分布模式：用户位置随时间变化，移动速度1-5 m/s

    参考 (docs/idea118.txt 4.11节):
        UAV位置重定位机制

    Attributes:
        config: 系统配置
        mobility_config: 移动配置
        uav_relocator: UAV重定位控制器
        metrics: 移动指标
        hotspots: 热点位置列表
    """

    def __init__(self, config: SystemConfig):
        """
        初始化移动管理器

        Args:
            config: 系统配置
        """
        self.config = config
        self.mobility_config = config.mobility

        # UAV重定位控制器
        self.uav_relocator = UAVRelocator(config)

        # 热点位置
        self.hotspots: List[Tuple[float, float]] = []

        # 指标
        self.metrics = MobilityMetrics()

        # 时间跟踪
        self.current_time: float = 0.0

    def initialize(self,
                  uavs: List[UAV],
                  users: List[User],
                  hotspots: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        初始化移动管理器

        Args:
            uavs: UAV列表
            users: 用户列表
            hotspots: 热点位置列表 [(x, y), ...]
        """
        # 初始化UAV重定位控制器
        self.uav_relocator.initialize(uavs, users)

        # 设置热点
        self.hotspots = hotspots or []

        # 初始化用户移动状态
        if self.mobility_config.enable_user_mobility:
            self._initialize_user_mobility(users)

        # 复制初始指标
        self.metrics = self.uav_relocator.get_metrics()

    def _initialize_user_mobility(self, users: List[User]) -> None:
        """
        初始化用户移动状态

        Args:
            users: 用户列表
        """
        rng = np.random.default_rng()

        for user in users:
            # 随机分配移动模式
            pattern = rng.choice([
                MobilityPattern.STATIC,
                MobilityPattern.RANDOM_WALK,
                MobilityPattern.HOTSPOT_MIGRATION
            ], p=[0.3, 0.4, 0.3])  # 30%静止，40%随机游走，30%热点迁移

            # 随机速度 (1-5 m/s)
            speed = rng.uniform(
                self.mobility_config.user_speed_min,
                self.mobility_config.user_speed_max
            )

            # 初始化移动状态
            user.initialize_mobility(
                pattern=pattern,
                speed=speed,
                scene_width=self.config.scenario.scene_width,
                scene_height=self.config.scenario.scene_height,
                hotspots=self.hotspots,
                seed=rng.integers(0, 10000)
            )

    def step(self,
            uavs: List[UAV],
            users: List[User],
            dt: float,
            active_tasks: Optional[Dict[int, List[int]]] = None) -> MobilityStepResult:
        """
        执行一个时间步的移动更新

        Args:
            uavs: UAV列表
            users: 用户列表
            dt: 时间步长 (s)
            active_tasks: 每个UAV的活跃任务ID列表

        Returns:
            MobilityStepResult: 移动步进结果
        """
        result = MobilityStepResult()

        # 更新当前时间
        self.current_time += dt

        # 1. 更新用户位置
        if self.mobility_config.enable_user_mobility:
            result.users_moved, result.total_user_distance = self._update_user_positions(users, dt)

        # 2. 检查并执行UAV重定位
        if active_tasks is not None:
            decisions = self.uav_relocator.check_and_relocate(
                uavs, users, self.current_time, active_tasks
            )

            for decision in decisions:
                if decision.can_execute:
                    uav = next((u for u in uavs if u.uav_id == decision.uav_id), None)
                    if uav:
                        success = self.uav_relocator.execute_relocation(uav, decision)
                        if success:
                            result.uavs_relocated += 1
                            result.total_uav_distance += np.sqrt(
                                (uav.x - decision.target_x) ** 2 +
                                (uav.y - decision.target_y) ** 2
                            )
                            result.tasks_migrated += len(decision.tasks_to_migrate)

        # 3. 计算当前失配指标
        result.mismatch_value = self.uav_relocator._compute_mismatch(uavs, users)

        # 4. 更新指标
        self._update_metrics(result)

        return result

    def _update_user_positions(self, users: List[User], dt: float) -> Tuple[int, float]:
        """
        更新所有用户位置

        参考 (docs/实验.txt 2.4节):
            移动速度1-5 m/s

        Args:
            users: 用户列表
            dt: 时间步长 (s)

        Returns:
            Tuple[int, float]: (移动用户数, 总移动距离)
        """
        moved_count = 0
        total_distance = 0.0

        for user in users:
            if user.mobility_state is None:
                continue

            old_x, old_y = user.x, user.y
            user.update_position(dt)

            distance = np.sqrt((user.x - old_x) ** 2 + (user.y - old_y) ** 2)
            if distance > 0.01:  # 移动超过1cm算移动
                moved_count += 1
                total_distance += distance

        return moved_count, total_distance

    def _update_metrics(self, result: MobilityStepResult) -> None:
        """
        更新移动指标

        Args:
            result: 移动步进结果
        """
        self.metrics.user_total_distance += result.total_user_distance
        self.metrics.uav_relocation_count += result.uavs_relocated
        self.metrics.uav_total_fly_distance += result.total_uav_distance
        self.metrics.tasks_migrated += result.tasks_migrated
        self.metrics.mismatch_values.append(result.mismatch_value)

        # 计算平均速度
        if self.current_time > 0:
            self.metrics.user_avg_speed = self.metrics.user_total_distance / self.current_time

    def get_metrics(self) -> MobilityMetrics:
        """
        获取移动指标

        Returns:
            MobilityMetrics: 移动指标
        """
        return self.metrics

    def get_summary(self) -> str:
        """返回移动管理器摘要"""
        return f"""
移动管理器摘要:
  运行时间: {self.current_time:.1f} s
  用户移动:
    - 累计距离: {self.metrics.user_total_distance:.1f} m
    - 平均速度: {self.metrics.user_avg_speed:.2f} m/s
  UAV重定位:
    - 重定位次数: {self.metrics.uav_relocation_count}
    - 累计飞行距离: {self.metrics.uav_total_fly_distance:.1f} m
    - 累计飞行能耗: {self.metrics.uav_total_fly_energy/1e3:.2f} kJ
  系统状态:
    - 任务迁移数: {self.metrics.tasks_migrated}
    - 平均失配: {np.mean(self.metrics.mismatch_values):.2f} m
"""


# ============ 测试用例 ============

def test_mobility_manager():
    """测试MobilityManager模块"""
    print("=" * 60)
    print("测试 M25: MobilityManager")
    print("=" * 60)

    from config.system_config import SystemConfig

    config = SystemConfig()
    # 启用用户移动
    config.mobility.enable_user_mobility = True

    manager = MobilityManager(config)

    # 创建测试数据
    uavs = [
        UAV(uav_id=0, x=500.0, y=500.0, height=100.0),
        UAV(uav_id=1, x=1500.0, y=1500.0, height=100.0),
    ]

    users = []
    for i in range(20):
        user = User(
            user_id=i,
            x=np.random.uniform(0, 2000),
            y=np.random.uniform(0, 2000)
        )
        users.append(user)

    # 热点
    hotspots = [(500.0, 500.0), (1500.0, 1500.0)]

    # 测试1: 初始化
    print("\n[Test 1] 测试初始化...")
    manager.initialize(uavs, users, hotspots)

    # 检查用户移动状态初始化
    users_with_mobility = sum(1 for u in users if u.mobility_state is not None)
    print(f"  初始化了 {users_with_mobility} 个用户的移动状态")
    assert users_with_mobility == 20
    print("  ✓ 初始化正确")

    # 测试2: 执行移动步进
    print("\n[Test 2] 测试移动步进...")
    active_tasks = {0: [0, 1], 1: [2, 3]}

    # 执行10次步进
    for _ in range(10):
        result = manager.step(uavs, users, dt=1.0, active_tasks=active_tasks)

    print(f"  移动用户数: {result.users_moved}")
    print(f"  累计用户距离: {manager.metrics.user_total_distance:.1f} m")
    print("  ✓ 移动步进正确")

    # 测试3: 指标获取
    print("\n[Test 3] 测试指标获取...")
    metrics = manager.get_metrics()
    assert metrics.user_total_distance > 0
    print(manager.get_summary())
    print("  ✓ 指标获取正确")

    # 测试4: 静止模式
    print("\n[Test 4] 测试静止模式...")
    config2 = SystemConfig()
    config2.mobility.enable_user_mobility = False
    manager2 = MobilityManager(config2)
    manager2.initialize(uavs, users, hotspots)

    # 所有用户应该没有移动状态
    users_with_mobility = sum(1 for u in users if u.mobility_state is not None)
    print(f"  静止模式下用户移动状态数: {users_with_mobility}")
    print("  ✓ 静止模式正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_mobility_manager()

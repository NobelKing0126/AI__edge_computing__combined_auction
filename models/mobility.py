"""
M08: Mobility - 移动模块

功能：定义用户和UAV的移动模式与状态
输入：移动模式配置
输出：移动状态更新

关键设计 (docs/实验.txt 2.4节):
    动态分布模式：用户位置随时间变化，移动速度1-5 m/s

UAV重定位 (docs/idea118.txt 4.11节):
    失配触发 + 周期触发
    移动期间任务处理规则
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto
import numpy as np


class MobilityPattern(Enum):
    """
    移动模式枚举

    参考 (docs/实验.txt 2.4节):
        - 静止模式：用户位置固定
        - 随机游走：用户随机移动，速度1-5 m/s
        - 热点迁移：用户向热点区域移动
    """
    STATIC = auto()              # 静止
    RANDOM_WALK = auto()         # 随机游走 (1-5 m/s)
    HOTSPOT_MIGRATION = auto()   # 热点迁移
    MIXED = auto()               # 混合模式（部分静止，部分移动）


class UAVRelocationTrigger(Enum):
    """
    UAV重定位触发类型

    参考 (docs/idea118.txt 4.11.1节)
    """
    MISMATCH = auto()      # 失配触发
    PERIODIC = auto()      # 周期触发
    MANUAL = auto()        # 手动触发
    ENERGY_BASED = auto()  # 能量优化触发


@dataclass
class UserMobilityState:
    """
    用户移动状态

    Attributes:
        pattern: 移动模式
        speed: 当前速度 (m/s)，范围1-5 m/s
        direction: 移动方向 (弧度)
        scene_width: 场景宽度 (m)
        scene_height: 场景高度 (m)
        hotspots: 热点位置列表 [(x, y), ...]
        target_hotspot_idx: 目标热点索引（热点迁移模式）
        hotspot_arrival_time: 在热点停留的剩余时间 (s)
        rng: 随机数生成器
    """
    pattern: MobilityPattern = MobilityPattern.STATIC
    speed: float = 0.0  # m/s
    direction: float = 0.0  # 弧度
    scene_width: float = 2000.0
    scene_height: float = 2000.0
    hotspots: List[Tuple[float, float]] = field(default_factory=list)
    target_hotspot_idx: Optional[int] = None
    hotspot_arrival_time: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    # 边界反弹计数
    boundary_hits: int = 0

    def __post_init__(self):
        """初始化方向"""
        if self.pattern == MobilityPattern.RANDOM_WALK and self.speed > 0:
            self.direction = self.rng.uniform(0, 2 * np.pi)


@dataclass
class UAVRelocationState:
    """
    UAV重定位状态

    Attributes:
        is_relocating: 是否正在重定位
        trigger: 触发类型
        target_x: 目标x坐标
        target_y: 目标y坐标
        start_time: 开始时间
        estimated_duration: 预计耗时
        tasks_to_migrate: 需迁移的任务ID列表
        energy_cost: 飞行能耗
    """
    is_relocating: bool = False
    trigger: Optional[UAVRelocationTrigger] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    start_time: float = 0.0
    estimated_duration: float = 0.0
    tasks_to_migrate: List[int] = field(default_factory=list)
    energy_cost: float = 0.0

    # 历史记录
    relocation_count: int = 0
    total_distance: float = 0.0


@dataclass
class MobilityMetrics:
    """
    移动相关指标

    Attributes:
        user_total_distance: 用户累计移动距离
        user_avg_speed: 用户平均速度
        uav_relocation_count: UAV重定位次数
        uav_total_fly_distance: UAV累计飞行距离
        uav_total_fly_energy: UAV累计飞行能耗
        mismatch_values: 失配指标历史
        tasks_migrated: 因移动迁移的任务数
    """
    user_total_distance: float = 0.0
    user_avg_speed: float = 0.0
    uav_relocation_count: int = 0
    uav_total_fly_distance: float = 0.0
    uav_total_fly_energy: float = 0.0
    mismatch_values: List[float] = field(default_factory=list)
    tasks_migrated: int = 0

    def summary(self) -> str:
        """返回指标摘要"""
        return f"""
移动指标摘要:
  用户累计移动距离: {self.user_total_distance:.1f} m
  用户平均速度: {self.user_avg_speed:.2f} m/s
  UAV重定位次数: {self.uav_relocation_count}
  UAV累计飞行距离: {self.uav_total_fly_distance:.1f} m
  UAV累计飞行能耗: {self.uav_total_fly_energy/1e3:.2f} kJ
  平均失配指标: {np.mean(self.mismatch_values):.2f} m (若>0)
  因移动迁移任务数: {self.tasks_migrated}
"""


# ============ 测试用例 ============

def test_mobility_enums():
    """测试移动枚举"""
    print("=" * 60)
    print("测试 M08: Mobility Enums")
    print("=" * 60)

    # 测试1: MobilityPattern
    print("\n[Test 1] 测试移动模式枚举...")
    assert MobilityPattern.STATIC.value == 1
    assert MobilityPattern.RANDOM_WALK.value == 2
    assert MobilityPattern.HOTSPOT_MIGRATION.value == 3
    print("  ✓ 移动模式枚举正确")

    # 测试2: UAVRelocationTrigger
    print("\n[Test 2] 测试UAV重定位触发枚举...")
    assert UAVRelocationTrigger.MISMATCH.value == 1
    assert UAVRelocationTrigger.PERIODIC.value == 2
    print("  ✓ 重定位触发枚举正确")

    # 测试3: UserMobilityState
    print("\n[Test 3] 测试用户移动状态...")
    state = UserMobilityState(
        pattern=MobilityPattern.RANDOM_WALK,
        speed=3.0
    )
    assert state.speed == 3.0
    assert 0 <= state.direction <= 2 * np.pi
    print(f"  速度: {state.speed} m/s, 方向: {np.degrees(state.direction):.1f}°")
    print("  ✓ 用户移动状态正确")

    # 测试4: UAVRelocationState
    print("\n[Test 4] 测试UAV重定位状态...")
    uav_state = UAVRelocationState(
        is_relocating=True,
        trigger=UAVRelocationTrigger.MISMATCH,
        target_x=1000.0,
        target_y=1000.0
    )
    assert uav_state.is_relocating
    print("  ✓ UAV重定位状态正确")

    # 测试5: MobilityMetrics
    print("\n[Test 5] 测试移动指标...")
    metrics = MobilityMetrics(
        user_total_distance=1500.0,
        uav_relocation_count=3
    )
    summary = metrics.summary()
    assert "1500.0 m" in summary
    print(summary)
    print("  ✓ 移动指标正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_mobility_enums()

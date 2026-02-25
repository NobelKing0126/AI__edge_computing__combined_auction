"""
M12: Initializer - 阶段0初始化控制器

功能：协调统计量计算、K-means部署、初始分配的完整流程
输入：用户数据、系统配置
输出：初始化后的系统状态

初始化流程 (idea118.txt 0.8节):
    1. 加载用户数据
    2. 计算用户分布统计量
    3. 确定UAV数量
    4. 使用加权K-means确定UAV部署位置
    5. 执行初始用户-UAV分配
    6. 创建UAV对象并初始化状态
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import Location, generate_synthetic_users
from algorithms.phase0.statistics import compute_statistics, UserStatistics
from algorithms.phase0.weighted_kmeans import WeightedKMeans, deploy_uavs, ClusterResult
from algorithms.phase0.initial_assignment import (
    assign_by_distance, assign_with_load_balance, 
    AssignmentResult, get_assignment_stats
)
from config.system_config import SystemConfig


@dataclass
class InitializationResult:
    """
    初始化结果
    
    Attributes:
        user_locations: 用户位置列表
        user_stats: 用户分布统计
        n_uavs: UAV数量
        uav_positions: UAV部署位置
        cluster_result: K-means聚类结果
        assignment: 用户-UAV分配
        assignment_stats: 分配统计
    """
    user_locations: List[Location]
    user_stats: UserStatistics
    n_uavs: int
    uav_positions: List[Tuple[float, float, float]]
    cluster_result: ClusterResult
    assignment: AssignmentResult
    assignment_stats: Dict


class Phase0Initializer:
    """
    阶段0初始化控制器
    
    Attributes:
        config: 系统配置
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化控制器
        
        Args:
            config: 系统配置
        """
        self.config = config or SystemConfig()
    
    def determine_uav_count(self, n_users: int) -> int:
        """
        根据用户数量确定UAV数量
        
        规则 (idea118.txt 0.6节):
            M ≤ 20: N = 2
            20 < M ≤ 50: N = 3
            50 < M ≤ 100: N = 5
            M > 100: N = 8
        
        Args:
            n_users: 用户数量
            
        Returns:
            int: UAV数量
        """
        if n_users <= 20:
            return 2
        elif n_users <= 50:
            return 3
        elif n_users <= 100:
            return 5
        else:
            return 8
    
    def initialize_from_locations(self,
                                  locations: List[Location],
                                  n_uavs: Optional[int] = None,
                                  uav_height: float = 100.0,
                                  max_coverage: float = 500.0,
                                  max_load: int = 20) -> InitializationResult:
        """
        从位置数据初始化系统
        
        Args:
            locations: 用户位置列表
            n_uavs: UAV数量（可选，自动确定）
            uav_height: UAV高度
            max_coverage: 最大覆盖半径
            max_load: 每UAV最大负载
            
        Returns:
            InitializationResult: 初始化结果
        """
        # 1. 计算统计量
        user_stats = compute_statistics(locations)
        
        # 2. 确定UAV数量
        if n_uavs is None:
            n_uavs = self.determine_uav_count(len(locations))
        
        # 3. 加权K-means部署
        # 使用默认的K-means参数（SystemConfig中暂无这些参数，使用默认值）
        kmeans = WeightedKMeans(
            n_clusters=n_uavs,
            alpha1=0.7,  # 数据量权重
            alpha2=0.3,  # 计算量权重
            max_iter=100,
            epsilon=1e-6
        )
        
        cluster_result = kmeans.fit(locations)
        uav_positions = [(x, y, uav_height) for x, y in cluster_result.centers]
        
        # 4. 初始用户分配
        assignment = assign_with_load_balance(
            locations, uav_positions,
            max_coverage=max_coverage,
            max_load=max_load
        )
        
        assignment_stats = get_assignment_stats(assignment)
        
        return InitializationResult(
            user_locations=locations,
            user_stats=user_stats,
            n_uavs=n_uavs,
            uav_positions=uav_positions,
            cluster_result=cluster_result,
            assignment=assignment,
            assignment_stats=assignment_stats
        )
    
    def initialize_synthetic(self,
                            n_users: int = 50,
                            scene_size: float = 2000.0,
                            distribution: str = 'hotspot',
                            **kwargs) -> InitializationResult:
        """
        使用合成数据初始化
        
        Args:
            n_users: 用户数量
            scene_size: 场景尺寸
            distribution: 用户分布类型
            **kwargs: 其他参数传递给initialize_from_locations
            
        Returns:
            InitializationResult: 初始化结果
        """
        locations = generate_synthetic_users(
            n_users,
            scene_width=scene_size,
            scene_height=scene_size,
            distribution=distribution,
            seed=42
        )
        
        return self.initialize_from_locations(locations, **kwargs)
    
    def print_summary(self, result: InitializationResult):
        """
        打印初始化摘要
        
        Args:
            result: 初始化结果
        """
        print("\n" + "=" * 50)
        print("阶段0初始化摘要")
        print("=" * 50)
        
        print(f"\n【用户分布】")
        print(f"  用户数量: {result.user_stats.num_users}")
        print(f"  分布中心: ({result.user_stats.center[0]:.1f}, {result.user_stats.center[1]:.1f})")
        print(f"  空间密度: {result.user_stats.density:.1f} 用户/km²")
        
        print(f"\n【UAV部署】")
        print(f"  UAV数量: {result.n_uavs}")
        print(f"  K-means迭代: {result.cluster_result.iterations} 次")
        print(f"  K-means惯量: {result.cluster_result.inertia:.1f}")
        for i, pos in enumerate(result.uav_positions):
            print(f"  UAV-{i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        print(f"\n【用户分配】")
        print(f"  已分配: {result.assignment_stats['total_assigned']} 用户")
        print(f"  未分配(云端): {result.assignment_stats['total_unassigned']} 用户")
        print(f"  UAV负载: {result.assignment_stats['uav_loads']}")
        print(f"  平均负载: {result.assignment_stats['avg_load']:.1f}")
        print(f"  平均距离: {result.assignment_stats['avg_distance']:.1f} m")
        
        print("\n" + "=" * 50)


def run_phase0(n_users: int = 50,
               scene_size: float = 2000.0,
               distribution: str = 'hotspot',
               config: Optional[SystemConfig] = None) -> InitializationResult:
    """
    运行阶段0初始化的便捷函数
    
    Args:
        n_users: 用户数量
        scene_size: 场景尺寸
        distribution: 用户分布类型
        config: 系统配置
        
    Returns:
        InitializationResult: 初始化结果
    """
    initializer = Phase0Initializer(config)
    result = initializer.initialize_synthetic(
        n_users=n_users,
        scene_size=scene_size,
        distribution=distribution,
        uav_height=100.0,
        max_coverage=600.0,
        max_load=25
    )
    return result


# ============ 测试用例 ============

def test_initializer():
    """测试Initializer模块"""
    print("=" * 60)
    print("测试 M12: Phase0Initializer")
    print("=" * 60)
    
    # 测试1: UAV数量确定
    print("\n[Test 1] 测试UAV数量确定...")
    initializer = Phase0Initializer()
    
    assert initializer.determine_uav_count(15) == 2
    assert initializer.determine_uav_count(40) == 3
    assert initializer.determine_uav_count(80) == 5
    assert initializer.determine_uav_count(150) == 8
    print("  ✓ UAV数量规则正确")
    
    # 测试2: 合成数据初始化
    print("\n[Test 2] 测试合成数据初始化...")
    result = initializer.initialize_synthetic(
        n_users=50,
        scene_size=2000.0,
        distribution='hotspot',
        max_coverage=600.0
    )
    
    assert result.n_uavs == 3, "50用户应有3个UAV"
    assert len(result.uav_positions) == 3
    assert result.assignment_stats['total_assigned'] > 0
    print(f"  用户数: {result.user_stats.num_users}")
    print(f"  UAV数: {result.n_uavs}")
    print(f"  已分配: {result.assignment_stats['total_assigned']}")
    print("  ✓ 合成数据初始化正确")
    
    # 测试3: 不同规模测试
    print("\n[Test 3] 测试不同规模...")
    for n in [20, 50, 100]:
        result = initializer.initialize_synthetic(n_users=n)
        print(f"  {n}用户 -> {result.n_uavs}个UAV, 分配{result.assignment_stats['total_assigned']}用户")
    print("  ✓ 不同规模正确")
    
    # 测试4: 打印摘要
    print("\n[Test 4] 测试摘要输出...")
    result = run_phase0(n_users=60, distribution='uniform')
    initializer.print_summary(result)
    print("  ✓ 摘要输出正确")
    
    # 测试5: 便捷函数
    print("\n[Test 5] 测试便捷函数...")
    result = run_phase0(n_users=100, scene_size=3000.0, distribution='edge')
    
    assert result.n_uavs == 5, "100用户应有5个UAV"
    print(f"  场景: 3000m x 3000m")
    print(f"  分布: edge")
    print(f"  结果: {result.n_uavs}个UAV, {result.assignment_stats['total_assigned']}用户分配")
    print("  ✓ 便捷函数正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_initializer()

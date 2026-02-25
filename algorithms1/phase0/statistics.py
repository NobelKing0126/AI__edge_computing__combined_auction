"""
M09: Statistics - 用户分布统计量计算

功能：计算用户分布的统计特征，用于加权K-means
输入：用户位置列表
输出：统计量字典

关键公式 (idea118.txt 0.5节):
    需求强度: demand_intensity = sum(weights) / area
    空间密度: density_k = N_k / Area_k
    加权中心: center = sum(w_i * pos_i) / sum(w_i)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import Location


@dataclass
class UserStatistics:
    """
    用户分布统计结果
    
    Attributes:
        num_users: 用户数量
        center: 分布中心 (x, y)
        std: 标准差 (x_std, y_std)
        bbox: 边界框 (x_min, x_max, y_min, y_max)
        area: 覆盖面积 (m²)
        density: 空间密度 (用户/km²)
        demand_intensity: 需求强度
        grid_stats: 网格统计 (可选)
    """
    num_users: int
    center: Tuple[float, float]
    std: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    area: float
    density: float
    demand_intensity: float
    grid_stats: Optional[Dict] = None


def compute_statistics(locations: List[Location],
                       weights: Optional[List[float]] = None) -> UserStatistics:
    """
    计算用户分布的统计量
    
    Args:
        locations: 用户位置列表
        weights: 用户权重列表（可选，默认为1）
        
    Returns:
        UserStatistics: 统计结果
    """
    if not locations:
        return UserStatistics(
            num_users=0,
            center=(0.0, 0.0),
            std=(0.0, 0.0),
            bbox=(0.0, 0.0, 0.0, 0.0),
            area=0.0,
            density=0.0,
            demand_intensity=0.0
        )
    
    n = len(locations)
    xs = np.array([loc.x for loc in locations])
    ys = np.array([loc.y for loc in locations])
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
    
    # 加权中心
    total_weight = np.sum(weights)
    center_x = np.sum(weights * xs) / total_weight
    center_y = np.sum(weights * ys) / total_weight
    
    # 标准差
    std_x = np.sqrt(np.sum(weights * (xs - center_x) ** 2) / total_weight)
    std_y = np.sqrt(np.sum(weights * (ys - center_y) ** 2) / total_weight)
    
    # 边界框
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    # 面积和密度
    width = max(x_max - x_min, 1.0)  # 避免除零
    height = max(y_max - y_min, 1.0)
    area = width * height
    density = n / (area / 1e6)  # 用户/km²
    
    # 需求强度
    demand_intensity = total_weight / (area / 1e6)
    
    return UserStatistics(
        num_users=n,
        center=(center_x, center_y),
        std=(std_x, std_y),
        bbox=(x_min, x_max, y_min, y_max),
        area=area,
        density=density,
        demand_intensity=demand_intensity
    )


def compute_grid_statistics(locations: List[Location],
                           grid_size: int = 10,
                           weights: Optional[List[float]] = None) -> Dict[Tuple[int, int], Dict]:
    """
    计算网格化统计量
    
    Args:
        locations: 用户位置列表
        grid_size: 网格划分数量 (grid_size x grid_size)
        weights: 用户权重
        
    Returns:
        Dict: 网格统计 {(i, j): {'count': n, 'weight': w, 'density': d}}
    """
    if not locations:
        return {}
    
    n = len(locations)
    xs = np.array([loc.x for loc in locations])
    ys = np.array([loc.y for loc in locations])
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
    
    # 边界
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    # 网格尺寸
    cell_width = (x_max - x_min) / grid_size
    cell_height = (y_max - y_min) / grid_size
    cell_area = max(cell_width * cell_height, 1.0)
    
    # 统计每个网格
    grid_stats = {}
    for i in range(grid_size):
        for j in range(grid_size):
            grid_stats[(i, j)] = {'count': 0, 'weight': 0.0, 'density': 0.0}
    
    for k, loc in enumerate(locations):
        # 确定所属网格
        i = min(int((loc.x - x_min) / max(cell_width, 1e-6)), grid_size - 1)
        j = min(int((loc.y - y_min) / max(cell_height, 1e-6)), grid_size - 1)
        i = max(0, i)
        j = max(0, j)
        
        grid_stats[(i, j)]['count'] += 1
        grid_stats[(i, j)]['weight'] += weights[k]
    
    # 计算密度
    for key in grid_stats:
        grid_stats[key]['density'] = grid_stats[key]['count'] / (cell_area / 1e6)
    
    return grid_stats


def find_hotspots(locations: List[Location],
                  weights: Optional[List[float]] = None,
                  num_hotspots: int = 3,
                  radius: float = 200.0) -> List[Tuple[float, float, float]]:
    """
    找出用户分布的热点区域
    
    使用简单的密度峰值检测
    
    Args:
        locations: 用户位置列表
        weights: 用户权重
        num_hotspots: 热点数量
        radius: 热点半径
        
    Returns:
        List[Tuple]: 热点列表 [(x, y, intensity), ...]
    """
    if not locations:
        return []
    
    n = len(locations)
    xs = np.array([loc.x for loc in locations])
    ys = np.array([loc.y for loc in locations])
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
    
    # 计算每个点的局部密度
    local_density = np.zeros(n)
    for i in range(n):
        # 计算半径内的加权用户数
        distances = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
        mask = distances <= radius
        local_density[i] = np.sum(weights[mask])
    
    # 选择密度最高的点作为热点候选
    candidates = []
    for i in np.argsort(local_density)[::-1]:
        # 检查是否与已有热点距离足够远
        is_new = True
        for cx, cy, _ in candidates:
            if np.sqrt((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2) < radius * 2:
                is_new = False
                break
        
        if is_new:
            candidates.append((xs[i], ys[i], local_density[i]))
            if len(candidates) >= num_hotspots:
                break
    
    return candidates


def estimate_uav_count(locations: List[Location],
                       max_coverage_radius: float = 500.0) -> int:
    """
    估算所需UAV数量
    
    基于覆盖问题的简单估算
    
    Args:
        locations: 用户位置列表
        max_coverage_radius: 最大覆盖半径
        
    Returns:
        int: 建议的UAV数量
    """
    if not locations:
        return 0
    
    stats = compute_statistics(locations)
    
    # 基于面积估算
    coverage_per_uav = np.pi * max_coverage_radius ** 2
    area_based = max(1, int(np.ceil(stats.area / coverage_per_uav)))
    
    # 基于用户数估算
    users_per_uav = 20  # 每个UAV服务约20个用户
    user_based = max(1, int(np.ceil(stats.num_users / users_per_uav)))
    
    # 取较大值
    return max(area_based, user_based)


# ============ 测试用例 ============

def test_statistics():
    """测试Statistics模块"""
    print("=" * 60)
    print("测试 M09: Statistics")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_users = 100
    locations = [
        Location(id=i, x=np.random.normal(1000, 300), y=np.random.normal(1000, 300))
        for i in range(n_users)
    ]
    
    # 测试1: 基本统计
    print("\n[Test 1] 测试基本统计...")
    stats = compute_statistics(locations)
    
    assert stats.num_users == 100, "用户数量错误"
    assert abs(stats.center[0] - 1000) < 100, "中心x偏差过大"
    assert abs(stats.center[1] - 1000) < 100, "中心y偏差过大"
    print(f"  用户数: {stats.num_users}")
    print(f"  中心: ({stats.center[0]:.1f}, {stats.center[1]:.1f})")
    print(f"  标准差: ({stats.std[0]:.1f}, {stats.std[1]:.1f})")
    print(f"  密度: {stats.density:.1f} 用户/km²")
    print("  ✓ 基本统计正确")
    
    # 测试2: 加权统计
    print("\n[Test 2] 测试加权统计...")
    weights = [1.0 if loc.x > 1000 else 2.0 for loc in locations]
    weighted_stats = compute_statistics(locations, weights)
    
    # 左侧权重更高，中心应该偏左
    assert weighted_stats.center[0] < stats.center[0], "加权中心应偏向高权重区域"
    print(f"  加权中心: ({weighted_stats.center[0]:.1f}, {weighted_stats.center[1]:.1f})")
    print(f"  需求强度: {weighted_stats.demand_intensity:.1f}")
    print("  ✓ 加权统计正确")
    
    # 测试3: 网格统计
    print("\n[Test 3] 测试网格统计...")
    grid = compute_grid_statistics(locations, grid_size=5)
    
    total_count = sum(g['count'] for g in grid.values())
    assert total_count == n_users, "网格用户总数应等于原始用户数"
    print(f"  网格数: {len(grid)}")
    print(f"  中心网格(2,2)用户数: {grid[(2,2)]['count']}")
    print("  ✓ 网格统计正确")
    
    # 测试4: 热点检测
    print("\n[Test 4] 测试热点检测...")
    hotspots = find_hotspots(locations, num_hotspots=3)
    
    assert len(hotspots) <= 3, "热点数量不应超过指定值"
    print(f"  检测到 {len(hotspots)} 个热点:")
    for i, (x, y, intensity) in enumerate(hotspots):
        print(f"    热点{i+1}: ({x:.1f}, {y:.1f}), 强度={intensity:.1f}")
    print("  ✓ 热点检测正确")
    
    # 测试5: UAV数量估算
    print("\n[Test 5] 测试UAV数量估算...")
    estimated = estimate_uav_count(locations, max_coverage_radius=400)
    
    assert 1 <= estimated <= 20, "估算数量应在合理范围内"
    print(f"  建议UAV数量: {estimated}")
    print("  ✓ UAV数量估算正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_statistics()

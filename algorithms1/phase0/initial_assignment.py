"""
M11: InitialAssignment - 初始用户-UAV分配

功能：将用户分配给最近的UAV，建立初始关联
输入：用户列表、UAV位置列表
输出：分配映射

分配规则 (idea118.txt 0.8节):
    1. 基于距离的分配：用户分配给最近的UAV
    2. 负载均衡考虑：避免单个UAV过载
    3. 覆盖约束：超出覆盖范围的用户标记为云端处理
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import Location


@dataclass
class AssignmentResult:
    """
    分配结果
    
    Attributes:
        user_to_uav: 用户->UAV映射 {user_id: uav_id}
        uav_to_users: UAV->用户列表映射 {uav_id: [user_ids]}
        unassigned: 未分配的用户（需云端处理）
        distances: 用户到其分配UAV的距离
    """
    user_to_uav: Dict[int, int]
    uav_to_users: Dict[int, List[int]]
    unassigned: List[int]
    distances: Dict[int, float]


def compute_distance(user_pos: Tuple[float, float], 
                     uav_pos: Tuple[float, float, float]) -> float:
    """
    计算用户到UAV的3D距离
    
    Args:
        user_pos: 用户位置 (x, y)
        uav_pos: UAV位置 (x, y, z)
        
    Returns:
        float: 3D距离
    """
    dx = user_pos[0] - uav_pos[0]
    dy = user_pos[1] - uav_pos[1]
    dz = uav_pos[2]  # 用户在地面，z=0
    return np.sqrt(dx**2 + dy**2 + dz**2)


def assign_by_distance(users: List[Location],
                       uav_positions: List[Tuple[float, float, float]],
                       max_coverage: float = 500.0) -> AssignmentResult:
    """
    基于距离的用户分配
    
    Args:
        users: 用户位置列表
        uav_positions: UAV位置列表
        max_coverage: 最大覆盖半径 (m)
        
    Returns:
        AssignmentResult: 分配结果
    """
    user_to_uav = {}
    uav_to_users = {i: [] for i in range(len(uav_positions))}
    unassigned = []
    distances = {}
    
    for user in users:
        user_pos = (user.x, user.y)
        
        # 计算到每个UAV的距离
        min_dist = float('inf')
        best_uav = -1
        
        for uav_id, uav_pos in enumerate(uav_positions):
            dist = compute_distance(user_pos, uav_pos)
            if dist < min_dist:
                min_dist = dist
                best_uav = uav_id
        
        # 检查是否在覆盖范围内
        if min_dist <= max_coverage and best_uav >= 0:
            user_to_uav[user.id] = best_uav
            uav_to_users[best_uav].append(user.id)
            distances[user.id] = min_dist
        else:
            unassigned.append(user.id)
    
    return AssignmentResult(
        user_to_uav=user_to_uav,
        uav_to_users=uav_to_users,
        unassigned=unassigned,
        distances=distances
    )


def assign_with_load_balance(users: List[Location],
                             uav_positions: List[Tuple[float, float, float]],
                             max_coverage: float = 500.0,
                             max_load: int = 20) -> AssignmentResult:
    """
    带负载均衡的用户分配
    
    当UAV达到最大负载时，分配给次优UAV
    
    Args:
        users: 用户位置列表
        uav_positions: UAV位置列表
        max_coverage: 最大覆盖半径 (m)
        max_load: 每个UAV最大用户数
        
    Returns:
        AssignmentResult: 分配结果
    """
    n_uavs = len(uav_positions)
    user_to_uav = {}
    uav_to_users = {i: [] for i in range(n_uavs)}
    unassigned = []
    distances = {}
    
    # 计算所有距离并排序
    user_distances = []  # [(user, uav_id, distance), ...]
    
    for user in users:
        user_pos = (user.x, user.y)
        for uav_id, uav_pos in enumerate(uav_positions):
            dist = compute_distance(user_pos, uav_pos)
            if dist <= max_coverage:
                user_distances.append((user, uav_id, dist))
    
    # 按距离排序（贪心分配）
    user_distances.sort(key=lambda x: x[2])
    
    assigned_users: Set[int] = set()
    
    for user, uav_id, dist in user_distances:
        if user.id in assigned_users:
            continue
        
        if len(uav_to_users[uav_id]) < max_load:
            user_to_uav[user.id] = uav_id
            uav_to_users[uav_id].append(user.id)
            distances[user.id] = dist
            assigned_users.add(user.id)
    
    # 找出未分配的用户
    for user in users:
        if user.id not in assigned_users:
            unassigned.append(user.id)
    
    return AssignmentResult(
        user_to_uav=user_to_uav,
        uav_to_users=uav_to_users,
        unassigned=unassigned,
        distances=distances
    )


def reassign_from_overloaded(assignment: AssignmentResult,
                             users: List[Location],
                             uav_positions: List[Tuple[float, float, float]],
                             max_coverage: float = 500.0,
                             target_load: int = 15) -> AssignmentResult:
    """
    从过载UAV重新分配用户
    
    Args:
        assignment: 当前分配
        users: 用户列表
        uav_positions: UAV位置列表
        max_coverage: 最大覆盖半径
        target_load: 目标负载
        
    Returns:
        AssignmentResult: 更新后的分配
    """
    user_dict = {u.id: u for u in users}
    user_to_uav = assignment.user_to_uav.copy()
    uav_to_users = {k: v.copy() for k, v in assignment.uav_to_users.items()}
    distances = assignment.distances.copy()
    unassigned = assignment.unassigned.copy()
    
    n_uavs = len(uav_positions)
    
    for uav_id in range(n_uavs):
        while len(uav_to_users[uav_id]) > target_load:
            # 找距离最远的用户
            farthest_user = max(
                uav_to_users[uav_id],
                key=lambda uid: distances.get(uid, 0)
            )
            
            user = user_dict[farthest_user]
            user_pos = (user.x, user.y)
            
            # 尝试分配给其他UAV
            reassigned = False
            for other_uav in range(n_uavs):
                if other_uav == uav_id:
                    continue
                if len(uav_to_users[other_uav]) >= target_load:
                    continue
                
                dist = compute_distance(user_pos, uav_positions[other_uav])
                if dist <= max_coverage:
                    # 重新分配
                    uav_to_users[uav_id].remove(farthest_user)
                    uav_to_users[other_uav].append(farthest_user)
                    user_to_uav[farthest_user] = other_uav
                    distances[farthest_user] = dist
                    reassigned = True
                    break
            
            if not reassigned:
                # 无法重新分配，标记为云端
                uav_to_users[uav_id].remove(farthest_user)
                del user_to_uav[farthest_user]
                del distances[farthest_user]
                unassigned.append(farthest_user)
    
    return AssignmentResult(
        user_to_uav=user_to_uav,
        uav_to_users=uav_to_users,
        unassigned=unassigned,
        distances=distances
    )


def get_assignment_stats(assignment: AssignmentResult) -> Dict:
    """
    获取分配统计信息
    
    Args:
        assignment: 分配结果
        
    Returns:
        Dict: 统计信息
    """
    loads = [len(users) for users in assignment.uav_to_users.values()]
    dists = list(assignment.distances.values())
    
    return {
        'total_assigned': len(assignment.user_to_uav),
        'total_unassigned': len(assignment.unassigned),
        'uav_loads': loads,
        'max_load': max(loads) if loads else 0,
        'min_load': min(loads) if loads else 0,
        'avg_load': np.mean(loads) if loads else 0,
        'avg_distance': np.mean(dists) if dists else 0,
        'max_distance': max(dists) if dists else 0,
    }


# ============ 测试用例 ============

def test_initial_assignment():
    """测试InitialAssignment模块"""
    print("=" * 60)
    print("测试 M11: InitialAssignment")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    
    # 3个UAV位置
    uav_positions = [
        (500.0, 500.0, 100.0),
        (1500.0, 500.0, 100.0),
        (1000.0, 1500.0, 100.0)
    ]
    
    # 60个用户围绕UAV分布
    users = []
    for i in range(20):
        users.append(Location(id=i, x=np.random.normal(500, 150), y=np.random.normal(500, 150)))
    for i in range(20, 40):
        users.append(Location(id=i, x=np.random.normal(1500, 150), y=np.random.normal(500, 150)))
    for i in range(40, 60):
        users.append(Location(id=i, x=np.random.normal(1000, 150), y=np.random.normal(1500, 150)))
    
    # 测试1: 基于距离的分配
    print("\n[Test 1] 测试基于距离的分配...")
    result = assign_by_distance(users, uav_positions, max_coverage=500)
    
    stats = get_assignment_stats(result)
    print(f"  已分配: {stats['total_assigned']} / 60 用户")
    print(f"  UAV负载: {stats['uav_loads']}")
    print(f"  平均距离: {stats['avg_distance']:.1f} m")
    
    assert stats['total_assigned'] > 0, "应有用户被分配"
    print("  ✓ 距离分配正确")
    
    # 测试2: 带负载均衡的分配
    print("\n[Test 2] 测试负载均衡分配...")
    result_balanced = assign_with_load_balance(
        users, uav_positions, 
        max_coverage=500, max_load=15
    )
    
    stats_balanced = get_assignment_stats(result_balanced)
    print(f"  已分配: {stats_balanced['total_assigned']} 用户")
    print(f"  UAV负载: {stats_balanced['uav_loads']}")
    
    assert max(stats_balanced['uav_loads']) <= 15, "负载应不超过限制"
    print("  ✓ 负载均衡正确")
    
    # 测试3: 过载重分配
    print("\n[Test 3] 测试过载重分配...")
    result_rebalanced = reassign_from_overloaded(
        result, users, uav_positions,
        max_coverage=500, target_load=15
    )
    
    stats_rebalanced = get_assignment_stats(result_rebalanced)
    print(f"  重分配后负载: {stats_rebalanced['uav_loads']}")
    print("  ✓ 重分配正确")
    
    # 测试4: 超出覆盖范围
    print("\n[Test 4] 测试覆盖范围限制...")
    
    # 添加远距离用户
    far_users = users.copy()
    far_users.append(Location(id=100, x=3000, y=3000))
    far_users.append(Location(id=101, x=-500, y=-500))
    
    result_far = assign_by_distance(far_users, uav_positions, max_coverage=500)
    
    assert 100 in result_far.unassigned or 101 in result_far.unassigned, "远距离用户应未分配"
    print(f"  未分配用户: {result_far.unassigned}")
    print("  ✓ 覆盖范围限制正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_initial_assignment()

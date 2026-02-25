"""
M20: Phase3Controller - 阶段3控制器

功能：协调组合拍卖的完整流程
输入：投标集合、UAV资源、任务优先级
输出：中标结果、执行计划

阶段3流程 (idea118.txt 3.15节):
    1. 收集所有投标
    2. 执行组合拍卖求解
    3. 生成中标通知
    4. 更新资源状态
    5. 生成执行计划
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.phase3.combinatorial_auction import (
    LagrangianAuction, BidInfo, UAVResource, AuctionResult, AuctionStatus
)


@dataclass
class ExecutionPlan:
    """
    执行计划
    
    Attributes:
        task_id: 任务ID
        uav_id: 执行UAV ID
        split_layer: 切分层
        f_edge: 边缘算力
        f_cloud: 云端算力
        T_deadline: 截止时间
        checkpoint_layer: Checkpoint层
        priority: 优先级
    """
    task_id: int
    uav_id: int
    split_layer: int
    f_edge: float
    f_cloud: float
    T_deadline: float
    checkpoint_layer: Optional[int]
    priority: float


@dataclass
class Phase3Result:
    """
    阶段3结果
    
    Attributes:
        auction_result: 拍卖结果
        execution_plans: 执行计划列表
        unserved_tasks: 未服务任务
        stats: 统计信息
    """
    auction_result: AuctionResult
    execution_plans: List[ExecutionPlan]
    unserved_tasks: List[int]
    stats: Dict


class Phase3Controller:
    """
    阶段3控制器
    
    Attributes:
        auction_solver: 拍卖求解器
    """
    
    def __init__(self,
                 epsilon_0: float = 0.5,
                 max_iterations: int = 100):
        """
        初始化控制器
        
        Args:
            epsilon_0: 初始步长
            max_iterations: 最大迭代次数
        """
        self.auction_solver = LagrangianAuction(
            epsilon_0=epsilon_0,
            max_iterations=max_iterations
        )
    
    def convert_bids(self,
                     candidate_bids: Dict[int, List[Dict]]) -> Dict[int, List[BidInfo]]:
        """
        转换投标格式
        
        Args:
            candidate_bids: 候选投标 {task_id: [bid_dict, ...]}
            
        Returns:
            Dict: 转换后的投标
        """
        converted = {}
        
        for task_id, bids in candidate_bids.items():
            converted[task_id] = []
            for i, bid in enumerate(bids):
                converted[task_id].append(BidInfo(
                    task_id=task_id,
                    bid_id=i,
                    uav_id=bid.get('uav_id', 0),
                    utility=bid.get('utility', 0.5),
                    f_required=bid.get('f_required', 1e9),
                    E_required=bid.get('E_required', 10e3),
                    T_predict=bid.get('T_predict', 1.0),
                    priority=bid.get('priority', 0.5)
                ))
        
        return converted
    
    def create_execution_plans(self,
                               auction_result: AuctionResult,
                               bid_details: Dict[int, Dict]) -> List[ExecutionPlan]:
        """
        创建执行计划
        
        Args:
            auction_result: 拍卖结果
            bid_details: 投标详情 {(task_id, bid_id): detail_dict}
            
        Returns:
            List[ExecutionPlan]: 执行计划列表
        """
        plans = []
        
        for task_id, (uav_id, bid_info) in auction_result.task_assignments.items():
            key = (task_id, bid_info.bid_id)
            detail = bid_details.get(key, {})
            
            plans.append(ExecutionPlan(
                task_id=task_id,
                uav_id=uav_id,
                split_layer=detail.get('split_layer', 0),
                f_edge=bid_info.f_required,
                f_cloud=detail.get('f_cloud', 0),
                T_deadline=detail.get('deadline', 3.0),
                checkpoint_layer=detail.get('checkpoint_layer'),
                priority=bid_info.priority
            ))
        
        # 按优先级排序
        plans.sort(key=lambda p: p.priority, reverse=True)
        
        return plans
    
    def run(self,
            bids: Dict[int, List[BidInfo]],
            uav_resources: List[UAVResource],
            bid_details: Optional[Dict] = None) -> Phase3Result:
        """
        运行阶段3
        
        Args:
            bids: 投标集合
            uav_resources: UAV资源列表
            bid_details: 投标详情
            
        Returns:
            Phase3Result: 阶段3结果
        """
        # 执行拍卖
        auction_result = self.auction_solver.solve(bids, uav_resources)
        
        # 创建执行计划
        if bid_details is None:
            bid_details = {}
        
        execution_plans = self.create_execution_plans(auction_result, bid_details)
        
        # 统计信息
        stats = self._compute_stats(auction_result, uav_resources)
        
        return Phase3Result(
            auction_result=auction_result,
            execution_plans=execution_plans,
            unserved_tasks=auction_result.unassigned_tasks,
            stats=stats
        )
    
    def _compute_stats(self,
                       result: AuctionResult,
                       resources: List[UAVResource]) -> Dict:
        """
        计算统计信息
        """
        total_tasks = len(result.winners) + len(result.unassigned_tasks)
        
        # UAV利用率
        uav_util = {}
        for r in resources:
            allocs = result.uav_allocations.get(r.uav_id, [])
            f_used = sum(bid.f_required for _, bid in allocs)
            E_used = sum(bid.E_required for _, bid in allocs)
            
            uav_util[r.uav_id] = {
                'f_util': f_used / r.F_available if r.F_available > 0 else 0,
                'E_util': E_used / r.E_available if r.E_available > 0 else 0,
                'task_count': len(allocs)
            }
        
        return {
            'total_tasks': total_tasks,
            'assigned_tasks': len(result.winners),
            'unassigned_tasks': len(result.unassigned_tasks),
            'total_utility': result.total_utility,
            'status': result.status.value,
            'iterations': result.iterations,
            'uav_utilization': uav_util
        }
    
    def print_summary(self, result: Phase3Result):
        """
        打印阶段3摘要
        """
        print("\n" + "=" * 50)
        print("阶段3摘要 - 组合拍卖结果")
        print("=" * 50)
        
        print(f"\n【拍卖统计】")
        print(f"  总任务数: {result.stats['total_tasks']}")
        print(f"  已分配: {result.stats['assigned_tasks']}")
        print(f"  未分配: {result.stats['unassigned_tasks']}")
        print(f"  总效用: {result.stats['total_utility']:.3f}")
        print(f"  状态: {result.stats['status']}")
        print(f"  迭代次数: {result.stats['iterations']}")
        
        print(f"\n【UAV利用率】")
        for uav_id, util in result.stats['uav_utilization'].items():
            print(f"  UAV-{uav_id}: 算力={util['f_util']*100:.1f}%, "
                  f"能量={util['E_util']*100:.1f}%, "
                  f"任务={util['task_count']}个")
        
        if result.execution_plans:
            print(f"\n【执行计划】(前5个)")
            for plan in result.execution_plans[:5]:
                print(f"  任务{plan.task_id}: UAV-{plan.uav_id}, "
                      f"优先级={plan.priority:.2f}")
        
        print("\n" + "=" * 50)


def run_phase3(bids: Dict[int, List[BidInfo]],
               uav_resources: List[UAVResource]) -> Phase3Result:
    """
    运行阶段3的便捷函数
    """
    controller = Phase3Controller()
    return controller.run(bids, uav_resources)


# ============ 测试用例 ============

def test_phase3_controller():
    """测试Phase3Controller模块"""
    print("=" * 60)
    print("测试 M20: Phase3Controller")
    print("=" * 60)
    
    controller = Phase3Controller()
    
    # 创建测试数据
    np.random.seed(42)
    
    # UAV资源
    uav_resources = [
        UAVResource(uav_id=0, F_available=10e9, E_available=150e3),
        UAVResource(uav_id=1, F_available=10e9, E_available=150e3),
        UAVResource(uav_id=2, F_available=10e9, E_available=150e3),
    ]
    
    # 投标
    bids = {}
    for task_id in range(10):
        task_bids = []
        for bid_id in range(3):
            uav_id = (task_id + bid_id) % 3
            task_bids.append(BidInfo(
                task_id=task_id,
                bid_id=bid_id,
                uav_id=uav_id,
                utility=np.random.uniform(0.4, 0.9),
                f_required=np.random.uniform(1e9, 3e9),
                E_required=np.random.uniform(10e3, 30e3),
                T_predict=np.random.uniform(1.0, 3.0),
                priority=np.random.uniform(0.3, 0.9)
            ))
        bids[task_id] = task_bids
    
    # 测试1: 完整流程
    print("\n[Test 1] 测试完整流程...")
    result = controller.run(bids, uav_resources)
    
    assert result.auction_result is not None
    assert result.stats is not None
    print(f"  分配任务: {result.stats['assigned_tasks']}/{result.stats['total_tasks']}")
    print(f"  总效用: {result.stats['total_utility']:.3f}")
    print("  ✓ 完整流程正确")
    
    # 测试2: 执行计划
    print("\n[Test 2] 测试执行计划...")
    assert len(result.execution_plans) == result.stats['assigned_tasks']
    
    # 验证按优先级排序
    priorities = [p.priority for p in result.execution_plans]
    assert priorities == sorted(priorities, reverse=True), "应按优先级降序"
    print(f"  生成 {len(result.execution_plans)} 个执行计划")
    print("  ✓ 执行计划正确")
    
    # 测试3: 便捷函数
    print("\n[Test 3] 测试便捷函数...")
    result2 = run_phase3(bids, uav_resources)
    
    assert result2.stats['assigned_tasks'] == result.stats['assigned_tasks']
    print("  ✓ 便捷函数正确")
    
    # 测试4: 摘要输出
    print("\n[Test 4] 测试摘要输出...")
    controller.print_summary(result)
    print("  ✓ 摘要输出正确")
    
    # 测试5: 投标格式转换
    print("\n[Test 5] 测试投标格式转换...")
    dict_bids = {
        0: [
            {'uav_id': 0, 'utility': 0.8, 'f_required': 2e9, 'E_required': 20e3},
            {'uav_id': 1, 'utility': 0.6, 'f_required': 1.5e9, 'E_required': 15e3},
        ]
    }
    
    converted = controller.convert_bids(dict_bids)
    assert len(converted[0]) == 2
    assert isinstance(converted[0][0], BidInfo)
    print("  ✓ 格式转换正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_phase3_controller()

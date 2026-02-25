"""
M15: Phase1Controller - 阶段1控制器

功能：协调优先级计算和拍卖方选举的完整流程
输入：任务列表、UAV状态
输出：任务优先级、拍卖方

阶段1流程 (idea118.txt 1.9节):
    1. 收集用户任务请求
    2. 计算任务优先级
    3. 任务分类（高/中/低）
    4. 分布式选举拍卖方
    5. 返回阶段1结果
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.phase1.priority import (
    PriorityCalculator, TaskPriority, PriorityLevel
)
from algorithms.phase1.election import (
    AuctioneerElector, UAVState, ElectionResult, ElectionStatus
)
from config.system_config import SystemConfig, PriorityConfig


@dataclass
class Phase1Result:
    """
    阶段1结果
    
    Attributes:
        task_priorities: 任务优先级列表
        classified_tasks: 分类后的任务 {等级: [任务ID]}
        priority_stats: 优先级统计
        election_result: 选举结果
        auctioneer_id: 拍卖方ID
    """
    task_priorities: List[TaskPriority]
    classified_tasks: Dict[PriorityLevel, List[int]]
    priority_stats: Dict
    election_result: ElectionResult
    auctioneer_id: Optional[int]


class Phase1Controller:
    """
    阶段1控制器
    
    Attributes:
        priority_calculator: 优先级计算器
        elector: 选举器
        config: 系统配置
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化控制器
        
        Args:
            config: 系统配置
        """
        self.config = config or SystemConfig()
        self.priority_calculator = PriorityCalculator(self.config.priority)
        self.elector = AuctioneerElector(
            w1=0.25, w2=0.30, w3=0.25, w4=0.20
        )
    
    def process_tasks(self, tasks: List[Dict]) -> Tuple[List[TaskPriority], Dict, Dict]:
        """
        处理任务优先级
        
        Args:
            tasks: 任务列表
            
        Returns:
            Tuple: (优先级列表, 分类结果, 统计信息)
        """
        priorities = self.priority_calculator.compute_batch_priorities(tasks)
        classified = self.priority_calculator.classify_tasks(priorities)
        stats = self.priority_calculator.get_priority_stats(priorities)
        
        return priorities, classified, stats
    
    def elect_auctioneer(self,
                         uav_states: List[UAVState],
                         scene_center: Optional[Tuple[float, float]] = None) -> ElectionResult:
        """
        选举拍卖方
        
        Args:
            uav_states: UAV状态列表
            scene_center: 场景中心
            
        Returns:
            ElectionResult: 选举结果
        """
        return self.elector.elect(uav_states, scene_center)
    
    def run(self,
            tasks: List[Dict],
            uav_states: List[UAVState],
            scene_center: Optional[Tuple[float, float]] = None) -> Phase1Result:
        """
        运行阶段1完整流程
        
        Args:
            tasks: 任务列表
            uav_states: UAV状态列表
            scene_center: 场景中心
            
        Returns:
            Phase1Result: 阶段1结果
        """
        # 1. 计算任务优先级
        priorities, classified, stats = self.process_tasks(tasks)
        
        # 2. 选举拍卖方
        election_result = self.elect_auctioneer(uav_states, scene_center)
        
        return Phase1Result(
            task_priorities=priorities,
            classified_tasks=classified,
            priority_stats=stats,
            election_result=election_result,
            auctioneer_id=election_result.auctioneer_id
        )
    
    def print_summary(self, result: Phase1Result):
        """
        打印阶段1摘要
        
        Args:
            result: 阶段1结果
        """
        print("\n" + "=" * 50)
        print("阶段1摘要 - 优先级计算与拍卖方选举")
        print("=" * 50)
        
        print(f"\n【任务优先级】")
        print(f"  总任务数: {result.priority_stats['count']}")
        print(f"  高优先级: {result.priority_stats['high_count']} 个")
        print(f"  中优先级: {result.priority_stats['medium_count']} 个")
        print(f"  低优先级: {result.priority_stats['low_count']} 个")
        print(f"  得分范围: [{result.priority_stats['score_min']:.3f}, {result.priority_stats['score_max']:.3f}]")
        
        print(f"\n【拍卖方选举】")
        print(f"  选举状态: {result.election_result.status.value}")
        if result.auctioneer_id is not None:
            print(f"  当选拍卖方: UAV-{result.auctioneer_id}")
            print(f"  拍卖方得分: {result.election_result.scores[result.auctioneer_id]:.4f}")
        else:
            print(f"  选举失败，无拍卖方")
        
        print(f"  投票详情: {result.election_result.votes}")
        
        print("\n" + "=" * 50)


def run_phase1(tasks: List[Dict],
               uav_states: List[UAVState],
               config: Optional[SystemConfig] = None) -> Phase1Result:
    """
    运行阶段1的便捷函数
    
    Args:
        tasks: 任务列表
        uav_states: UAV状态列表
        config: 系统配置
        
    Returns:
        Phase1Result: 阶段1结果
    """
    controller = Phase1Controller(config)
    return controller.run(tasks, uav_states)


# ============ 测试用例 ============

def test_phase1_controller():
    """测试Phase1Controller模块"""
    print("=" * 60)
    print("测试 M15: Phase1Controller")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    
    # 任务数据
    tasks = []
    for i in range(30):
        tasks.append({
            'task_id': i,
            'user_id': i,
            'data_size': np.random.uniform(1e6, 10e6),
            'compute_size': np.random.uniform(5e9, 50e9),
            'deadline': np.random.uniform(1.0, 5.0),
            'user_level': np.random.randint(1, 6)
        })
    
    # UAV状态
    uav_states = [
        UAVState(
            uav_id=0,
            energy=400e3, energy_max=500e3,
            compute_cap=8e9, compute_max=10e9,
            position=(500, 500, 100),
            load=5, load_max=20
        ),
        UAVState(
            uav_id=1,
            energy=450e3, energy_max=500e3,
            compute_cap=9e9, compute_max=10e9,
            position=(1000, 1000, 100),
            load=3, load_max=20
        ),
        UAVState(
            uav_id=2,
            energy=350e3, energy_max=500e3,
            compute_cap=7e9, compute_max=10e9,
            position=(1500, 500, 100),
            load=8, load_max=20
        ),
    ]
    
    controller = Phase1Controller()
    
    # 测试1: 任务处理
    print("\n[Test 1] 测试任务处理...")
    priorities, classified, stats = controller.process_tasks(tasks)
    
    assert len(priorities) == 30, "应有30个优先级结果"
    assert sum(len(v) for v in classified.values()) == 30, "分类后总数应正确"
    
    print(f"  处理了 {len(priorities)} 个任务")
    print(f"  高/中/低优先级: {stats['high_count']}/{stats['medium_count']}/{stats['low_count']}")
    print("  ✓ 任务处理正确")
    
    # 测试2: 拍卖方选举
    print("\n[Test 2] 测试拍卖方选举...")
    election_result = controller.elect_auctioneer(uav_states)
    
    assert election_result.auctioneer_id is not None
    print(f"  当选拍卖方: UAV-{election_result.auctioneer_id}")
    print("  ✓ 选举正确")
    
    # 测试3: 完整流程
    print("\n[Test 3] 测试完整流程...")
    result = controller.run(tasks, uav_states)
    
    assert result.task_priorities is not None
    assert result.election_result is not None
    assert result.auctioneer_id is not None
    
    print(f"  任务数: {len(result.task_priorities)}")
    print(f"  拍卖方: UAV-{result.auctioneer_id}")
    print("  ✓ 完整流程正确")
    
    # 测试4: 便捷函数
    print("\n[Test 4] 测试便捷函数...")
    result2 = run_phase1(tasks, uav_states)
    
    assert result2.auctioneer_id == result.auctioneer_id
    print("  ✓ 便捷函数正确")
    
    # 测试5: 摘要输出
    print("\n[Test 5] 测试摘要输出...")
    controller.print_summary(result)
    print("  ✓ 摘要输出正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_phase1_controller()

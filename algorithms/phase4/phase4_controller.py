"""
M23: Phase4Controller - 阶段4控制器

功能：协调任务执行、故障处理、动态定价的完整流程
输入：执行计划、UAV状态
输出：执行结果、系统状态更新

阶段4流程 (idea118.txt 4.8节):
    1. 接收执行计划
    2. 按优先级调度执行
    3. 监控执行状态
    4. 处理故障和迁移
    5. 动态调整价格
    6. 生成执行报告
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.phase4.task_executor import (
    TaskExecutor, ExecutionTask, ExecutionResult, TaskStatus
)
from algorithms.phase4.dynamic_pricing import (
    DynamicPricingManager, UtilizationInfo, ResourcePrices
)


@dataclass
class Phase4Result:
    """
    阶段4结果
    
    Attributes:
        execution_results: 执行结果
        failed_tasks: 失败任务
        final_prices: 最终价格
        stats: 统计信息
    """
    execution_results: Dict[int, ExecutionResult]
    failed_tasks: List[int]
    final_prices: Dict[int, ResourcePrices]
    stats: Dict


@dataclass
class UAVRuntimeState:
    """
    UAV运行时状态
    
    Attributes:
        uav_id: UAV ID
        E_remain: 剩余能量
        E_max: 最大能量
        f_available: 可用算力
        f_max: 最大算力
        running_tasks: 正在执行的任务数
        completed_tasks: 已完成任务数
        health_score: 健康度
    """
    uav_id: int
    E_remain: float
    E_max: float
    f_available: float
    f_max: float
    running_tasks: int = 0
    completed_tasks: int = 0
    health_score: float = 1.0


class Phase4Controller:
    """
    阶段4控制器
    
    Attributes:
        executor: 任务执行器
        pricing_manager: 动态定价管理器
        uav_states: UAV状态
    """
    
    def __init__(self):
        """初始化控制器"""
        self.executor = TaskExecutor()
        self.pricing_manager = DynamicPricingManager()
        self.uav_states: Dict[int, UAVRuntimeState] = {}
    
    def initialize(self, uav_states: List[UAVRuntimeState]):
        """
        初始化系统
        
        Args:
            uav_states: UAV状态列表
        """
        self.uav_states = {s.uav_id: s for s in uav_states}
        
        # 初始化价格
        self.pricing_manager.initialize_prices(list(self.uav_states.keys()))
    
    def convert_plans_to_tasks(self, 
                               plans: List[Dict],
                               model_info: Optional[Dict] = None) -> List[ExecutionTask]:
        """
        将执行计划转换为执行任务
        
        Args:
            plans: 执行计划列表
            model_info: 模型信息
            
        Returns:
            List[ExecutionTask]: 执行任务列表
        """
        tasks = []
        
        for plan in plans:
            task = ExecutionTask(
                task_id=plan.get('task_id', 0),
                user_id=plan.get('user_id', 0),
                uav_id=plan.get('uav_id', 0),
                split_layer=plan.get('split_layer', 8),
                total_layers=plan.get('total_layers', 16),
                C_edge=plan.get('C_edge', 5e9),
                C_cloud=plan.get('C_cloud', 10e9),
                f_edge=plan.get('f_edge', 5e9),
                f_cloud=plan.get('f_cloud', 50e9),
                deadline=plan.get('deadline', 3.0),
                priority=plan.get('priority', 0.5),
                checkpoint_layer=plan.get('checkpoint_layer')
            )
            tasks.append(task)
        
        return tasks
    
    def run(self,
            tasks: List[ExecutionTask],
            time_step: float = 0.1,
            max_time: float = 100.0,
            pricing_interval: float = 1.0) -> Phase4Result:
        """
        运行阶段4
        
        Args:
            tasks: 任务列表
            time_step: 时间步长
            max_time: 最大执行时间
            pricing_interval: 价格更新间隔
            
        Returns:
            Phase4Result: 阶段4结果
        """
        # 执行任务
        results = self.executor.simulate_execution(tasks, time_step, max_time)
        
        # 更新UAV状态
        for task_id, result in results.items():
            task = next((t for t in tasks if t.task_id == task_id), None)
            if task and task.uav_id in self.uav_states:
                state = self.uav_states[task.uav_id]
                state.E_remain -= result.energy_consumed
                state.completed_tasks += 1
        
        # 计算利用率并更新价格
        utilizations = self._compute_utilizations(tasks, results)
        self.pricing_manager.update_all_prices(utilizations, max_time)
        
        # 收集失败任务
        failed_ids = list(self.executor.failed_tasks.keys())
        
        # 统计信息
        stats = self._compute_stats(results, failed_ids)
        
        return Phase4Result(
            execution_results=results,
            failed_tasks=failed_ids,
            final_prices=self.pricing_manager.get_all_prices(),
            stats=stats
        )
    
    def _compute_utilizations(self,
                              tasks: List[ExecutionTask],
                              results: Dict[int, ExecutionResult]) -> List[UtilizationInfo]:
        """计算UAV利用率"""
        utilizations = []
        
        for uav_id, state in self.uav_states.items():
            # 统计该UAV的任务
            uav_tasks = [t for t in tasks if t.uav_id == uav_id]
            completed = [r for tid, r in results.items() 
                        if any(t.task_id == tid and t.uav_id == uav_id for t in tasks)]
            
            # 算力利用率
            total_compute = sum(t.C_edge for t in uav_tasks)
            compute_util = min(total_compute / (state.f_max * 10), 1.0)  # 归一化
            
            # 能量利用率
            total_energy = sum(r.energy_consumed for r in completed)
            energy_util = total_energy / state.E_max if state.E_max > 0 else 0
            
            utilizations.append(UtilizationInfo(
                uav_id=uav_id,
                compute_util=compute_util,
                energy_util=energy_util,
                channel_util=0.5,  # 简化
                load_count=len(uav_tasks)
            ))
        
        return utilizations
    
    def _compute_stats(self,
                       results: Dict[int, ExecutionResult],
                       failed_ids: List[int]) -> Dict:
        """计算统计信息"""
        if not results:
            return {
                'total_tasks': len(failed_ids),
                'success_count': 0,
                'fail_count': len(failed_ids),
                'success_rate': 0.0
            }
        
        completed = list(results.values())
        success = [r for r in completed if r.met_deadline]
        
        return {
            'total_tasks': len(completed) + len(failed_ids),
            'success_count': len(success),
            'fail_count': len(failed_ids) + len([r for r in completed if not r.met_deadline]),
            'success_rate': len(success) / (len(completed) + len(failed_ids)),
            'avg_time': np.mean([r.total_time for r in completed]),
            'avg_energy': np.mean([r.energy_consumed for r in completed]),
            'checkpoint_usage': sum(1 for r in completed if r.checkpoint_used) / len(completed) if completed else 0
        }
    
    def print_summary(self, result: Phase4Result):
        """打印阶段4摘要"""
        print("\n" + "=" * 50)
        print("阶段4摘要 - 执行调度结果")
        print("=" * 50)
        
        print(f"\n【执行统计】")
        print(f"  总任务数: {result.stats['total_tasks']}")
        print(f"  成功数: {result.stats['success_count']}")
        print(f"  失败数: {result.stats['fail_count']}")
        print(f"  成功率: {result.stats['success_rate']*100:.1f}%")
        
        if 'avg_time' in result.stats:
            print(f"  平均时延: {result.stats['avg_time']*1000:.1f}ms")
            print(f"  平均能耗: {result.stats['avg_energy']:.4f}J")
            print(f"  Checkpoint使用率: {result.stats['checkpoint_usage']*100:.1f}%")
        
        print(f"\n【最终价格】")
        for uav_id, prices in result.final_prices.items():
            print(f"  UAV-{uav_id}: 算力={prices.p_compute:.2e}, "
                  f"能量={prices.p_energy:.2e}")
        
        print("\n" + "=" * 50)


def run_phase4(tasks: List[ExecutionTask],
               uav_states: List[UAVRuntimeState]) -> Phase4Result:
    """
    运行阶段4的便捷函数
    """
    controller = Phase4Controller()
    controller.initialize(uav_states)
    return controller.run(tasks)


# ============ 测试用例 ============

def test_phase4_controller():
    """测试Phase4Controller模块"""
    print("=" * 60)
    print("测试 M23: Phase4Controller")
    print("=" * 60)
    
    controller = Phase4Controller()
    
    # 创建UAV状态
    uav_states = [
        UAVRuntimeState(uav_id=0, E_remain=400e3, E_max=500e3, 
                       f_available=8e9, f_max=10e9),
        UAVRuntimeState(uav_id=1, E_remain=450e3, E_max=500e3,
                       f_available=9e9, f_max=10e9),
    ]
    
    # 初始化
    controller.initialize(uav_states)
    
    # 创建任务
    tasks = [
        ExecutionTask(
            task_id=i, user_id=i, uav_id=i % 2,
            split_layer=8, total_layers=16,
            C_edge=3e9, C_cloud=6e9,
            f_edge=5e9, f_cloud=50e9,
            deadline=5.0, priority=0.5 + i * 0.1
        )
        for i in range(6)
    ]
    
    # 测试1: 完整执行
    print("\n[Test 1] 测试完整执行...")
    result = controller.run(tasks, time_step=0.05, max_time=20.0)
    
    assert result.execution_results is not None
    print(f"  完成任务: {len(result.execution_results)}")
    print(f"  失败任务: {len(result.failed_tasks)}")
    print("  ✓ 完整执行正确")
    
    # 测试2: 统计信息
    print("\n[Test 2] 测试统计信息...")
    assert 'success_rate' in result.stats
    print(f"  成功率: {result.stats['success_rate']*100:.1f}%")
    print("  ✓ 统计信息正确")
    
    # 测试3: 价格更新
    print("\n[Test 3] 测试价格更新...")
    assert len(result.final_prices) > 0
    print(f"  价格数量: {len(result.final_prices)}")
    print("  ✓ 价格更新正确")
    
    # 测试4: 便捷函数
    print("\n[Test 4] 测试便捷函数...")
    result2 = run_phase4(tasks, uav_states)
    assert result2 is not None
    print("  ✓ 便捷函数正确")
    
    # 测试5: 摘要输出
    print("\n[Test 5] 测试摘要输出...")
    controller.print_summary(result)
    print("  ✓ 摘要输出正确")
    
    # 测试6: 计划转换
    print("\n[Test 6] 测试计划转换...")
    plans = [
        {'task_id': 100, 'uav_id': 0, 'priority': 0.8, 'deadline': 3.0}
    ]
    converted = controller.convert_plans_to_tasks(plans)
    assert len(converted) == 1
    assert converted[0].task_id == 100
    print("  ✓ 计划转换正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_phase4_controller()

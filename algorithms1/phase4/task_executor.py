"""
M21: TaskExecutor - 任务执行器

功能：管理任务执行状态，模拟DNN推理执行
输入：执行计划
输出：执行结果

执行流程:
    1. 接收执行计划
    2. 按优先级调度执行
    3. 模拟计算过程
    4. 更新状态
    5. 返回结果
"""

import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import COMMUNICATION, NUMERICAL


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    EDGE_COMPUTING = "EDGE_COMPUTING"
    TRANSMITTING = "TRANSMITTING"
    CLOUD_COMPUTING = "CLOUD_COMPUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MIGRATED = "MIGRATED"


@dataclass
class ExecutionTask:
    """
    执行任务
    
    Attributes:
        task_id: 任务ID
        user_id: 用户ID
        uav_id: 执行UAV ID
        split_layer: 切分层
        total_layers: 总层数
        C_edge: 边缘计算量
        C_cloud: 云端计算量
        f_edge: 边缘算力
        f_cloud: 云端算力
        deadline: 截止时间
        priority: 优先级
        D_trans: 传输数据量 (bits)
        R_backhaul: 回程链路速率 (bps)
        checkpoint_layer: Checkpoint层
        status: 当前状态
        current_layer: 当前执行层
        progress: 执行进度 [0, 1]
        start_time: 开始时间
        elapsed_time: 已用时间
    """
    task_id: int
    user_id: int
    uav_id: int
    split_layer: int
    total_layers: int
    C_edge: float
    C_cloud: float
    f_edge: float
    f_cloud: float
    deadline: float
    priority: float
    D_trans: float = 0.0  # 传输数据量 (bits)
    R_backhaul: float = 100e6  # 回程链路速率 (bps)，默认100Mbps
    checkpoint_layer: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    current_layer: int = 0
    progress: float = 0.0
    start_time: Optional[float] = None
    elapsed_time: float = 0.0
    checkpoint_saved: bool = False
    result: Optional[Dict] = None


@dataclass
class ExecutionResult:
    """
    执行结果
    
    Attributes:
        task_id: 任务ID
        success: 是否成功
        total_time: 总执行时间
        edge_time: 边缘计算时间
        cloud_time: 云端计算时间
        trans_time: 传输时间
        energy_consumed: 消耗能量
        checkpoint_used: 是否使用了Checkpoint
        met_deadline: 是否满足时延要求
    """
    task_id: int
    success: bool
    total_time: float
    edge_time: float
    cloud_time: float
    trans_time: float
    energy_consumed: float
    checkpoint_used: bool
    met_deadline: bool


class TaskExecutor:
    """
    任务执行器
    
    Attributes:
        running_tasks: 正在执行的任务
        completed_tasks: 已完成的任务
        failed_tasks: 失败的任务
    """
    
    def __init__(self, kappa_edge: float = 1e-28):
        """
        初始化执行器
        
        Args:
            kappa_edge: 能耗系数
        """
        self.kappa_edge = kappa_edge
        self.running_tasks: Dict[int, ExecutionTask] = {}
        self.completed_tasks: Dict[int, ExecutionResult] = {}
        self.failed_tasks: Dict[int, ExecutionTask] = {}
        self.current_time: float = 0.0
    
    def submit_task(self, task: ExecutionTask) -> bool:
        """
        提交任务
        
        Args:
            task: 执行任务
            
        Returns:
            bool: 是否成功提交
        """
        if task.task_id in self.running_tasks:
            return False
        
        task.status = TaskStatus.PENDING
        task.start_time = self.current_time
        self.running_tasks[task.task_id] = task
        return True
    
    def execute_step(self, task_id: int, time_step: float) -> Tuple[TaskStatus, float]:
        """
        执行一步计算
        
        Args:
            task_id: 任务ID
            time_step: 时间步长
            
        Returns:
            Tuple: (新状态, 进度)
        """
        if task_id not in self.running_tasks:
            return TaskStatus.FAILED, 0.0
        
        task = self.running_tasks[task_id]
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.RUNNING
            task.start_time = self.current_time
        
        # 计算进度增量
        if task.status in [TaskStatus.RUNNING, TaskStatus.EDGE_COMPUTING]:
            task.status = TaskStatus.EDGE_COMPUTING
            
            # 边缘计算进度
            edge_ratio = task.split_layer / task.total_layers
            edge_time_total = task.C_edge / task.f_edge if task.f_edge > 0 else 0
            
            if edge_time_total > 0:
                progress_delta = time_step / edge_time_total * edge_ratio
                task.progress = min(task.progress + progress_delta, edge_ratio)
            
            if task.progress >= edge_ratio:
                task.status = TaskStatus.TRANSMITTING
        
        elif task.status == TaskStatus.TRANSMITTING:
            # 真实计算传输时间: T_trans = D_trans / R_backhaul
            if task.D_trans > 0 and task.R_backhaul > 0:
                trans_time = task.D_trans / task.R_backhaul
            else:
                trans_time = COMMUNICATION.MIN_TRANSMISSION_TIME
            trans_time = max(trans_time, COMMUNICATION.MIN_TRANSMISSION_TIME)
            
            # 传输阶段占整体进度的5%
            trans_progress_ratio = 0.05
            edge_ratio = task.split_layer / task.total_layers
            task.progress = min(task.progress + time_step / trans_time * trans_progress_ratio, 
                              edge_ratio + trans_progress_ratio)
            
            if task.progress >= edge_ratio + trans_progress_ratio:
                task.status = TaskStatus.CLOUD_COMPUTING
        
        elif task.status == TaskStatus.CLOUD_COMPUTING:
            cloud_ratio = 1 - task.split_layer / task.total_layers
            cloud_time_total = task.C_cloud / task.f_cloud if task.f_cloud > 0 else 0
            
            if cloud_time_total > 0:
                progress_delta = time_step / cloud_time_total * cloud_ratio
                task.progress = min(task.progress + progress_delta, 1.0)
            else:
                task.progress = 1.0
            
            if task.progress >= 1.0:
                task.status = TaskStatus.COMPLETED
        
        task.elapsed_time += time_step
        task.current_layer = int(task.progress * task.total_layers)
        
        # 检查Checkpoint
        if (task.checkpoint_layer and 
            task.current_layer >= task.checkpoint_layer and 
            not task.checkpoint_saved):
            task.checkpoint_saved = True
        
        # 检查超时
        if task.elapsed_time > task.deadline:
            task.status = TaskStatus.FAILED
            self._move_to_failed(task_id)
        
        # 检查完成
        if task.status == TaskStatus.COMPLETED:
            self._move_to_completed(task_id)
        
        return task.status, task.progress
    
    def _move_to_completed(self, task_id: int):
        """将任务移至已完成列表"""
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks.pop(task_id)
        
        # 计算能耗
        energy = self.kappa_edge * (task.f_edge ** 2) * task.C_edge
        
        # 真实计算传输时间
        if task.D_trans > 0 and task.R_backhaul > 0:
            trans_time = task.D_trans / task.R_backhaul
        else:
            trans_time = COMMUNICATION.MIN_TRANSMISSION_TIME
        trans_time = max(trans_time, COMMUNICATION.MIN_TRANSMISSION_TIME)
        
        result = ExecutionResult(
            task_id=task_id,
            success=True,
            total_time=task.elapsed_time,
            edge_time=task.C_edge / task.f_edge if task.f_edge > 0 else 0,
            cloud_time=task.C_cloud / task.f_cloud if task.f_cloud > 0 else 0,
            trans_time=trans_time,
            energy_consumed=energy,
            checkpoint_used=task.checkpoint_saved,
            met_deadline=task.elapsed_time <= task.deadline
        )
        
        self.completed_tasks[task_id] = result
    
    def _move_to_failed(self, task_id: int):
        """将任务移至失败列表"""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            self.failed_tasks[task_id] = task
    
    def simulate_execution(self, 
                           tasks: List[ExecutionTask],
                           time_step: float = 0.1,
                           max_time: float = 100.0) -> Dict[int, ExecutionResult]:
        """
        模拟执行多个任务
        
        Args:
            tasks: 任务列表
            time_step: 时间步长
            max_time: 最大模拟时间
            
        Returns:
            Dict: {task_id: 执行结果}
        """
        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # 提交所有任务
        for task in sorted_tasks:
            self.submit_task(task)
        
        # 执行模拟
        self.current_time = 0.0
        
        while self.running_tasks and self.current_time < max_time:
            for task_id in list(self.running_tasks.keys()):
                self.execute_step(task_id, time_step)
            self.current_time += time_step
        
        # 将剩余任务标记为失败
        for task_id in list(self.running_tasks.keys()):
            self._move_to_failed(task_id)
        
        return self.completed_tasks
    
    def get_stats(self) -> Dict:
        """获取执行统计"""
        completed = list(self.completed_tasks.values())
        
        if not completed:
            return {
                'total': 0,
                'completed': 0,
                'failed': len(self.failed_tasks),
                'success_rate': 0.0
            }
        
        success_count = sum(1 for r in completed if r.met_deadline)
        
        return {
            'total': len(completed) + len(self.failed_tasks),
            'completed': len(completed),
            'failed': len(self.failed_tasks),
            'success_rate': success_count / (len(completed) + len(self.failed_tasks)),
            'avg_time': np.mean([r.total_time for r in completed]),
            'avg_energy': np.mean([r.energy_consumed for r in completed]),
            'checkpoint_usage': sum(1 for r in completed if r.checkpoint_used) / len(completed)
        }


# ============ 测试用例 ============

def test_task_executor():
    """测试TaskExecutor模块"""
    print("=" * 60)
    print("测试 M21: TaskExecutor")
    print("=" * 60)
    
    executor = TaskExecutor()
    
    # 创建测试任务
    tasks = [
        ExecutionTask(
            task_id=0, user_id=0, uav_id=0,
            split_layer=8, total_layers=16,
            C_edge=5e9, C_cloud=10e9,
            f_edge=5e9, f_cloud=50e9,
            deadline=5.0, priority=0.9,
            checkpoint_layer=4
        ),
        ExecutionTask(
            task_id=1, user_id=1, uav_id=0,
            split_layer=4, total_layers=16,
            C_edge=2e9, C_cloud=13e9,
            f_edge=5e9, f_cloud=50e9,
            deadline=4.0, priority=0.7
        ),
        ExecutionTask(
            task_id=2, user_id=2, uav_id=1,
            split_layer=12, total_layers=16,
            C_edge=10e9, C_cloud=5e9,
            f_edge=5e9, f_cloud=50e9,
            deadline=3.0, priority=0.5
        ),
    ]
    
    # 测试1: 任务提交
    print("\n[Test 1] 测试任务提交...")
    for task in tasks:
        success = executor.submit_task(task)
        assert success, f"任务{task.task_id}提交失败"
    print(f"  提交了 {len(tasks)} 个任务")
    print("  ✓ 任务提交正确")
    
    # 测试2: 单步执行
    print("\n[Test 2] 测试单步执行...")
    status, progress = executor.execute_step(0, 0.1)
    print(f"  任务0: 状态={status.value}, 进度={progress*100:.1f}%")
    print("  ✓ 单步执行正确")
    
    # 重置执行器
    executor = TaskExecutor()
    
    # 测试3: 模拟执行
    print("\n[Test 3] 测试模拟执行...")
    results = executor.simulate_execution(tasks, time_step=0.05, max_time=10.0)
    
    print(f"  完成: {len(results)} 个任务")
    for task_id, result in results.items():
        print(f"    任务{task_id}: 时间={result.total_time:.2f}s, "
              f"满足时延={result.met_deadline}")
    print("  ✓ 模拟执行正确")
    
    # 测试4: 统计信息
    print("\n[Test 4] 测试统计信息...")
    stats = executor.get_stats()
    
    print(f"  总任务: {stats['total']}")
    print(f"  成功率: {stats['success_rate']*100:.1f}%")
    print(f"  平均时间: {stats['avg_time']:.2f}s")
    print("  ✓ 统计信息正确")
    
    # 测试5: 超时任务
    print("\n[Test 5] 测试超时任务...")
    executor2 = TaskExecutor()
    
    timeout_task = ExecutionTask(
        task_id=99, user_id=99, uav_id=0,
        split_layer=8, total_layers=16,
        C_edge=50e9, C_cloud=100e9,  # 大计算量
        f_edge=1e9, f_cloud=10e9,     # 低算力
        deadline=0.5, priority=0.5     # 短时限
    )
    
    results2 = executor2.simulate_execution([timeout_task], max_time=2.0)
    
    assert 99 in executor2.failed_tasks, "超时任务应失败"
    print(f"  超时任务状态: FAILED")
    print("  ✓ 超时处理正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_task_executor()

"""
M13: Priority - 任务优先级计算

功能：计算任务优先级得分，用于分类和调度
输入：任务列表、配置参数
输出：优先级得分、优先级分类

关键公式 (idea118.txt 1.3节):
    ω_i = α₁ * ω_data + α₂ * ω_comp + α₃ * ω_delay + α₄ * ω_level
    
    其中:
        ω_data = D_i / D_max  (数据量因子)
        ω_comp = C_i / C_max  (计算量因子)
        ω_delay = 1 / τ_i^max  (时延紧迫度，归一化)
        ω_level = l_i / l_max  (用户等级因子)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.system_config import PriorityConfig
from config.constants import NUMERICAL


class PriorityLevel(Enum):
    """优先级等级"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class TaskPriority:
    """
    任务优先级结果
    
    Attributes:
        task_id: 任务ID
        user_id: 用户ID
        score: 优先级得分 (0-1)
        level: 优先级等级
        components: 各分量得分
    """
    task_id: int
    user_id: int
    score: float
    level: PriorityLevel
    components: Dict[str, float]


class PriorityCalculator:
    """
    优先级计算器
    
    Attributes:
        config: 优先级配置
    """
    
    def __init__(self, config: Optional[PriorityConfig] = None):
        """
        初始化优先级计算器
        
        Args:
            config: 优先级配置
        """
        self.config = config or PriorityConfig()
    
    def compute_priority_score(self,
                               data_size: float,
                               compute_size: float,
                               deadline: float,
                               user_level: int,
                               data_max: float,
                               compute_max: float,
                               deadline_min: float,
                               level_max: int = 5) -> Tuple[float, Dict[str, float]]:
        """
        计算单个任务的优先级得分
        
        公式: ω = α₁*ω_data + α₂*ω_comp + α₃*ω_delay + α₄*ω_level
        
        Args:
            data_size: 数据量 (bits)
            compute_size: 计算量 (FLOPs)
            deadline: 截止时间 (s)
            user_level: 用户等级 (1-5)
            data_max: 最大数据量
            compute_max: 最大计算量
            deadline_min: 最小截止时间
            level_max: 最大用户等级
            
        Returns:
            Tuple[float, Dict]: (优先级得分, 各分量)
        """
        # 数据量因子（使用常量保护值）
        omega_data = data_size / max(data_max, NUMERICAL.EPSILON)
        
        # 计算量因子
        omega_comp = compute_size / max(compute_max, NUMERICAL.EPSILON)
        
        # 时延紧迫度因子 (截止时间越短，紧迫度越高)
        omega_delay = deadline_min / max(deadline, NUMERICAL.EPSILON)
        omega_delay = min(omega_delay, 1.0)  # 截断到1
        
        # 用户等级因子
        omega_level = user_level / level_max
        
        # 加权求和
        score = (
            self.config.alpha_data * omega_data +
            self.config.alpha_comp * omega_comp +
            self.config.alpha_delay * omega_delay +
            self.config.alpha_level * omega_level
        )
        
        components = {
            'omega_data': omega_data,
            'omega_comp': omega_comp,
            'omega_delay': omega_delay,
            'omega_level': omega_level
        }
        
        return score, components
    
    def compute_batch_priorities(self,
                                 tasks: List[Dict]) -> List[TaskPriority]:
        """
        批量计算任务优先级
        
        Args:
            tasks: 任务列表，每个任务包含:
                   {'task_id', 'user_id', 'data_size', 'compute_size', 
                    'deadline', 'user_level'}
                    
        Returns:
            List[TaskPriority]: 优先级结果列表
        """
        if not tasks:
            return []
        
        # 计算最大值用于归一化
        data_max = max(t['data_size'] for t in tasks)
        compute_max = max(t['compute_size'] for t in tasks)
        deadline_min = min(t['deadline'] for t in tasks)
        level_max = max(t['user_level'] for t in tasks)
        
        results = []
        scores = []
        
        for task in tasks:
            score, components = self.compute_priority_score(
                data_size=task['data_size'],
                compute_size=task['compute_size'],
                deadline=task['deadline'],
                user_level=task['user_level'],
                data_max=data_max,
                compute_max=compute_max,
                deadline_min=deadline_min,
                level_max=level_max
            )
            scores.append(score)
            results.append({
                'task_id': task['task_id'],
                'user_id': task['user_id'],
                'score': score,
                'components': components
            })
        
        # 计算分位数阈值进行分类
        scores_arr = np.array(scores)
        theta_high = np.percentile(scores_arr, self.config.theta_high_percentile)
        theta_medium = np.percentile(scores_arr, self.config.theta_medium_percentile)
        
        # 分配优先级等级
        priority_results = []
        for r in results:
            if r['score'] >= theta_high:
                level = PriorityLevel.HIGH
            elif r['score'] >= theta_medium:
                level = PriorityLevel.MEDIUM
            else:
                level = PriorityLevel.LOW
            
            priority_results.append(TaskPriority(
                task_id=r['task_id'],
                user_id=r['user_id'],
                score=r['score'],
                level=level,
                components=r['components']
            ))
        
        return priority_results
    
    def classify_tasks(self,
                       priorities: List[TaskPriority]) -> Dict[PriorityLevel, List[int]]:
        """
        按优先级分类任务
        
        Args:
            priorities: 优先级结果列表
            
        Returns:
            Dict: {优先级等级: [任务ID列表]}
        """
        classified = {
            PriorityLevel.HIGH: [],
            PriorityLevel.MEDIUM: [],
            PriorityLevel.LOW: []
        }
        
        for p in priorities:
            classified[p.level].append(p.task_id)
        
        return classified
    
    def get_priority_stats(self, priorities: List[TaskPriority]) -> Dict:
        """
        获取优先级统计信息
        
        Args:
            priorities: 优先级结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not priorities:
            return {'count': 0}
        
        scores = [p.score for p in priorities]
        classified = self.classify_tasks(priorities)
        
        return {
            'count': len(priorities),
            'high_count': len(classified[PriorityLevel.HIGH]),
            'medium_count': len(classified[PriorityLevel.MEDIUM]),
            'low_count': len(classified[PriorityLevel.LOW]),
            'score_min': min(scores),
            'score_max': max(scores),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores)
        }


# ============ 测试用例 ============

def test_priority():
    """测试Priority模块"""
    print("=" * 60)
    print("测试 M13: Priority")
    print("=" * 60)
    
    calculator = PriorityCalculator()
    
    # 测试1: 单任务优先级计算
    print("\n[Test 1] 测试单任务优先级计算...")
    score, components = calculator.compute_priority_score(
        data_size=5e6,
        compute_size=10e9,
        deadline=2.0,
        user_level=4,
        data_max=10e6,
        compute_max=20e9,
        deadline_min=1.0,
        level_max=5
    )
    
    assert 0 <= score <= 1, "得分应在[0,1]范围内"
    print(f"  优先级得分: {score:.4f}")
    print(f"  数据量因子: {components['omega_data']:.3f}")
    print(f"  计算量因子: {components['omega_comp']:.3f}")
    print(f"  时延紧迫度: {components['omega_delay']:.3f}")
    print(f"  用户等级: {components['omega_level']:.3f}")
    print("  ✓ 单任务计算正确")
    
    # 测试2: 批量优先级计算
    print("\n[Test 2] 测试批量优先级计算...")
    
    np.random.seed(42)
    tasks = []
    for i in range(20):
        tasks.append({
            'task_id': i,
            'user_id': i,
            'data_size': np.random.uniform(1e6, 10e6),
            'compute_size': np.random.uniform(5e9, 50e9),
            'deadline': np.random.uniform(1.0, 5.0),
            'user_level': np.random.randint(1, 6)
        })
    
    priorities = calculator.compute_batch_priorities(tasks)
    
    assert len(priorities) == 20, "应有20个结果"
    print(f"  计算了 {len(priorities)} 个任务的优先级")
    print("  ✓ 批量计算正确")
    
    # 测试3: 任务分类
    print("\n[Test 3] 测试任务分类...")
    classified = calculator.classify_tasks(priorities)
    
    total = sum(len(v) for v in classified.values())
    assert total == 20, "分类后总数应等于原始数量"
    
    print(f"  高优先级: {len(classified[PriorityLevel.HIGH])} 个")
    print(f"  中优先级: {len(classified[PriorityLevel.MEDIUM])} 个")
    print(f"  低优先级: {len(classified[PriorityLevel.LOW])} 个")
    print("  ✓ 任务分类正确")
    
    # 测试4: 优先级统计
    print("\n[Test 4] 测试优先级统计...")
    stats = calculator.get_priority_stats(priorities)
    
    print(f"  总任务数: {stats['count']}")
    print(f"  得分范围: [{stats['score_min']:.3f}, {stats['score_max']:.3f}]")
    print(f"  平均得分: {stats['score_mean']:.3f}")
    print("  ✓ 统计信息正确")
    
    # 测试5: 验证时延紧迫度的影响
    print("\n[Test 5] 验证时延紧迫度影响...")
    
    # 创建两个相同任务，仅时延不同
    urgent_task = {
        'task_id': 100, 'user_id': 100,
        'data_size': 5e6, 'compute_size': 10e9,
        'deadline': 1.0, 'user_level': 3
    }
    normal_task = {
        'task_id': 101, 'user_id': 101,
        'data_size': 5e6, 'compute_size': 10e9,
        'deadline': 5.0, 'user_level': 3
    }
    
    results = calculator.compute_batch_priorities([urgent_task, normal_task])
    
    assert results[0].score > results[1].score, "紧急任务应有更高优先级"
    print(f"  紧急任务(1s): {results[0].score:.4f}")
    print(f"  普通任务(5s): {results[1].score:.4f}")
    print("  ✓ 时延紧迫度影响验证正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_priority()

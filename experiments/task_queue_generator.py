"""
基于泊松分布的任务队列生成器

用于在线算法的任务队列实现：
- 任务按照泊松过程动态到达
- 不预先知道所有任务（符合在线算法假设）
- 支持到达速率控制和总任务数控制
- 返回适合任务生成器使用的格式
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import task type configs
from experiments.task_types import (
    LATENCY_SENSITIVE_CONFIG, COMPUTE_INTENSIVE_CONFIG,
    MOBILENETV2_SPEC, VGG16_SPEC
)


@dataclass
class TaskQueueConfig:
    """任务队列配置"""
    # 泊松过程参数
    arrival_rate: float = 0.1      # 任务到达速率 (任务/秒)
    mean_inter_arrival: float = 10.0  # 平均到达间隔 (秒)
    simulation_time: float = 100.0   # 仿真总时长 (秒)

    # 任务生成参数
    task_generator: Optional['MNISTTaskGenerator'] = None
    n_users: int = 50                # 目标用户数
    seed: int = 42

    # 到达模式控制
    enable_burst_arrival: bool = False  # 是否启用突发到达
    burst_probability: float = 0.05      # 突发到达概率
    burst_size: int = 5              # 突发任务数


@dataclass
class ArrivedTask:
    """到达的任务"""
    task_id: int
    user_id: int
    task_type: str
    n_images: int
    data_size_bytes: float
    total_flops: float
    deadline: float
    priority: float
    user_pos_x: float
    user_pos_y: float
    arrival_time: float              # 任务到达时间戳
    arrival_interval: float           # 距上一个任务的到达间隔


class TaskQueueGenerator:
    """
    基于泊松分布的任务队列生成器

    功能：
    1. 生成任务按照泊松过程到达
    2. 适应不同实验的任务数和场景配置
    3. 返回适合任务生成器的格式
    """

    def __init__(self, config: TaskQueueConfig = None):
        """
        初始化任务队列生成器

        Args:
            config: 任务队列配置
        """
        self.config = config if config else TaskQueueConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # 初始化任务生成器
        if self.config.task_generator is None:
            # 默认使用MNISTTaskGenerator
            from experiments.task_types import MNISTTaskGenerator
            self.config.task_generator = MNISTTaskGenerator(
                area_size=200.0,
                latency_ratio=0.5,
                tasks_per_user=5,
                seed=self.config.seed
            )

        # 生成目标任务总数（基于到达速率和仿真时长）
        self._calculate_expected_tasks()

        # 任务状态跟踪
        self.generated_tasks: List[ArrivedTask] = []
        self.current_task_id = 0
        self.current_user_id = 0
        self.current_arrival_time = 0.0

    def _calculate_expected_tasks(self):
        """计算预期任务总数"""
        if self.config.enable_burst_arrival:
            # 突发模式：预期任务数 = 基础任务数 + 突发概率 * 突发大小
            base_tasks = int(self.config.n_users * self.config.task_generator.tasks_per_user)
            expected = base_tasks + int(base_tasks * self.config.burst_probability * self.config.burst_size)
        else:
            # 泊松过程：预期任务数 = 到达速率 × 仿真时长
            expected = int(self.config.arrival_rate * self.config.simulation_time)

        self.expected_total_tasks = expected

    def generate_poisson_arrivals(self, n_users: int,
                                  seed: int = None) -> List[float]:
        """
        生成泊松到达时间序列

        Args:
            n_users: 用户数量
            seed: 随机种子

        Returns:
            到达间隔列表 (秒)
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        # 为每个用户生成到达间隔
        arrival_intervals = []
        total_arrivals = 0

        for user_id in range(n_users):
            # 泊松过程：到达间隔服从指数分布
            # P(arrival_interval <= t) = 1 - exp(-lambda * t)
            # 其中 lambda = arrival_rate（任务/秒）

            while total_arrivals < self.config.n_users * self.config.task_generator.tasks_per_user:
                # 生成泊松到达间隔
                interval = rng.exponential(scale=1.0 / self.config.arrival_rate)
                arrival_intervals.append(interval)
                total_arrivals += 1

        return arrival_intervals

    def generate_task_queue(self, n_users: int = None) -> Iterator[ArrivedTask]:
        """
        生成任务队列（迭代器模式）

        Args:
            n_users: 用户数量（如果为None则使用配置中的值）

        Returns:
            任务队列迭代器（按到达时间排序）
        """
        if n_users is None:
            n_users = self.config.n_users

        # 生成到达间隔
        arrival_intervals = self.generate_poisson_arrivals(n_users, self.config.seed)

        # 为每个用户分配任务生成参数
        user_configs = self._allocate_user_configs(n_users)

        # 生成所有任务并记录到达时间
        all_tasks = []
        arrival_time = 0.0

        for user_id, user_config in enumerate(user_configs):
            for _ in range(self.config.task_generator.tasks_per_user):
                # 生成任务（带到达时间）
                task = self._generate_single_task(
                    user_id=user_id,
                    task_num=_,
                    user_config=user_config,
                    arrival_time=arrival_time
                )
                all_tasks.append(task)
                self.current_task_id += 1
                arrival_time += task.arrival_interval

        # 按到达时间排序（模拟在线算法看到的任务顺序）
        all_tasks.sort(key=lambda t: t.arrival_time)

        return iter(all_tasks)

    def _allocate_user_configs(self, n_users: int) -> List[Dict]:
        """
        为每个用户分配任务配置

        确保每个用户分配公平的任务数量和类型
        """
        user_configs = []

        # 计算总任务数
        total_tasks = n_users * self.config.task_generator.tasks_per_user

        # 分配任务类型（循环分配延迟敏感型和计算密集型）
        for user_id in range(n_users):
            if user_id % 2 == 0:
                # 延迟敏感型任务（使用MobileNetV2）
                task_config = {
                    'task_type': 'latency_sensitive',
                    'model_spec': MOBILENETV2_SPEC,
                    'n_images_range': (LATENCY_SENSITIVE_CONFIG.min_images,
                                      LATENCY_SENSITIVE_CONFIG.max_images),
                    'deadline_range': (LATENCY_SENSITIVE_CONFIG.min_deadline,
                                      LATENCY_SENSITIVE_CONFIG.max_deadline),
                }
            else:
                # 计算密集型任务（使用VGG16）
                task_config = {
                    'task_type': 'compute_intensive',
                    'model_spec': VGG16_SPEC,
                    'n_images_range': (COMPUTE_INTENSIVE_CONFIG.min_images,
                                      COMPUTE_INTENSIVE_CONFIG.max_images),
                    'deadline_range': (COMPUTE_INTENSIVE_CONFIG.min_deadline,
                                      COMPUTE_INTENSIVE_CONFIG.max_deadline),
                }

            user_configs.append(task_config)

        return user_configs

    def _generate_single_task(self, user_id: int, task_num: int,
                               user_config: Dict, arrival_time: float) -> ArrivedTask:
        """
        生成单个任务

        Args:
            user_id: 用户ID
            task_num: 用户内任务编号
            user_config: 用户配置
            arrival_time: 到达时间

        Returns:
            ArrivedTask对象
        """
        task_type = user_config['task_type']
        model_spec = user_config['model_spec']

        # 根据配置随机生成任务参数（使用TaskQueueGenerator的rng）
        n_images = self.rng.integers(
            user_config['n_images_range'][0],
            user_config['n_images_range'][1]
        )
        deadline = self.rng.uniform(
            user_config['deadline_range'][0],
            user_config['deadline_range'][1]
        )
        priority = self.rng.uniform(0.3, 0.9)

        # 生成用户位置
        area_size = self.config.task_generator.area_size
        user_pos_x = self.rng.uniform(50, area_size - 50)
        user_pos_y = self.rng.uniform(50, area_size - 50)

        # 计算数据大小和FLOPs
        from experiments.mnist_loader import compute_input_data_size
        data_size = compute_input_data_size(n_images)
        total_flops = model_spec.get_flops_for_images(n_images)

        # 计算到达间隔（使用泊松过程）
        if task_num < self.config.task_generator.tasks_per_user:
            interval = self.rng.exponential(
                scale=1.0 / self.config.arrival_rate
            )
        else:
            interval = float('inf')  # 用户最后一个任务后的间隔

        return ArrivedTask(
            task_id=self.current_task_id,
            user_id=user_id,
            task_type=task_type,
            n_images=n_images,
            data_size_bytes=data_size['bytes'],
            total_flops=total_flops,
            deadline=deadline,
            priority=priority,
            user_pos_x=user_pos_x,
            user_pos_y=user_pos_y,
            arrival_time=arrival_time,
            arrival_interval=interval
        )

    def get_task_dict(self, task: ArrivedTask) -> Dict:
        """
        将ArrivedTask转换为任务字典（与任务生成器兼容）

        Args:
            task: ArrivedTask对象

        Returns:
            任务字典
        """
        return {
            'task_id': task.task_id,
            'user_id': task.user_id,
            'task_type': task.task_type,
            'model_name': task.task_type,
            'n_images': task.n_images,
            'data_size': task.data_size_bytes,
            'total_flops': task.total_flops,
            'deadline': task.deadline,
            'priority': task.priority,
            'user_pos': (task.user_pos_x, task.user_pos_y),
            'user_x': task.user_pos_x,
            'user_y': task.user_pos_y,
            'arrival_time': task.arrival_time,
            'arrival_interval': task.arrival_interval,
            'compute_size': task.total_flops,
            'C_total': task.total_flops
        }

    def convert_to_batch(self, task_queue: Iterator[ArrivedTask],
                      n_batch: int = 10) -> List[List[Dict]]:
        """
        将任务队列转换为批次列表

        Args:
            task_queue: 任务队列迭代器
            n_batch: 每批任务数

        Returns:
            任务批次列表
        """
        batches = []
        current_batch = []

        for task in task_queue:
            current_batch.append(task)

            if len(current_batch) == n_batch:
                batches.append([self.get_task_dict(t) for t in current_batch])
                current_batch = []

        # 处理剩余任务
        if current_batch:
            batches.append([self.get_task_dict(t) for t in current_batch])

        return batches

    def generate_fixed_tasks(self, n_users: int = None,
                        tasks_per_user: int = None) -> List[Dict]:
        """
        生成固定任务集（非泊松到达模式）
        用于兼容现有实验配置的静态任务生成

        Args:
            n_users: 用户数量
            tasks_per_user: 每用户任务数
        """
        if n_users is None:
            n_users = self.config.n_users
        if tasks_per_user is None:
            tasks_per_user = self.config.task_generator.tasks_per_user

        # 使用任务生成器生成所有任务
        self.current_task_id = 0
        all_tasks = []

        for user_id in range(n_users):
            for task_num in range(tasks_per_user):
                task_config = {
                    'task_type': 'latency_sensitive' if user_id % 2 == 0 else 'compute_intensive',
                    'model_spec': self.config.task_generator.model_spec,
                    'n_images_range': (5, 50) if user_id % 2 == 0 else (10, 100),
                    'deadline_range': (0.5, 1.5) if user_id % 2 == 0 else (1.0, 3.0),
                }

                task = self.config.task_generator.generate_single_task(
                    task_id=self.current_task_id,
                    user_id=user_id,
                    n_images=self.config.task_generator.rng.integers(
                        task_config['n_images_range'][0],
                        task_config['n_images_range'][1]
                    ),
                    deadline=self.config.task_generator.rng.uniform(
                        task_config['deadline_range'][0],
                        task_config['deadline_range'][1]
                    ),
                    priority=self.config.task_generator.rng.uniform(0.3, 0.9),
                    user_pos=(self.config.task_generator.rng.uniform(50, 200),
                               self.config.task_generator.rng.uniform(50, 200))
                )

                all_tasks.append(self.get_task_dict(task))
                self.current_task_id += 1

        return all_tasks


# ============ 便捷函数 ============

def create_task_queue_generator(arrival_rate: float = 0.1,
                                  n_users: int = 50,
                                  simulation_time: float = 100.0,
                                  tasks_per_user: int = 5,
                                  seed: int = 42) -> TaskQueueGenerator:
    """
    创建任务队列生成器（便捷函数）

    Args:
        arrival_rate: 任务到达速率 (任务/秒)
        n_users: 用户数量
        simulation_time: 仿真总时长 (秒)
        tasks_per_user: 每用户任务数
        seed: 随机种子

    Returns:
        TaskQueueGenerator实例
    """
    config = TaskQueueConfig(
        arrival_rate=arrival_rate,
        n_users=n_users,
        simulation_time=simulation_time,
        tasks_per_user=tasks_per_user,
        seed=seed
    )
    return TaskQueueGenerator(config)


def adapt_to_experiment_config(exp_config: Dict,
                             queue_generator: TaskQueueGenerator) -> Dict:
    """
    将任务队列生成器适配到实验配置

    Args:
        exp_config: 实验配置（来自scenario_config）
        queue_generator: 任务队列生成器

    Returns:
        适配后的配置字典
    """
    # 确定任务数
    if 'n_users' in exp_config:
        n_users = exp_config['n_users']
    elif 'variable_values' in exp_config and exp_config['variable_values']:
        n_users = exp_config['variable_values'][0]
    else:
        n_users = queue_generator.config.n_users

    # 计算总任务数
    total_tasks = n_users * queue_generator.config.task_generator.tasks_per_user

    # 根据任务数调整仿真时长
    if queue_generator.config.arrival_rate > 0:
        # 保证足够的时间生成所有任务
        simulation_time = (total_tasks / queue_generator.config.arrival_rate) * 1.5 + 10
    else:
        # 固定任务模式
        simulation_time = queue_generator.config.simulation_time

    return {
        'task_queue_enabled': True,
        'arrival_mode': 'poisson',
        'n_users': n_users,
        'total_tasks': total_tasks,
        'simulation_time': simulation_time,
        'tasks_per_user': queue_generator.config.task_generator.tasks_per_user,
        'arrival_rate': queue_generator.config.arrival_rate,
    }


# ============ 示例 ============

if __name__ == "__main__":
    print("任务队列生成器示例")
    print("=" * 50)

    # 创建任务队列生成器
    queue_gen = create_task_queue_generator(
        arrival_rate=0.05,  # 平均每5秒一个任务
        n_users=50,
        simulation_time=100.0,
        tasks_per_user=5,
        seed=42
    )

    # 生成任务队列
    task_queue = queue_gen.generate_task_queue()

    # 打印前10个任务的到达信息
    print(f"\n预期总任务数: {queue_gen.expected_total_tasks}")
    print(f"任务数/用户: {queue_gen.config.task_generator.tasks_per_user}")
    print(f"\n前20个任务到达信息:")
    print(f"{'TaskID':<6} {'User':<6} {'Type':<18} {'Images':<8} {'Deadline':<10} {'Priority':<10} {'ArrivalTime':<12} {'Interval':<10}")
    print("-" * 80)

    count = 0
    for task in task_queue:
        if count >= 20:
            break
        t = task
        print(f"{t.task_id:4d}  | {t.user_id:2d}  | {t.task_type:18s}  | {t.n_images:2d}  | {t.deadline:.2f}  | {t.priority:.2f}  | {t.arrival_time:.2f}  | {t.arrival_interval:.2f}")
        count += 1

    print("-" * 80)
    print(f"\n总实际任务数: {count}")

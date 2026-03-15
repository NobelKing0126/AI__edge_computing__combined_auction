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
    MOBILENETV2_SPEC, VGG16_SPEC,
    get_task_configs_for_scale
)


@dataclass
class TaskQueueConfig:
    """
    任务队列配置

    支持多种到达模式:
    - 'poisson': 泊松过程，任务随机到达
    - 'fixed': 固定模式，任务均匀分布
    - 'batch': 批次模式，任务按批次到达
    """
    # 基础参数
    n_users: int = 50                # 目标用户数
    tasks_per_user: int = 5          # 每用户任务数
    seed: int = 42

    # 泊松过程参数
    arrival_rate: float = 0.1        # 任务到达速率 (任务/秒)
    mean_inter_arrival: float = 10.0  # 平均到达间隔 (秒)
    simulation_time: float = 100.0   # 仿真总时长 (秒)

    # 到达模式
    arrival_mode: str = 'poisson'    # 'poisson' | 'fixed' | 'batch'

    # 场景参数
    area_size: float = 200.0         # 场景大小 (米)
    n_uavs: int = 5                  # UAV数量

    # 突发模式参数
    enable_burst_arrival: bool = False  # 是否启用突发到达
    burst_probability: float = 0.05      # 突发到达概率
    burst_size: int = 5              # 突发任务数

    # 任务生成器 (可选)
    task_generator: Optional['MNISTTaskGenerator'] = None


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
                area_size=self.config.area_size,
                latency_ratio=0.5,
                tasks_per_user=self.config.tasks_per_user,
                seed=self.config.seed
            )
        else:
            # 同步配置
            self.config.task_generator.tasks_per_user = self.config.tasks_per_user

        # 生成目标任务总数（基于到达速率和仿真时长）
        self._calculate_expected_tasks()

        # 任务状态跟踪
        self.generated_tasks: List[ArrivedTask] = []
        self.current_task_id = 0
        self.current_user_id = 0
        self.current_arrival_time = 0.0

    # ============ 动态调整方法（支持链式调用） ============

    def set_n_users(self, n_users: int) -> 'TaskQueueGenerator':
        """
        动态设置用户数

        Args:
            n_users: 用户数量

        Returns:
            self (支持链式调用)
        """
        self.config.n_users = n_users
        self._calculate_expected_tasks()
        return self

    def set_n_uavs(self, n_uavs: int) -> 'TaskQueueGenerator':
        """
        动态设置UAV数量

        Args:
            n_uavs: UAV数量

        Returns:
            self (支持链式调用)
        """
        self.config.n_uavs = n_uavs
        return self

    def set_arrival_rate(self, rate: float) -> 'TaskQueueGenerator':
        """
        动态设置到达速率

        Args:
            rate: 任务到达速率 (任务/秒)

        Returns:
            self (支持链式调用)
        """
        self.config.arrival_rate = rate
        self._calculate_expected_tasks()
        return self

    def set_tasks_per_user(self, tasks: int) -> 'TaskQueueGenerator':
        """
        动态设置每用户任务数

        Args:
            tasks: 每用户任务数

        Returns:
            self (支持链式调用)
        """
        self.config.tasks_per_user = tasks
        if self.config.task_generator is not None:
            self.config.task_generator.tasks_per_user = tasks
        self._calculate_expected_tasks()
        return self

    def set_arrival_mode(self, mode: str) -> 'TaskQueueGenerator':
        """
        动态设置到达模式

        Args:
            mode: 到达模式 ('poisson' | 'fixed' | 'batch')

        Returns:
            self (支持链式调用)
        """
        if mode not in ['poisson', 'fixed', 'batch']:
            raise ValueError(f"无效的到达模式: {mode}，必须是 'poisson', 'fixed' 或 'batch'")
        self.config.arrival_mode = mode
        return self

    def set_area_size(self, size: float) -> 'TaskQueueGenerator':
        """
        动态设置场景大小

        Args:
            size: 场景大小 (米)

        Returns:
            self (支持链式调用)
        """
        self.config.area_size = size
        if self.config.task_generator is not None:
            self.config.task_generator.area_size = size
        return self

    # ============ 实验适配方法 ============

    def configure_for_experiment(self, exp_id: int, **kwargs) -> 'TaskQueueGenerator':
        """
        根据实验ID自动配置参数

        实验配置对照表:
        | Exp | 名称               | 用户数 | UAV数 | 每用户任务数 | 到达模式 |
        |-----|-------------------|-------|------|------------|---------|
        | 1   | 小规模基线对比       | 200   | 15   | 5          | poisson |
        | 2   | 小规模用户扩展       | 可变  | 5    | 5          | fixed   |
        | 3   | 小规模UAV扩展       | 30    | 可变  | 5          | fixed   |
        | 4   | 大规模用户扩展       | 可变  | 15   | 10         | fixed   |
        | 5   | 大规模UAV扩展       | 150   | 可变  | 10         | fixed   |

        Args:
            exp_id: 实验编号 (1-5)
            **kwargs: 覆盖默认参数
                - n_users: 用户数量（用于Exp2, Exp4）
                - n_uavs: UAV数量（用于Exp3, Exp5）
                - arrival_rate: 到达速率
                - seed: 随机种子

        Returns:
            self (支持链式调用)
        """
        # 实验默认配置
        exp_configs = {
            1: {
                'n_users': 200,
                'n_uavs': 15,
                'tasks_per_user': 5,
                'arrival_mode': 'poisson',
                'arrival_rate': 0.1,
                'area_size': 200.0
            },
            2: {
                'n_users': kwargs.get('n_users', 30),
                'n_uavs': 5,
                'tasks_per_user': 5,
                'arrival_mode': 'fixed',
                'arrival_rate': 0.1,
                'area_size': 200.0
            },
            3: {
                'n_users': 30,
                'n_uavs': kwargs.get('n_uavs', 5),
                'tasks_per_user': 5,
                'arrival_mode': 'fixed',
                'arrival_rate': 0.1,
                'area_size': 200.0
            },
            4: {
                'n_users': kwargs.get('n_users', 100),
                'n_uavs': 15,
                'tasks_per_user': 10,
                'arrival_mode': 'fixed',
                'arrival_rate': 0.05,
                'area_size': 500.0
            },
            5: {
                'n_users': 150,
                'n_uavs': kwargs.get('n_uavs', 15),
                'tasks_per_user': 10,
                'arrival_mode': 'fixed',
                'arrival_rate': 0.05,
                'area_size': 500.0
            }
        }

        if exp_id not in exp_configs:
            raise ValueError(f"无效的实验ID: {exp_id}，必须是 1-5")

        # 应用配置
        config = exp_configs[exp_id]

        # 允许kwargs覆盖
        if 'seed' in kwargs:
            self.config.seed = kwargs['seed']
            self.rng = np.random.default_rng(self.config.seed)
        if 'arrival_rate' in kwargs:
            config['arrival_rate'] = kwargs['arrival_rate']

        # 更新配置
        self.config.n_users = config['n_users']
        self.config.n_uavs = config['n_uavs']
        self.config.tasks_per_user = config['tasks_per_user']
        self.config.arrival_mode = config['arrival_mode']
        self.config.arrival_rate = config['arrival_rate']
        self.config.area_size = config['area_size']

        # 同步到任务生成器
        if self.config.task_generator is not None:
            self.config.task_generator.tasks_per_user = self.config.tasks_per_user
            self.config.task_generator.area_size = self.config.area_size

        self._calculate_expected_tasks()
        return self

    def get_experiment_config(self) -> Dict:
        """
        获取当前实验配置

        Returns:
            配置字典
        """
        return {
            'n_users': self.config.n_users,
            'n_uavs': self.config.n_uavs,
            'tasks_per_user': self.config.tasks_per_user,
            'arrival_mode': self.config.arrival_mode,
            'arrival_rate': self.config.arrival_rate,
            'area_size': self.config.area_size,
            'seed': self.config.seed,
            'expected_total_tasks': self.expected_total_tasks
        }

    def _calculate_expected_tasks(self):
        """
        计算预期任务总数

        根据不同的到达模式计算:
        - poisson: 预期任务数 = 用户数 × 每用户任务数（到达时间随机）
        - fixed: 预期任务数 = 用户数 × 每用户任务数
        - batch: 预期任务数 = 用户数 × 每用户任务数
        """
        # 获取每用户任务数
        tasks_per_user = self.config.tasks_per_user
        if self.config.task_generator is not None:
            tasks_per_user = self.config.task_generator.tasks_per_user

        # 所有模式都基于用户数×每用户任务数计算
        base_tasks = int(self.config.n_users * tasks_per_user)

        if self.config.enable_burst_arrival:
            # 突发模式：预期任务数 = 基础任务数 + 突发概率 * 突发大小
            expected = base_tasks + int(base_tasks * self.config.burst_probability * self.config.burst_size)
        else:
            # 所有模式：预期任务数 = 用户数 × 每用户任务数
            expected = base_tasks

        self.expected_total_tasks = max(expected, 1)  # 至少1个任务

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

        # 获取规模特定的任务配置
        # 优先从task_generator获取is_small_scale属性
        is_small_scale = True  # 默认小规模
        if hasattr(self.config.task_generator, 'is_small_scale'):
            is_small_scale = self.config.task_generator.is_small_scale

        latency_config, compute_config = get_task_configs_for_scale(is_small_scale)

        # 计算总任务数
        total_tasks = n_users * self.config.task_generator.tasks_per_user

        # 分配任务类型（循环分配延迟敏感型和计算密集型）
        for user_id in range(n_users):
            if user_id % 2 == 0:
                # 延迟敏感型任务
                task_config = {
                    'task_type': 'latency_sensitive',
                    'model_spec': latency_config.model_spec,
                    'n_images_range': (latency_config.min_images,
                                      latency_config.max_images),
                    'deadline_range': (latency_config.min_deadline,
                                      latency_config.max_deadline),
                }
            else:
                # 计算密集型任务
                task_config = {
                    'task_type': 'compute_intensive',
                    'model_spec': compute_config.model_spec,
                    'n_images_range': (compute_config.min_images,
                                      compute_config.max_images),
                    'deadline_range': (compute_config.min_deadline,
                                      compute_config.max_deadline),
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
                                  seed: int = 42,
                                  arrival_mode: str = 'poisson',
                                  n_uavs: int = 5,
                                  area_size: float = 200.0) -> TaskQueueGenerator:
    """
    创建任务队列生成器（便捷函数）

    Args:
        arrival_rate: 任务到达速率 (任务/秒)
        n_users: 用户数量
        simulation_time: 仿真总时长 (秒)
        tasks_per_user: 每用户任务数
        seed: 随机种子
        arrival_mode: 到达模式 ('poisson' | 'fixed' | 'batch')
        n_uavs: UAV数量
        area_size: 场景大小 (米)

    Returns:
        TaskQueueGenerator实例
    """
    config = TaskQueueConfig(
        n_users=n_users,
        tasks_per_user=tasks_per_user,
        seed=seed,
        arrival_rate=arrival_rate,
        simulation_time=simulation_time,
        arrival_mode=arrival_mode,
        n_uavs=n_uavs,
        area_size=area_size
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

    # 确定UAV数
    if 'n_uavs' in exp_config:
        n_uavs = exp_config['n_uavs']
    else:
        n_uavs = queue_generator.config.n_uavs

    # 获取每用户任务数
    tasks_per_user = queue_generator.config.tasks_per_user
    if queue_generator.config.task_generator is not None:
        tasks_per_user = queue_generator.config.task_generator.tasks_per_user

    # 计算总任务数
    total_tasks = n_users * tasks_per_user

    # 根据任务数调整仿真时长
    arrival_mode = queue_generator.config.arrival_mode
    if arrival_mode == 'poisson' and queue_generator.config.arrival_rate > 0:
        # 保证足够的时间生成所有任务
        simulation_time = (total_tasks / queue_generator.config.arrival_rate) * 1.5 + 10
    else:
        # 固定任务模式
        simulation_time = queue_generator.config.simulation_time

    return {
        'task_queue_enabled': True,
        'arrival_mode': arrival_mode,
        'n_users': n_users,
        'n_uavs': n_uavs,
        'total_tasks': total_tasks,
        'simulation_time': simulation_time,
        'tasks_per_user': tasks_per_user,
        'arrival_rate': queue_generator.config.arrival_rate,
        'area_size': queue_generator.config.area_size,
    }


# ============ 示例 ============

if __name__ == "__main__":
    print("任务队列生成器示例")
    print("=" * 60)

    # ============ 示例1: 基本使用 ============
    print("\n【示例1】基本使用 - 泊松到达模式")
    print("-" * 60)

    queue_gen = create_task_queue_generator(
        arrival_rate=0.05,  # 平均每5秒一个任务
        n_users=50,
        simulation_time=100.0,
        tasks_per_user=5,
        seed=42
    )

    # 生成任务队列
    task_queue = list(queue_gen.generate_task_queue())

    print(f"预期总任务数: {queue_gen.expected_total_tasks}")
    print(f"实际任务数: {len(task_queue)}")

    # ============ 示例2: 链式调用配置 ============
    print("\n【示例2】链式调用配置")
    print("-" * 60)

    queue_gen2 = TaskQueueGenerator()
    queue_gen2.set_n_users(100).set_arrival_rate(0.08).set_tasks_per_user(10)

    config = queue_gen2.get_experiment_config()
    print(f"用户数: {config['n_users']}")
    print(f"到达速率: {config['arrival_rate']}")
    print(f"每用户任务数: {config['tasks_per_user']}")
    print(f"预期任务数: {config['expected_total_tasks']}")

    # ============ 示例3: 实验适配 ============
    print("\n【示例3】实验适配 - 5个实验配置")
    print("-" * 60)

    for exp_id in [1, 2, 3, 4, 5]:
        gen = TaskQueueGenerator().configure_for_experiment(exp_id)
        cfg = gen.get_experiment_config()
        print(f"Exp{exp_id}: 用户={cfg['n_users']:3d}, UAV={cfg['n_uavs']:2d}, "
              f"任务/用户={cfg['tasks_per_user']:2d}, 模式={cfg['arrival_mode']:7s}, "
              f"预期任务={cfg['expected_total_tasks']}")

    # ============ 示例4: 动态调整实验参数 ============
    print("\n【示例4】动态调整实验参数")
    print("-" * 60)

    # Exp2: 用户数扩展实验 (10, 20, 30, 40, 50)
    print("Exp2 - 用户扩展:")
    for n_users in [10, 20, 30, 40, 50]:
        gen = TaskQueueGenerator().configure_for_experiment(2, n_users=n_users)
        cfg = gen.get_experiment_config()
        print(f"  用户={cfg['n_users']:2d}: 预期任务={cfg['expected_total_tasks']}")

    # Exp3: UAV扩展实验 (3, 4, 5, 6, 7, 8)
    print("\nExp3 - UAV扩展:")
    for n_uavs in [3, 4, 5, 6, 7, 8]:
        gen = TaskQueueGenerator().configure_for_experiment(3, n_uavs=n_uavs)
        cfg = gen.get_experiment_config()
        print(f"  UAV={cfg['n_uavs']:2d}: 预期任务={cfg['expected_total_tasks']}")

    # Exp4: 大规模用户扩展 (50, 80, 100, 150, 200)
    print("\nExp4 - 大规模用户扩展:")
    for n_users in [50, 80, 100, 150, 200]:
        gen = TaskQueueGenerator().configure_for_experiment(4, n_users=n_users)
        cfg = gen.get_experiment_config()
        print(f"  用户={cfg['n_users']:3d}: 预期任务={cfg['expected_total_tasks']}")

    # Exp5: 大规模UAV扩展 (10, 12, 15, 18, 20)
    print("\nExp5 - 大规模UAV扩展:")
    for n_uavs in [10, 12, 15, 18, 20]:
        gen = TaskQueueGenerator().configure_for_experiment(5, n_uavs=n_uavs)
        cfg = gen.get_experiment_config()
        print(f"  UAV={cfg['n_uavs']:2d}: 预期任务={cfg['expected_total_tasks']}")

    print("\n" + "=" * 60)
    print("示例完成")

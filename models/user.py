"""
M03: User - 用户与任务模型

功能：定义用户和DNN推理任务的数据结构
输入：用户位置、DNN任务参数
输出：User和Task对象

关键参数 (idea118.txt 0.3节):
    Task_i = (ModelID_i, InputSize_i, L_cut, τ_max, UserLevel)
    pos_i = (x_i, y_i) ∈ R²
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
from enum import Enum, auto
import numpy as np


class TaskState(Enum):
    """
    任务状态枚举
    
    参考 (idea118.txt 4.3节)
    """
    QUEUED = auto()         # 等待执行
    TRANSMITTING = auto()   # 数据上传中
    COMPUTING = auto()      # 边缘计算中
    CHECKPOINTING = auto()  # 保存检查点
    COMPLETED = auto()      # 执行完成
    FAILED = auto()         # 执行失败


class PriorityLevel(Enum):
    """
    任务优先级等级
    
    参考 (idea118.txt 1.4节)
    """
    HIGH = auto()    # 高优先级 (前20%)
    MEDIUM = auto()  # 中优先级 (中间40%)
    LOW = auto()     # 低优先级 (后40%)


@dataclass
class Task:
    """
    DNN推理任务
    
    Attributes:
        task_id: 任务唯一标识
        user_id: 所属用户ID
        model_id: DNN模型标识
        input_size: 输入数据大小 (bits)
        tau_max: 最大容忍时延 (seconds)
        user_level: 用户等级 (1-5)
        
        # 运行时状态
        state: 任务状态
        priority: 综合优先级值 ω_i ∈ [0,1]
        priority_level: 优先级等级
        assigned_uav: 分配的UAV ID
        assigned_cut_layer: 分配的切分层
        start_time: 开始执行时间
        finish_time: 完成时间
    """
    task_id: int
    user_id: int
    model_id: int
    input_size: float  # bits
    tau_max: float  # seconds
    user_level: int  # 1-5
    
    # 计算属性（由阶段0填充）
    total_flops: float = 0.0  # 总计算量
    
    # 运行时状态
    state: TaskState = TaskState.QUEUED
    priority: float = 0.0  # 综合优先级 ω_i
    priority_level: PriorityLevel = PriorityLevel.MEDIUM
    
    # 分配结果（由阶段3填充）
    assigned_uav: Optional[int] = None
    assigned_cut_layer: Optional[int] = None
    assigned_channel: Optional[int] = None
    checkpoint_enabled: bool = False
    checkpoint_layer: Optional[int] = None
    
    # 执行跟踪
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    actual_delay: Optional[float] = None
    
    def get_urgency(self) -> float:
        """
        计算任务紧迫度
        
        公式: Urgency_i = C_i^total / τ_i^max
        
        Returns:
            float: 紧迫度值
        """
        if self.tau_max <= 0:
            return float('inf')
        return self.total_flops / self.tau_max
    
    def is_completed(self) -> bool:
        """任务是否已完成"""
        return self.state == TaskState.COMPLETED
    
    def is_failed(self) -> bool:
        """任务是否失败"""
        return self.state == TaskState.FAILED
    
    def is_active(self) -> bool:
        """任务是否正在执行"""
        return self.state in [TaskState.TRANSMITTING, TaskState.COMPUTING, 
                              TaskState.CHECKPOINTING]
    
    def get_remaining_time(self, current_time: float) -> float:
        """
        计算剩余时间预算
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 剩余时间 (秒)，负值表示超时
        """
        if self.start_time is None:
            return self.tau_max
        elapsed = current_time - self.start_time
        return self.tau_max - elapsed


@dataclass
class User:
    """
    用户实体
    
    Attributes:
        user_id: 用户唯一标识
        x: x坐标 (m)
        y: y坐标 (m)
        task: 用户的DNN任务
        
        # 统计信息
        tasks_completed: 已完成任务数
        tasks_failed: 失败任务数
        total_delay: 累计时延
    """
    user_id: int
    x: float  # 位置x坐标 (m)
    y: float  # 位置y坐标 (m)
    task: Optional[Task] = None
    
    # 统计信息
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_delay: float = 0.0
    
    @property
    def position(self) -> Tuple[float, float]:
        """获取位置坐标"""
        return (self.x, self.y)
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """
        计算到目标点的二维距离
        
        Args:
            other_x: 目标x坐标
            other_y: 目标y坐标
            
        Returns:
            float: 二维欧氏距离 (m)
        """
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)
    
    def distance_to_3d(self, other_x: float, other_y: float, height: float) -> float:
        """
        计算到UAV的三维距离
        
        公式: d_{i,j} = sqrt((x_i-x_j)² + (y_i-y_j)² + H²)
        
        Args:
            other_x: UAV x坐标
            other_y: UAV y坐标
            height: UAV飞行高度
            
        Returns:
            float: 三维欧氏距离 (m)
        """
        return np.sqrt((self.x - other_x) ** 2 + 
                      (self.y - other_y) ** 2 + 
                      height ** 2)


class UserGenerator:
    """
    用户生成器
    
    支持多种用户分布模式
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化生成器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.default_rng(seed)
    
    def generate_uniform(
        self,
        num_users: int,
        scene_width: float,
        scene_height: float,
        model_ids: List[int],
        tau_range: Tuple[float, float] = (1.0, 5.0),
        level_range: Tuple[int, int] = (1, 5),
        input_size_range: Tuple[float, float] = (1e6, 10e6)  # 1-10 Mbits
    ) -> List[User]:
        """
        生成均匀分布的用户
        
        Args:
            num_users: 用户数量
            scene_width: 场景宽度 (m)
            scene_height: 场景高度 (m)
            model_ids: 可用的模型ID列表
            tau_range: 时延范围 (秒)
            level_range: 用户等级范围
            input_size_range: 输入大小范围 (bits)
            
        Returns:
            List[User]: 用户列表
        """
        users = []
        
        for i in range(num_users):
            # 均匀分布的位置
            x = self.rng.uniform(0, scene_width)
            y = self.rng.uniform(0, scene_height)
            
            # 随机选择模型
            model_id = self.rng.choice(model_ids)
            
            # 随机参数
            input_size = self.rng.uniform(*input_size_range)
            tau_max = self.rng.uniform(*tau_range)
            user_level = self.rng.integers(level_range[0], level_range[1] + 1)
            
            task = Task(
                task_id=i,
                user_id=i,
                model_id=model_id,
                input_size=input_size,
                tau_max=tau_max,
                user_level=user_level
            )
            
            user = User(user_id=i, x=x, y=y, task=task)
            users.append(user)
        
        return users
    
    def generate_hotspot(
        self,
        num_users: int,
        scene_width: float,
        scene_height: float,
        model_ids: List[int],
        num_hotspots: int = 3,
        hotspot_sigma: float = 200.0,
        tau_range: Tuple[float, float] = (1.0, 5.0),
        level_range: Tuple[int, int] = (1, 5),
        input_size_range: Tuple[float, float] = (1e6, 10e6)
    ) -> List[User]:
        """
        生成热点分布的用户
        
        Args:
            num_users: 用户数量
            scene_width: 场景宽度 (m)
            scene_height: 场景高度 (m)
            model_ids: 可用的模型ID列表
            num_hotspots: 热点数量
            hotspot_sigma: 热点高斯分布标准差 (m)
            tau_range: 时延范围
            level_range: 用户等级范围
            input_size_range: 输入大小范围
            
        Returns:
            List[User]: 用户列表
        """
        # 随机生成热点中心
        hotspot_centers = [
            (self.rng.uniform(0.2 * scene_width, 0.8 * scene_width),
             self.rng.uniform(0.2 * scene_height, 0.8 * scene_height))
            for _ in range(num_hotspots)
        ]
        
        users = []
        
        for i in range(num_users):
            # 随机选择一个热点
            center = hotspot_centers[self.rng.integers(0, num_hotspots)]
            
            # 高斯分布位置
            x = np.clip(self.rng.normal(center[0], hotspot_sigma), 0, scene_width)
            y = np.clip(self.rng.normal(center[1], hotspot_sigma), 0, scene_height)
            
            model_id = self.rng.choice(model_ids)
            input_size = self.rng.uniform(*input_size_range)
            tau_max = self.rng.uniform(*tau_range)
            user_level = self.rng.integers(level_range[0], level_range[1] + 1)
            
            task = Task(
                task_id=i,
                user_id=i,
                model_id=model_id,
                input_size=input_size,
                tau_max=tau_max,
                user_level=user_level
            )
            
            user = User(user_id=i, x=x, y=y, task=task)
            users.append(user)
        
        return users
    
    def generate_edge(
        self,
        num_users: int,
        scene_width: float,
        scene_height: float,
        model_ids: List[int],
        edge_width: float = 200.0,
        tau_range: Tuple[float, float] = (1.0, 5.0),
        level_range: Tuple[int, int] = (1, 5),
        input_size_range: Tuple[float, float] = (1e6, 10e6)
    ) -> List[User]:
        """
        生成边缘分布的用户
        
        Args:
            num_users: 用户数量
            scene_width: 场景宽度 (m)
            scene_height: 场景高度 (m)
            model_ids: 可用的模型ID列表
            edge_width: 边缘宽度 (m)
            tau_range: 时延范围
            level_range: 用户等级范围
            input_size_range: 输入大小范围
            
        Returns:
            List[User]: 用户列表
        """
        users = []
        
        for i in range(num_users):
            # 随机选择一条边
            edge = self.rng.integers(0, 4)
            
            if edge == 0:  # 上边
                x = self.rng.uniform(0, scene_width)
                y = self.rng.uniform(scene_height - edge_width, scene_height)
            elif edge == 1:  # 下边
                x = self.rng.uniform(0, scene_width)
                y = self.rng.uniform(0, edge_width)
            elif edge == 2:  # 左边
                x = self.rng.uniform(0, edge_width)
                y = self.rng.uniform(0, scene_height)
            else:  # 右边
                x = self.rng.uniform(scene_width - edge_width, scene_width)
                y = self.rng.uniform(0, scene_height)
            
            model_id = self.rng.choice(model_ids)
            input_size = self.rng.uniform(*input_size_range)
            tau_max = self.rng.uniform(*tau_range)
            user_level = self.rng.integers(level_range[0], level_range[1] + 1)
            
            task = Task(
                task_id=i,
                user_id=i,
                model_id=model_id,
                input_size=input_size,
                tau_max=tau_max,
                user_level=user_level
            )
            
            user = User(user_id=i, x=x, y=y, task=task)
            users.append(user)
        
        return users


# ============ 测试用例 ============

def test_user_model():
    """测试User模块"""
    print("=" * 60)
    print("测试 M03: User")
    print("=" * 60)
    
    # 测试1: 创建Task
    print("\n[Test 1] 创建Task...")
    task = Task(
        task_id=0,
        user_id=0,
        model_id=1,
        input_size=5e6,  # 5 Mbits
        tau_max=2.0,     # 2秒
        user_level=3
    )
    task.total_flops = 10e9  # 10 GFLOPs
    
    assert task.state == TaskState.QUEUED, "初始状态应为QUEUED"
    assert not task.is_completed(), "初始不应为完成状态"
    print(f"  任务紧迫度: {task.get_urgency()/1e9:.2f} GFLOPs/s")
    print("  ✓ Task创建成功")
    
    # 测试2: 创建User
    print("\n[Test 2] 创建User...")
    user = User(user_id=0, x=500.0, y=800.0, task=task)
    
    assert user.position == (500.0, 800.0), "位置应为(500, 800)"
    print(f"  用户位置: {user.position}")
    print("  ✓ User创建成功")
    
    # 测试3: 距离计算
    print("\n[Test 3] 测试距离计算...")
    dist_2d = user.distance_to(1000.0, 800.0)
    dist_3d = user.distance_to_3d(1000.0, 800.0, 100.0)
    
    assert abs(dist_2d - 500.0) < 1e-6, "二维距离计算错误"
    expected_3d = np.sqrt(500**2 + 100**2)
    assert abs(dist_3d - expected_3d) < 1e-6, "三维距离计算错误"
    
    print(f"  二维距离: {dist_2d:.2f} m")
    print(f"  三维距离: {dist_3d:.2f} m")
    print("  ✓ 距离计算正确")
    
    # 测试4: 用户生成器 - 均匀分布
    print("\n[Test 4] 测试均匀分布用户生成...")
    generator = UserGenerator(seed=42)
    users = generator.generate_uniform(
        num_users=20,
        scene_width=2000.0,
        scene_height=2000.0,
        model_ids=[1, 2, 3]
    )
    
    assert len(users) == 20, "应生成20个用户"
    
    # 检查位置范围
    all_in_range = all(0 <= u.x <= 2000 and 0 <= u.y <= 2000 for u in users)
    assert all_in_range, "用户位置应在场景范围内"
    print(f"  生成{len(users)}个用户")
    print("  ✓ 均匀分布生成正确")
    
    # 测试5: 用户生成器 - 热点分布
    print("\n[Test 5] 测试热点分布用户生成...")
    users_hotspot = generator.generate_hotspot(
        num_users=50,
        scene_width=2000.0,
        scene_height=2000.0,
        model_ids=[1, 2],
        num_hotspots=3
    )
    
    assert len(users_hotspot) == 50, "应生成50个用户"
    print(f"  生成{len(users_hotspot)}个用户")
    print("  ✓ 热点分布生成正确")
    
    # 测试6: 任务状态转换
    print("\n[Test 6] 测试任务状态...")
    task.state = TaskState.COMPUTING
    assert task.is_active(), "COMPUTING应为活跃状态"
    
    task.state = TaskState.COMPLETED
    assert task.is_completed(), "应为完成状态"
    assert not task.is_active(), "完成后不应为活跃状态"
    print("  ✓ 任务状态转换正确")
    
    # 测试7: 剩余时间计算
    print("\n[Test 7] 测试剩余时间计算...")
    task2 = Task(task_id=1, user_id=1, model_id=1, 
                 input_size=1e6, tau_max=3.0, user_level=2)
    task2.start_time = 10.0
    
    remaining = task2.get_remaining_time(11.5)
    assert abs(remaining - 1.5) < 1e-6, "剩余时间计算错误"
    print(f"  剩余时间: {remaining:.1f} s")
    print("  ✓ 剩余时间计算正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_user_model()

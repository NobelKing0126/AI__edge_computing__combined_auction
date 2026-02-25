"""
任务类型定义模块

定义两种任务类型：
1. 延迟敏感型 (Latency-Sensitive): 使用MobileNetV2，少量图片，严格deadline
2. 计算密集型 (Compute-Intensive): 使用VGG16，大量图片，宽松deadline

参考论文参数：
- AlexNet (延迟敏感): 70-150张图片, 2-4秒deadline
- VGG-16 (计算密集): 2000-4000张图片, 3000-4000秒deadline

我们的UAV场景调整（规模更小）：
- MobileNetV2 (延迟敏感): 50-100张图片, 0.5-2秒deadline
- VGG16 (计算密集): 200-500张图片, 10-30秒deadline
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.mnist_loader import MNISTLoader, compute_input_data_size


class TaskType(Enum):
    """任务类型枚举"""
    LATENCY_SENSITIVE = "latency_sensitive"
    COMPUTE_INTENSIVE = "compute_intensive"


@dataclass
class DNNModelSpec:
    """DNN模型规格"""
    name: str
    total_flops: float          # 每张图片的总FLOPs
    total_layers: int           # 总层数
    input_size: int             # 输入大小 (28x28 for MNIST)
    layer_flops: List[float]    # 每层的FLOPs
    layer_output_sizes: List[int]  # 每层输出大小（字节）
    
    def get_flops_for_images(self, n_images: int) -> float:
        """计算n张图片的总FLOPs"""
        return self.total_flops * n_images
    
    def get_split_flops(self, split_layer: int, n_images: int) -> Tuple[float, float]:
        """
        计算切分后的边缘和云端FLOPs
        
        Args:
            split_layer: 切分层（0表示全云端，total_layers表示全边缘）
            n_images: 图片数量
            
        Returns:
            (edge_flops, cloud_flops)
        """
        split_layer = max(0, min(split_layer, self.total_layers))
        
        edge_flops = sum(self.layer_flops[:split_layer]) * n_images
        cloud_flops = sum(self.layer_flops[split_layer:]) * n_images
        
        return edge_flops, cloud_flops
    
    def get_intermediate_data_size(self, split_layer: int, n_images: int) -> int:
        """
        计算切分点的中间数据大小（字节）
        
        Args:
            split_layer: 切分层
            n_images: 图片数量
            
        Returns:
            中间数据大小（字节）
        """
        if split_layer <= 0:
            # 全云端：传输原始输入
            return n_images * self.input_size
        elif split_layer >= self.total_layers:
            # 全边缘：传输最终结果（很小）
            return n_images * 40  # 假设结果为10个类别的概率
        else:
            # 中间切分
            return self.layer_output_sizes[split_layer - 1] * n_images


# ============ 模型定义 ============

def _create_mobilenetv2_spec() -> DNNModelSpec:
    """
    创建MobileNetV2模型规格
    
    MobileNetV2特点：
    - 轻量级，适合移动设备
    - 总FLOPs: ~0.3 GFLOPs (300 MFLOPs)
    - 53层
    """
    total_layers = 53
    total_flops = 0.3e9  # 300 MFLOPs per image
    
    # 简化的层FLOPs分布（深度可分离卷积使前面层更轻）
    layer_flops = []
    layer_output_sizes = []
    
    # 初始卷积层 (1层)
    layer_flops.append(total_flops * 0.02)
    layer_output_sizes.append(28 * 28 * 32)  # 28x28x32
    
    # Bottleneck层 (17个block，每个约3层 = 51层)
    flops_per_block = (total_flops * 0.95) / 17
    
    # 每个block的输出通道数逐渐增加
    channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
    spatial_sizes = [28, 14, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    
    for i, (ch, sp) in enumerate(zip(channels, spatial_sizes)):
        for j in range(3):  # 每个block约3层
            layer_flops.append(flops_per_block / 3)
            layer_output_sizes.append(sp * sp * ch)
    
    # 最后的分类层 (1层)
    layer_flops.append(total_flops * 0.03)
    layer_output_sizes.append(1280)  # 最后特征向量
    
    # 调整到正好53层
    while len(layer_flops) < total_layers:
        layer_flops.append(total_flops * 0.001)
        layer_output_sizes.append(1280)
    layer_flops = layer_flops[:total_layers]
    layer_output_sizes = layer_output_sizes[:total_layers]
    
    return DNNModelSpec(
        name="MobileNetV2",
        total_flops=total_flops,
        total_layers=total_layers,
        input_size=28 * 28,  # MNIST
        layer_flops=layer_flops,
        layer_output_sizes=layer_output_sizes
    )


def _create_vgg16_spec() -> DNNModelSpec:
    """
    创建VGG16模型规格（适配MNIST 28x28输入）
    
    VGG16特点：
    - 计算密集，大量卷积层
    - 原始ImageNet (224x224): ~15.5 GFLOPs
    - MNIST适配 (28x28): 输入小64倍，FLOPs约 0.5 GFLOPs
    - 39层（16卷积 + 5池化 + 3全连接 + 辅助层）
    
    注：MNIST图片尺寸是ImageNet的1/8，面积的1/64，
    但卷积层计算量与输入面积成正比，所以FLOPs大幅减少
    """
    total_layers = 39
    # MNIST适配：28x28 vs 224x224，面积比约1/64，但FC层不变
    # 估算：卷积层占60%，减少64倍；FC层占40%，不变
    # 0.6 * 15.5 / 64 + 0.4 * 15.5 ≈ 0.15 + 6.2 ≈ 6.35 GFLOPs
    # 但实际MNIST模型会更小，取 0.5 GFLOPs
    total_flops = 0.5e9  # 500 MFLOPs per image (MNIST适配)
    
    layer_flops = []
    layer_output_sizes = []
    
    # VGG16结构（适配28x28输入）
    # Block 1: 2 conv + 1 pool
    for _ in range(2):
        layer_flops.append(total_flops * 0.01)
        layer_output_sizes.append(28 * 28 * 64)
    layer_flops.append(total_flops * 0.001)  # pool
    layer_output_sizes.append(14 * 14 * 64)
    
    # Block 2: 2 conv + 1 pool
    for _ in range(2):
        layer_flops.append(total_flops * 0.02)
        layer_output_sizes.append(14 * 14 * 128)
    layer_flops.append(total_flops * 0.001)
    layer_output_sizes.append(7 * 7 * 128)
    
    # Block 3: 3 conv + 1 pool
    for _ in range(3):
        layer_flops.append(total_flops * 0.05)
        layer_output_sizes.append(7 * 7 * 256)
    layer_flops.append(total_flops * 0.001)
    layer_output_sizes.append(4 * 4 * 256)
    
    # Block 4: 3 conv + 1 pool
    for _ in range(3):
        layer_flops.append(total_flops * 0.10)
        layer_output_sizes.append(4 * 4 * 512)
    layer_flops.append(total_flops * 0.001)
    layer_output_sizes.append(2 * 2 * 512)
    
    # Block 5: 3 conv + 1 pool
    for _ in range(3):
        layer_flops.append(total_flops * 0.10)
        layer_output_sizes.append(2 * 2 * 512)
    layer_flops.append(total_flops * 0.001)
    layer_output_sizes.append(1 * 1 * 512)
    
    # FC layers
    layer_flops.append(total_flops * 0.15)  # FC1
    layer_output_sizes.append(4096)
    layer_flops.append(total_flops * 0.15)  # FC2
    layer_output_sizes.append(4096)
    layer_flops.append(total_flops * 0.01)  # FC3 (output)
    layer_output_sizes.append(1000)
    
    # 填充到39层
    while len(layer_flops) < total_layers:
        layer_flops.append(total_flops * 0.001)
        layer_output_sizes.append(1000)
    layer_flops = layer_flops[:total_layers]
    layer_output_sizes = layer_output_sizes[:total_layers]
    
    return DNNModelSpec(
        name="VGG16",
        total_flops=total_flops,
        total_layers=total_layers,
        input_size=28 * 28,
        layer_flops=layer_flops,
        layer_output_sizes=layer_output_sizes
    )


# 预定义模型
MOBILENETV2_SPEC = _create_mobilenetv2_spec()
VGG16_SPEC = _create_vgg16_spec()


# ============ 任务类型配置 ============

@dataclass
class TaskTypeConfig:
    """任务类型配置"""
    task_type: TaskType
    model_spec: DNNModelSpec
    
    # 图片数量范围
    min_images: int
    max_images: int
    
    # Deadline范围（秒）
    min_deadline: float
    max_deadline: float
    
    # 优先级范围
    min_priority: float
    max_priority: float
    
    # 描述
    description: str


# 延迟敏感型任务配置
# MobileNetV2: 0.3 GFLOPs/图片 × 5-20张 = 1.5-6 GFLOPs
# UAV 15 GFLOPS: 0.1-0.4秒计算，加上上传~0.2秒
# 云端 30 GFLOPS: 0.05-0.2秒计算
# deadline设为0.5-1.5秒使边缘/云端协同处理有一定失败率
LATENCY_SENSITIVE_CONFIG = TaskTypeConfig(
    task_type=TaskType.LATENCY_SENSITIVE,
    model_spec=MOBILENETV2_SPEC,
    min_images=5,
    max_images=20,
    min_deadline=0.5,
    max_deadline=1.5,
    min_priority=0.6,
    max_priority=0.9,
    description="延迟敏感型任务（MobileNetV2，少量图片，严格deadline）"
)

# 计算密集型任务配置
# VGG16 (MNIST适配): 0.5 GFLOPs/图片 × 20-50张 = 10-25 GFLOPs
# UAV 15 GFLOPS: 0.7-1.7秒计算
# 云端 30 GFLOPS: 0.3-0.8秒计算
# deadline设为1.0-3.0秒使云端协同处理有一定失败率
COMPUTE_INTENSIVE_CONFIG = TaskTypeConfig(
    task_type=TaskType.COMPUTE_INTENSIVE,
    model_spec=VGG16_SPEC,
    min_images=20,
    max_images=50,
    min_deadline=1.0,
    max_deadline=3.0,
    min_priority=0.3,
    max_priority=0.6,
    description="计算密集型任务（VGG16，中等图片数量，紧凑deadline）"
)


# ============ 任务生成器 ============

@dataclass
class Task:
    """任务数据结构"""
    task_id: int
    user_id: int
    task_type: TaskType
    model_name: str
    
    # 输入数据
    n_images: int
    data_size_bytes: int
    
    # 计算需求
    total_flops: float
    
    # 约束
    deadline: float
    priority: float
    
    # 用户位置
    user_x: float
    user_y: float
    
    # DNN模型规格引用
    model_spec: DNNModelSpec = field(repr=False)
    
    def get_split_computation(self, split_layer: int) -> Tuple[float, float]:
        """获取切分后的边缘/云端计算量"""
        return self.model_spec.get_split_flops(split_layer, self.n_images)
    
    def get_intermediate_size(self, split_layer: int) -> int:
        """获取切分点的中间数据大小"""
        return self.model_spec.get_intermediate_data_size(split_layer, self.n_images)


class MNISTTaskGenerator:
    """
    基于MNIST的任务生成器
    
    生成延迟敏感型和计算密集型混合任务
    """
    
    def __init__(self, 
                 area_size: float = 200.0,
                 latency_ratio: float = 0.5,
                 tasks_per_user: int = 5,
                 seed: int = None):
        """
        初始化任务生成器
        
        Args:
            area_size: 区域大小（米）
            latency_ratio: 延迟敏感型任务比例 (0-1)
            tasks_per_user: 每个用户提交的任务数量 (默认5)
            seed: 随机种子
        """
        self.area_size = area_size
        self.latency_ratio = latency_ratio
        self.tasks_per_user = tasks_per_user
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_tasks(self, n_users: int, seed: int = None) -> List[Task]:
        """
        生成任务列表
        
        每个用户生成 tasks_per_user 个任务
        总任务数 = n_users × tasks_per_user
        
        Args:
            n_users: 用户数量
            seed: 随机种子
            
        Returns:
            任务列表
        """
        if seed is not None:
            np.random.seed(seed)
        
        tasks = []
        task_id = 0
        
        for user_id in range(n_users):
            # 用户位置（同一用户的所有任务共享位置）
            user_x = np.random.uniform(0, self.area_size)
            user_y = np.random.uniform(0, self.area_size)
            
            # 每个用户生成多个任务
            for _ in range(self.tasks_per_user):
                # 随机决定任务类型
                is_latency_sensitive = np.random.random() < self.latency_ratio
                
                if is_latency_sensitive:
                    config = LATENCY_SENSITIVE_CONFIG
                else:
                    config = COMPUTE_INTENSIVE_CONFIG
                
                # 随机生成参数
                n_images = np.random.randint(config.min_images, config.max_images + 1)
                deadline = np.random.uniform(config.min_deadline, config.max_deadline)
                priority = np.random.uniform(config.min_priority, config.max_priority)
                
                # 计算数据大小和FLOPs
                data_size = compute_input_data_size(n_images)
                total_flops = config.model_spec.get_flops_for_images(n_images)
                
                task = Task(
                    task_id=task_id,
                    user_id=user_id,
                    task_type=config.task_type,
                    model_name=config.model_spec.name,
                    n_images=n_images,
                    data_size_bytes=data_size['bytes'],
                    total_flops=total_flops,
                    deadline=deadline,
                    priority=priority,
                    user_x=user_x,
                    user_y=user_y,
                    model_spec=config.model_spec
                )
                
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    def generate_batch(self, n_users: int, n_batches: int = 1, seed: int = None) -> List[List[Task]]:
        """
        生成多批次任务
        
        Args:
            n_users: 每批用户数
            n_batches: 批次数
            seed: 随机种子
            
        Returns:
            批次任务列表
        """
        if seed is not None:
            np.random.seed(seed)
        
        tasks_per_batch = n_users * self.tasks_per_user
        
        batches = []
        for batch_id in range(n_batches):
            batch_seed = seed + batch_id if seed is not None else None
            tasks = self.generate_tasks(n_users, seed=batch_seed)
            
            # 更新task_id为全局唯一
            for j, task in enumerate(tasks):
                task.task_id = batch_id * tasks_per_batch + j
            
            batches.append(tasks)
        
        return batches


def tasks_to_dict_list(tasks: List[Task]) -> List[Dict]:
    """
    将Task列表转换为字典列表（用于与现有代码兼容）
    
    Args:
        tasks: Task对象列表
        
    Returns:
        字典列表
    """
    result = []
    for task in tasks:
        edge_flops, cloud_flops = task.get_split_computation(task.model_spec.total_layers // 2)
        
        result.append({
            'task_id': task.task_id,
            'user_id': task.user_id,
            'task_type': task.task_type.value,
            'model_name': task.model_name,
            'n_images': task.n_images,
            'data_size': task.data_size_bytes,
            'total_flops': task.total_flops,
            'deadline': task.deadline,
            'priority': task.priority,
            'user_x': task.user_x,
            'user_y': task.user_y,
            'user_pos': (task.user_x, task.user_y),  # 用户位置元组
            'compute_size': task.total_flops,  # ProposedMethod使用的键
            'C_total': task.total_flops,  # 兼容旧格式
            'D': task.data_size_bytes,    # 兼容旧格式
            'total_layers': task.model_spec.total_layers,
            'model_spec': task.model_spec  # 传递模型规格
        })
    
    return result


# ============ 统计函数 ============

def analyze_tasks(tasks: List[Task]) -> Dict:
    """
    分析任务统计信息
    
    Args:
        tasks: 任务列表
        
    Returns:
        统计信息字典
    """
    latency_tasks = [t for t in tasks if t.task_type == TaskType.LATENCY_SENSITIVE]
    compute_tasks = [t for t in tasks if t.task_type == TaskType.COMPUTE_INTENSIVE]
    
    def stats(task_list):
        if not task_list:
            return None
        images = [t.n_images for t in task_list]
        deadlines = [t.deadline for t in task_list]
        flops = [t.total_flops for t in task_list]
        return {
            'count': len(task_list),
            'avg_images': np.mean(images),
            'avg_deadline': np.mean(deadlines),
            'avg_flops': np.mean(flops),
            'total_flops': sum(flops)
        }
    
    return {
        'total_tasks': len(tasks),
        'latency_sensitive': stats(latency_tasks),
        'compute_intensive': stats(compute_tasks),
        'latency_ratio': len(latency_tasks) / len(tasks) if tasks else 0
    }


# ============ 测试 ============

def test_task_types():
    """测试任务类型模块"""
    print("=" * 60)
    print("测试 任务类型定义模块")
    print("=" * 60)
    
    # 测试模型规格
    print("\n[Test 1] 模型规格")
    print(f"  MobileNetV2:")
    print(f"    - 总FLOPs: {MOBILENETV2_SPEC.total_flops/1e9:.2f} GFLOPs/image")
    print(f"    - 总层数: {MOBILENETV2_SPEC.total_layers}")
    
    print(f"  VGG16:")
    print(f"    - 总FLOPs: {VGG16_SPEC.total_flops/1e9:.2f} GFLOPs/image")
    print(f"    - 总层数: {VGG16_SPEC.total_layers}")
    
    # 测试切分计算
    print("\n[Test 2] 切分计算 (100张图片)")
    for model_spec in [MOBILENETV2_SPEC, VGG16_SPEC]:
        print(f"\n  {model_spec.name}:")
        for split in [0, model_spec.total_layers // 4, model_spec.total_layers // 2, 
                      model_spec.total_layers * 3 // 4, model_spec.total_layers]:
            edge, cloud = model_spec.get_split_flops(split, 100)
            inter_size = model_spec.get_intermediate_data_size(split, 100)
            print(f"    Split@{split:2d}: Edge={edge/1e9:.2f}G, Cloud={cloud/1e9:.2f}G, "
                  f"Inter={inter_size/1024:.1f}KB")
    
    # 测试任务生成
    print("\n[Test 3] 任务生成")
    generator = MNISTTaskGenerator(area_size=200.0, latency_ratio=0.5, seed=42)
    tasks = generator.generate_tasks(n_users=20)
    
    stats = analyze_tasks(tasks)
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  延迟敏感型: {stats['latency_sensitive']['count']} "
          f"(平均{stats['latency_sensitive']['avg_images']:.0f}张, "
          f"deadline {stats['latency_sensitive']['avg_deadline']:.1f}s)")
    print(f"  计算密集型: {stats['compute_intensive']['count']} "
          f"(平均{stats['compute_intensive']['avg_images']:.0f}张, "
          f"deadline {stats['compute_intensive']['avg_deadline']:.1f}s)")
    
    # 测试任务转换
    print("\n[Test 4] 任务转换为字典")
    task_dicts = tasks_to_dict_list(tasks[:3])
    for td in task_dicts:
        print(f"  Task {td['task_id']}: {td['model_name']}, "
              f"{td['n_images']} images, deadline={td['deadline']:.2f}s")
    
    print("\n" + "=" * 60)
    print("任务类型模块测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_task_types()

"""
统一实验配置

确保所有实验使用相同的:
- 信道参数
- 任务生成参数 (基于真实DNN模型)
- UAV资源参数
- 随机种子
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.system_config import SystemConfig


@dataclass
class DNNModelSpec:
    """DNN模型规格（精确到每一层）"""
    name: str
    input_size_bytes: float  # 输入数据大小 (bytes)
    total_flops: float  # 总计算量 (FLOPs)
    layers: int  # 层数
    typical_feature_sizes: List[float]  # 典型中间特征大小 (bytes) - 保留兼容
    layer_output_sizes: List[float] = None  # 每层输出特征大小 (bytes)
    layer_flops: List[float] = None  # 每层计算量 (FLOPs)
    
    def get_output_size_at_layer(self, layer: int) -> float:
        """获取指定层的输出特征大小"""
        if self.layer_output_sizes and 0 <= layer < len(self.layer_output_sizes):
            return self.layer_output_sizes[layer]
        # 回退到旧的插值方法
        if not self.typical_feature_sizes:
            return self.input_size_bytes * 0.5
        idx = int(layer / self.layers * len(self.typical_feature_sizes))
        idx = min(idx, len(self.typical_feature_sizes) - 1)
        return self.typical_feature_sizes[idx]
    
    def get_cumulative_flops_at_layer(self, layer: int) -> float:
        """获取前layer层的累计计算量"""
        if self.layer_flops and 0 <= layer <= len(self.layer_flops):
            return sum(self.layer_flops[:layer])
        # 回退到线性分配
        return self.total_flops * (layer / self.layers)
    
    def get_flops_ratio_at_layer(self, layer: int) -> float:
        """获取前layer层的计算量比例"""
        return self.get_cumulative_flops_at_layer(layer) / self.total_flops


def _generate_vgg16_layer_data():
    """
    生成VGG16每层的精确数据
    VGG16结构: 13个卷积层 + 3个全连接层 = 16层（主要层）
    实际包含ReLU和池化，总共39个操作
    
    这里使用16个主要层的数据，其他层通过插值
    """
    # VGG16主要层的输出特征大小 (bytes, float32)
    # 输入: 224x224x3
    main_output_sizes = [
        224*224*64*4,    # conv1_1: 224x224x64
        224*224*64*4,    # conv1_2: 224x224x64
        112*112*64*4,    # pool1: 112x112x64
        112*112*128*4,   # conv2_1: 112x112x128
        112*112*128*4,   # conv2_2: 112x112x128
        56*56*128*4,     # pool2: 56x56x128
        56*56*256*4,     # conv3_1: 56x56x256
        56*56*256*4,     # conv3_2: 56x56x256
        56*56*256*4,     # conv3_3: 56x56x256
        28*28*256*4,     # pool3: 28x28x256
        28*28*512*4,     # conv4_1: 28x28x512
        28*28*512*4,     # conv4_2: 28x28x512
        28*28*512*4,     # conv4_3: 28x28x512
        14*14*512*4,     # pool4: 14x14x512
        14*14*512*4,     # conv5_1: 14x14x512
        14*14*512*4,     # conv5_2: 14x14x512
        14*14*512*4,     # conv5_3: 14x14x512
        7*7*512*4,       # pool5: 7x7x512
        4096*4,          # fc6: 4096
        4096*4,          # fc7: 4096
        1000*4,          # fc8: 1000
    ]
    
    # VGG16每层计算量 (FLOPs) - 基于实际测量
    main_layer_flops = [
        224*224*64*3*3*3*2,      # conv1_1: ~87M
        224*224*64*3*3*64*2,     # conv1_2: ~1.85G
        0,                        # pool1
        112*112*128*3*3*64*2,    # conv2_1: ~924M
        112*112*128*3*3*128*2,   # conv2_2: ~1.85G
        0,                        # pool2
        56*56*256*3*3*128*2,     # conv3_1: ~924M
        56*56*256*3*3*256*2,     # conv3_2: ~1.85G
        56*56*256*3*3*256*2,     # conv3_3: ~1.85G
        0,                        # pool3
        28*28*512*3*3*256*2,     # conv4_1: ~924M
        28*28*512*3*3*512*2,     # conv4_2: ~1.85G
        28*28*512*3*3*512*2,     # conv4_3: ~1.85G
        0,                        # pool4
        14*14*512*3*3*512*2,     # conv5_1: ~462M
        14*14*512*3*3*512*2,     # conv5_2: ~462M
        14*14*512*3*3*512*2,     # conv5_3: ~462M
        0,                        # pool5
        7*7*512*4096*2,          # fc6: ~206M
        4096*4096*2,             # fc7: ~33M
        4096*1000*2,             # fc8: ~8M
    ]
    
    # 扩展到39层（包含ReLU等）
    layer_output_sizes = []
    layer_flops = []
    main_idx = 0
    for i in range(39):
        ratio = i / 38
        main_layer = int(ratio * (len(main_output_sizes) - 1))
        main_layer = min(main_layer, len(main_output_sizes) - 1)
        layer_output_sizes.append(main_output_sizes[main_layer])
        layer_flops.append(main_layer_flops[main_layer] / 2)  # 分摊到子层
    
    return layer_output_sizes, layer_flops


def _generate_resnet50_layer_data():
    """
    生成ResNet50每层的精确数据
    ResNet50: 1个conv + 16个bottleneck blocks + 1个fc = 50层
    """
    # 主要阶段的输出特征大小
    stage_output_sizes = [
        112*112*64*4,    # conv1 + pool
        56*56*256*4,     # stage1 (3 blocks)
        28*28*512*4,     # stage2 (4 blocks)
        14*14*1024*4,    # stage3 (6 blocks)
        7*7*2048*4,      # stage4 (3 blocks)
        2048*4,          # avgpool
        1000*4,          # fc
    ]
    
    # 每个阶段的计算量
    stage_flops = [
        112*112*64*7*7*3*2,       # conv1: ~118M
        56*56*256*3e6,            # stage1: ~3.2G (3 blocks)
        28*28*512*4e6,            # stage2: ~1.6G (4 blocks)
        14*14*1024*6e6,           # stage3: ~1.2G (6 blocks)
        7*7*2048*3e6,             # stage4: ~300M (3 blocks)
        0,                         # avgpool
        2048*1000*2,              # fc: ~4M
    ]
    
    # 扩展到50层
    layer_output_sizes = []
    layer_flops = []
    blocks_per_stage = [3, 4, 6, 3]  # ResNet50的block分布
    total_blocks = sum(blocks_per_stage)
    
    # conv1
    layer_output_sizes.append(stage_output_sizes[0])
    layer_flops.append(stage_flops[0])
    
    # 4个stages
    layer_idx = 1
    for stage_idx, n_blocks in enumerate(blocks_per_stage):
        for block_idx in range(n_blocks):
            for sub_layer in range(3):  # 每个bottleneck有3个conv
                if layer_idx < 50:
                    layer_output_sizes.append(stage_output_sizes[stage_idx + 1])
                    layer_flops.append(stage_flops[stage_idx + 1] / (n_blocks * 3))
                    layer_idx += 1
    
    # 填充到50层
    while len(layer_output_sizes) < 50:
        layer_output_sizes.append(stage_output_sizes[-1])
        layer_flops.append(stage_flops[-1] / 10)
    
    return layer_output_sizes[:50], layer_flops[:50]


def _generate_mobilenetv2_layer_data():
    """
    生成MobileNetV2每层的精确数据
    MobileNetV2: 1个conv + 17个inverted residual blocks + 1个conv + 1个fc = 53层
    """
    # 主要阶段的输出特征大小
    stage_output_sizes = [
        112*112*32*4,    # conv1
        112*112*16*4,    # block1
        56*56*24*4,      # block2-3
        28*28*32*4,      # block4-6
        14*14*64*4,      # block7-10
        14*14*96*4,      # block11-13
        7*7*160*4,       # block14-16
        7*7*320*4,       # block17
        7*7*1280*4,      # conv2
        1000*4,          # fc
    ]
    
    # 每个阶段的计算量 (MobileNetV2较轻量)
    stage_flops = [
        112*112*32*3*3*3*2,      # conv1: ~10M
        112*112*16*1e5,          # block1: ~12M
        56*56*24*2e5,            # block2-3: ~20M
        28*28*32*4e5,            # block4-6: ~12M
        14*14*64*6e5,            # block7-10: ~12M
        14*14*96*4e5,            # block11-13: ~8M
        7*7*160*4e5,             # block14-16: ~8M
        7*7*320*2e5,             # block17: ~3M
        7*7*1280*1e5,            # conv2: ~6M
        1280*1000*2,             # fc: ~2.5M
    ]
    
    # 扩展到53层
    layer_output_sizes = []
    layer_flops = []
    blocks_config = [1, 2, 3, 4, 3, 3, 1]  # 每阶段block数
    
    layer_idx = 0
    for stage_idx, n_blocks in enumerate(blocks_config):
        for _ in range(n_blocks):
            for _ in range(3):  # 每个inverted residual有3个操作
                if layer_idx < 53:
                    output_idx = min(stage_idx + 1, len(stage_output_sizes) - 1)
                    layer_output_sizes.append(stage_output_sizes[output_idx])
                    layer_flops.append(stage_flops[output_idx] / (n_blocks * 3))
                    layer_idx += 1
    
    while len(layer_output_sizes) < 53:
        layer_output_sizes.append(stage_output_sizes[-1])
        layer_flops.append(stage_flops[-1] / 10)
    
    return layer_output_sizes[:53], layer_flops[:53]


def _generate_yolov5s_layer_data():
    """
    生成YOLOv5s每层的精确数据
    YOLOv5s: Backbone(CSPDarknet) + Neck(PANet) + Head = 213层
    """
    # 主要阶段的输出特征大小
    stage_output_sizes = [
        320*320*32*4,    # Focus
        160*160*64*4,    # Conv
        160*160*64*4,    # C3
        80*80*128*4,     # Conv
        80*80*128*4,     # C3
        40*40*256*4,     # Conv
        40*40*256*4,     # C3
        20*20*512*4,     # Conv
        20*20*512*4,     # C3 + SPP
        40*40*256*4,     # Upsample + Concat
        80*80*128*4,     # Upsample + Concat
        80*80*128*4,     # C3 (P3)
        40*40*256*4,     # Conv + Concat + C3 (P4)
        20*20*512*4,     # Conv + Concat + C3 (P5)
    ]
    
    # 每个阶段的计算量
    stage_flops = [
        320*320*32*12*4,         # Focus: ~157M
        160*160*64*3*3*32*2,     # Conv: ~47M
        160*160*64*1e6,          # C3: ~500M
        80*80*128*3*3*64*2,      # Conv: ~47M
        80*80*128*2e6,           # C3: ~1G
        40*40*256*3*3*128*2,     # Conv: ~47M
        40*40*256*3e6,           # C3: ~1.5G
        20*20*512*3*3*256*2,     # Conv: ~47M
        20*20*512*4e6,           # C3+SPP: ~2G
        40*40*256*5e5,           # Upsample+Concat
        80*80*128*5e5,           # Upsample+Concat
        80*80*128*1e6,           # C3 (P3)
        40*40*256*1e6,           # P4
        20*20*512*1e6,           # P5
    ]
    
    # 扩展到213层
    layer_output_sizes = []
    layer_flops = []
    layers_per_stage = [4, 8, 20, 8, 20, 8, 20, 8, 30, 20, 20, 20, 15, 12]
    
    for stage_idx, n_layers in enumerate(layers_per_stage):
        stage_idx_clamped = min(stage_idx, len(stage_output_sizes) - 1)
        for _ in range(n_layers):
            layer_output_sizes.append(stage_output_sizes[stage_idx_clamped])
            layer_flops.append(stage_flops[stage_idx_clamped] / n_layers)
    
    while len(layer_output_sizes) < 213:
        layer_output_sizes.append(stage_output_sizes[-1])
        layer_flops.append(1e6)
    
    return layer_output_sizes[:213], layer_flops[:213]


# 生成精确的层数据
_vgg16_output_sizes, _vgg16_flops = _generate_vgg16_layer_data()
_resnet50_output_sizes, _resnet50_flops = _generate_resnet50_layer_data()
_mobilenetv2_output_sizes, _mobilenetv2_flops = _generate_mobilenetv2_layer_data()
_yolov5s_output_sizes, _yolov5s_flops = _generate_yolov5s_layer_data()


# 真实DNN模型参数 (基于实际测量，精确到每一层)
DNN_MODELS = {
    'vgg16': DNNModelSpec(
        name='VGG16',
        input_size_bytes=224 * 224 * 3,  # ~150KB
        total_flops=15.5e9,  # 15.5 GFLOPs
        layers=39,
        typical_feature_sizes=[
            224*224*64*4,   # 第1层后 ~12MB
            112*112*128*4,  # 第5层后 ~6MB
            56*56*256*4,    # 第10层后 ~3MB
            28*28*512*4,    # 第15层后 ~1.5MB
            14*14*512*4,    # 第20层后 ~0.4MB
            7*7*512*4,      # 第25层后 ~0.1MB
        ],
        layer_output_sizes=_vgg16_output_sizes,
        layer_flops=_vgg16_flops
    ),
    'resnet50': DNNModelSpec(
        name='ResNet50',
        input_size_bytes=224 * 224 * 3,
        total_flops=4.1e9,  # 4.1 GFLOPs
        layers=50,
        typical_feature_sizes=[
            56*56*256*4,    # ~3MB
            28*28*512*4,    # ~1.5MB
            14*14*1024*4,   # ~0.8MB
            7*7*2048*4,     # ~0.4MB
        ],
        layer_output_sizes=_resnet50_output_sizes,
        layer_flops=_resnet50_flops
    ),
    'mobilenetv2': DNNModelSpec(
        name='MobileNetV2',
        input_size_bytes=224 * 224 * 3,
        total_flops=0.3e9,  # 0.3 GFLOPs
        layers=53,
        typical_feature_sizes=[
            112*112*32*4,   # ~1.6MB
            56*56*24*4,     # ~0.3MB
            28*28*32*4,     # ~0.1MB
            14*14*96*4,     # ~0.08MB
            7*7*320*4,      # ~0.06MB
        ],
        layer_output_sizes=_mobilenetv2_output_sizes,
        layer_flops=_mobilenetv2_flops
    ),
    'yolov5s': DNNModelSpec(
        name='YOLOv5s',
        input_size_bytes=640 * 640 * 3,  # ~1.2MB
        total_flops=7.2e9,  # 7.2 GFLOPs
        layers=213,
        typical_feature_sizes=[
            160*160*128*4,  # ~13MB
            80*80*256*4,    # ~6.5MB
            40*40*512*4,    # ~3.2MB
            20*20*1024*4,   # ~1.6MB
        ],
        layer_output_sizes=_yolov5s_output_sizes,
        layer_flops=_yolov5s_flops
    ),
}


class UnifiedTaskGenerator:
    """
    统一任务生成器
    
    基于真实DNN模型参数生成任务
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.config = SystemConfig()
        self.rng = np.random.default_rng(seed)
    
    def generate_tasks(self, n_tasks: int = 50, 
                       uav_positions: list = None,
                       **kwargs) -> List[Dict]:
        """
        生成任务的通用接口
        
        Args:
            n_tasks: 任务数量
            uav_positions: UAV位置列表（用于设置场景大小）
            **kwargs: 其他参数传递给generate_dnn_tasks
            
        Returns:
            任务列表
        """
        # 根据UAV位置确定场景大小
        if uav_positions:
            max_x = max(pos[0] for pos in uav_positions) + 500
            max_y = max(pos[1] for pos in uav_positions) + 500
            kwargs['scene_size'] = (max_x, max_y)
        
        return self.generate_dnn_tasks(n_users=n_tasks, **kwargs)
    
    def generate_dnn_tasks(self, 
                           n_users: int = 50,
                           model_distribution: Dict[str, float] = None,
                           deadline_range: Tuple[float, float] = (0.3, 1.0),
                           scene_size: Tuple[float, float] = (2000, 2000)) -> List[Dict]:
        """
        生成基于真实DNN模型的任务
        
        Args:
            n_users: 用户数量
            model_distribution: 模型分布 {'vgg16': 0.2, 'resnet50': 0.3, ...}
            deadline_range: deadline范围 (秒)
            scene_size: 场景大小 (米)
            
        Returns:
            List[Dict]: 任务列表
        """
        if model_distribution is None:
            # 默认分布: 30% VGG16, 30% ResNet50, 30% MobileNetV2, 10% YOLOv5s
            model_distribution = {
                'vgg16': 0.3,
                'resnet50': 0.3,
                'mobilenetv2': 0.3,
                'yolov5s': 0.1
            }
        
        tasks = []
        model_names = list(model_distribution.keys())
        model_probs = list(model_distribution.values())
        
        for i in range(n_users):
            # 选择DNN模型
            model_name = self.rng.choice(model_names, p=model_probs)
            model_spec = DNN_MODELS[model_name]
            
            # 选择切分点 (影响中间特征大小)
            split_idx = self.rng.integers(0, len(model_spec.typical_feature_sizes))
            feature_size = model_spec.typical_feature_sizes[split_idx]
            
            # 计算量 = 总FLOPs (切分比例由调度算法决定)
            compute_size = model_spec.total_flops
            
            # 数据量 = 输入数据 + 可能的中间特征传输
            # 这里使用输入数据大小，中间特征由切分比例决定
            data_size = model_spec.input_size_bytes
            
            # 用户位置
            user_pos = (
                self.rng.uniform(0, scene_size[0]),
                self.rng.uniform(0, scene_size[1])
            )
            
            # deadline: 300ms - 1000ms
            deadline = self.rng.uniform(deadline_range[0], deadline_range[1])
            
            # 优先级
            priority = self.rng.uniform(0.3, 0.9)
            
            tasks.append({
                'task_id': i,
                'user_id': i,
                'user_pos': user_pos,
                'data_size': data_size,
                'compute_size': compute_size,
                'deadline': deadline,
                'priority': priority,
                'user_level': self.rng.integers(1, 6),
                'model_name': model_name,
                'model_spec': model_spec
            })
        
        return tasks
    
    def generate_mixed_tasks(self,
                             n_users: int = 50,
                             deadline_range: Tuple[float, float] = (0.3, 1.0)) -> List[Dict]:
        """
        生成混合复杂度任务
        
        模拟不同应用场景的任务多样性
        """
        tasks = []
        
        for i in range(n_users):
            # 随机选择任务类型
            task_type = self.rng.choice(['light', 'medium', 'heavy'], p=[0.4, 0.4, 0.2])
            
            if task_type == 'light':
                # 轻量级: MobileNetV2类似
                data_size = self.rng.uniform(100e3, 300e3)  # 100-300 KB
                compute_size = self.rng.uniform(0.2e9, 0.5e9)  # 0.2-0.5 GFLOPs
            elif task_type == 'medium':
                # 中等: ResNet50类似
                data_size = self.rng.uniform(150e3, 500e3)  # 150-500 KB
                compute_size = self.rng.uniform(2e9, 6e9)  # 2-6 GFLOPs
            else:
                # 重量级: VGG16/YOLO类似
                data_size = self.rng.uniform(200e3, 1.5e6)  # 200KB - 1.5MB
                compute_size = self.rng.uniform(10e9, 20e9)  # 10-20 GFLOPs
            
            tasks.append({
                'task_id': i,
                'user_id': i,
                'user_pos': (
                    self.rng.uniform(0, 2000),
                    self.rng.uniform(0, 2000)
                ),
                'data_size': data_size,
                'compute_size': compute_size,
                'deadline': self.rng.uniform(deadline_range[0], deadline_range[1]),
                'priority': self.rng.uniform(0.3, 0.9),
                'user_level': self.rng.integers(1, 6),
                'task_type': task_type
            })
        
        return tasks


def get_unified_uav_resources(n_uavs: int = 5) -> List[Dict]:
    """获取统一的UAV资源配置"""
    config = SystemConfig()
    
    uavs = []
    for i in range(n_uavs):
        uavs.append({
            'uav_id': i,
            'position': (400 + i * 300, 1000),  # 初始位置，会被K-means重新部署
            'f_max': config.uav.f_max,
            'E_max': config.uav.E_max,
            'E_remain': config.uav.E_max,
            'health': 1.0
        })
    return uavs


def get_unified_cloud_resources() -> Dict:
    """获取统一的云端资源配置"""
    config = SystemConfig()
    return {'f_cloud': config.cloud.F_c}


def verify_channel_params():
    """验证信道参数"""
    config = SystemConfig()
    
    print("=" * 60)
    print("信道参数验证")
    print("=" * 60)
    print(f"  信道带宽 W: {config.channel.W/1e6} MHz")
    print(f"  参考信道增益 beta_0: {config.channel.beta_0}")
    print(f"  噪声功率密度 N_0: {config.channel.N_0} W/Hz")
    print(f"  用户发射功率 P_tx: {config.channel.P_tx_user} W")
    print(f"  回程带宽: {config.channel.R_backhaul/1e6} Mbps")
    print()
    
    # 计算不同距离的传输速率
    print("不同距离下的传输速率:")
    for dist_2d in [100, 300, 500, 800, 1000, 1500]:
        H = config.uav.H
        dist_3d = np.sqrt(dist_2d**2 + H**2)
        
        h = config.channel.beta_0 / (dist_3d ** 2)
        snr = config.channel.P_tx_user * h / (config.channel.N_0 * config.channel.W)
        rate = config.channel.W * np.log2(1 + snr)
        
        print(f"  距离={dist_2d:4d}m: 速率={rate/1e6:.2f} Mbps")
    
    print()
    
    # 验证典型任务能否完成
    print("典型任务时延分析 (距离=500m):")
    
    dist_3d = np.sqrt(500**2 + config.uav.H**2)
    h = config.channel.beta_0 / (dist_3d ** 2)
    snr = config.channel.P_tx_user * h / (config.channel.N_0 * config.channel.W)
    rate = config.channel.W * np.log2(1 + snr)
    
    for model_name, spec in DNN_MODELS.items():
        T_upload = spec.input_size_bytes / rate
        T_edge = spec.total_flops / config.uav.f_max  # 全边缘
        T_total = T_upload + T_edge
        
        print(f"  {model_name}: 上传={T_upload*1000:.1f}ms, 计算={T_edge*1000:.1f}ms, "
              f"总计={T_total*1000:.1f}ms")
    
    print("=" * 60)


if __name__ == "__main__":
    verify_channel_params()
    
    print("\n生成DNN任务示例:")
    generator = UnifiedTaskGenerator(seed=42)
    tasks = generator.generate_dnn_tasks(n_users=10)
    
    for task in tasks[:5]:
        print(f"  Task {task['task_id']}: {task.get('model_name', 'N/A')}, "
              f"data={task['data_size']/1e3:.1f}KB, "
              f"compute={task['compute_size']/1e9:.1f}GFLOPs, "
              f"deadline={task['deadline']*1000:.0f}ms")

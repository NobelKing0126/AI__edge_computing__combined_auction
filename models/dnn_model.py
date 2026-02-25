"""
M02: DNNModel - DNN模型结构模块

功能：提取DNN模型每层的FLOPs和输出大小，支持所有层边界切分
输入：PyTorch模型或模型名称
输出：LayerProfile列表，包含每层的计算量和输出大小

关键公式 (idea118.txt 2.3节):
    边缘计算量: C_edge(l) = Σ_{k=1}^{l} FLOPs_k
    云端计算量: C_cloud(l) = Σ_{k=l+1}^{L} FLOPs_k  
    传输数据量: D_trans(l) = OutputSize_l
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


@dataclass
class LayerProfile:
    """
    单层DNN的计算特征
    
    Attributes:
        layer_id: 层索引 (从1开始)
        layer_name: 层名称
        layer_type: 层类型 (Conv2d, Linear, etc.)
        flops: 该层计算量 (浮点运算次数)
        output_size: 该层输出大小 (bits)
        output_shape: 输出张量形状
        is_checkpoint_suitable: 是否适合作为Checkpoint点
    """
    layer_id: int
    layer_name: str
    layer_type: str
    flops: float  # FLOPs
    output_size: float  # bits
    output_shape: Tuple[int, ...]
    is_checkpoint_suitable: bool = True


@dataclass  
class DNNModel:
    """
    DNN模型结构信息
    
    Attributes:
        model_id: 模型唯一标识
        model_name: 模型名称 (vgg16, resnet50, mobilenetv2, yolov5s)
        input_size: 输入尺寸 (C, H, W)
        num_layers: 总层数 L
        layers: 每层的计算特征
        total_flops: 总计算量
        total_params: 总参数量
    """
    model_id: int
    model_name: str
    input_size: Tuple[int, int, int]  # (C, H, W)
    num_layers: int
    layers: List[LayerProfile]
    total_flops: float
    total_params: int
    
    def get_cut_points(self) -> List[int]:
        """
        获取所有可选切分点
        
        Returns:
            List[int]: 切分点列表 [0, 1, 2, ..., L]
            - l=0 表示全云端
            - l=L 表示全边缘
        """
        return list(range(self.num_layers + 1))
    
    def get_edge_compute(self, cut_layer: int) -> float:
        """
        计算边缘计算量 (前cut_layer层)
        
        公式: C_edge(l) = Σ_{k=1}^{l} FLOPs_k
        
        Args:
            cut_layer: 切分层 l ∈ [0, L]
            
        Returns:
            float: 边缘计算量 (FLOPs)
        """
        if cut_layer <= 0:
            return 0.0
        return sum(layer.flops for layer in self.layers[:cut_layer])
    
    def get_cloud_compute(self, cut_layer: int) -> float:
        """
        计算云端计算量 (第cut_layer+1层到第L层)
        
        公式: C_cloud(l) = Σ_{k=l+1}^{L} FLOPs_k
        
        Args:
            cut_layer: 切分层 l ∈ [0, L]
            
        Returns:
            float: 云端计算量 (FLOPs)
        """
        if cut_layer >= self.num_layers:
            return 0.0
        return sum(layer.flops for layer in self.layers[cut_layer:])
    
    def get_transfer_data(self, cut_layer: int) -> float:
        """
        计算传输数据量 (第cut_layer层输出)
        
        公式: D_trans(l) = OutputSize_l
        
        Args:
            cut_layer: 切分层 l ∈ [0, L]
            
        Returns:
            float: 传输数据量 (bits)
        """
        if cut_layer <= 0:
            # 全云端: 传输输入数据
            C, H, W = self.input_size
            return C * H * W * 32  # 32 bits per float
        elif cut_layer >= self.num_layers:
            # 全边缘: 只传输最终输出 (很小)
            return self.layers[-1].output_size
        else:
            return self.layers[cut_layer - 1].output_size
    
    def get_checkpoint_layer(self, cut_layer: int) -> int:
        """
        获取不超过cut_layer的最大可Checkpoint层
        
        公式: l_cp = max{l' : l' ∈ L_cp, l' <= l}
        
        Args:
            cut_layer: 切分层
            
        Returns:
            int: Checkpoint层索引
        """
        for i in range(cut_layer, 0, -1):
            if self.layers[i - 1].is_checkpoint_suitable:
                return i
        return 0
    
    def get_checkpoint_data_size(self, checkpoint_layer: int) -> float:
        """
        计算Checkpoint数据量
        
        公式: S_cp = OutputSize_{l_cp} + S_metadata
        
        Args:
            checkpoint_layer: Checkpoint层索引
            
        Returns:
            float: Checkpoint数据量 (bits)
        """
        if checkpoint_layer <= 0:
            return 0.0
        
        S_metadata = 8 * 1024  # 1KB metadata in bits
        return self.layers[checkpoint_layer - 1].output_size + S_metadata
    
    def summary(self) -> str:
        """返回模型摘要"""
        return f"""
========== DNN模型摘要 ==========
模型名称: {self.model_name}
输入尺寸: {self.input_size}
总层数: {self.num_layers}
总FLOPs: {self.total_flops / 1e9:.2f} GFLOPs
总参数: {self.total_params / 1e6:.2f} M
可切分点: {self.num_layers + 1} 个 (0~{self.num_layers})
=================================
"""


class DNNProfiler:
    """
    DNN模型分析器
    
    用于从PyTorch模型提取每层的FLOPs和输出大小
    """
    
    # 支持的模型配置
    SUPPORTED_MODELS = {
        'vgg16': {
            'input_size': (3, 224, 224),
            'loader': lambda: models.vgg16(weights=None)
        },
        'resnet50': {
            'input_size': (3, 224, 224),
            'loader': lambda: models.resnet50(weights=None)
        },
        'mobilenetv2': {
            'input_size': (3, 224, 224),
            'loader': lambda: models.mobilenet_v2(weights=None)
        }
    }
    
    def __init__(self):
        self._model_cache: Dict[str, DNNModel] = {}
    
    @staticmethod
    def count_conv_flops(module: nn.Conv2d, input_shape: Tuple[int, ...], 
                         output_shape: Tuple[int, ...]) -> float:
        """计算卷积层FLOPs"""
        batch_size = output_shape[0]
        output_channels = output_shape[1]
        output_height = output_shape[2]
        output_width = output_shape[3]
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        output_elements = batch_size * output_channels * output_height * output_width
        
        # 乘加运算
        flops = 2 * kernel_ops * output_elements
        
        # 偏置
        if module.bias is not None:
            flops += output_elements
            
        return flops
    
    @staticmethod
    def count_linear_flops(module: nn.Linear, input_shape: Tuple[int, ...],
                           output_shape: Tuple[int, ...]) -> float:
        """计算全连接层FLOPs"""
        batch_size = output_shape[0]
        # 乘加运算
        flops = 2 * module.in_features * module.out_features * batch_size
        
        if module.bias is not None:
            flops += module.out_features * batch_size
            
        return flops
    
    @staticmethod
    def count_bn_flops(module: nn.BatchNorm2d, input_shape: Tuple[int, ...],
                       output_shape: Tuple[int, ...]) -> float:
        """计算BatchNorm层FLOPs"""
        # 归一化: 减均值、除标准差、缩放、偏移
        elements = np.prod(output_shape)
        return 4 * elements
    
    @staticmethod
    def get_output_size_bits(output_shape: Tuple[int, ...]) -> float:
        """计算输出数据大小 (bits)"""
        return np.prod(output_shape) * 32  # 32 bits per float32
    
    @staticmethod
    def is_checkpoint_suitable(layer_type: str) -> bool:
        """判断层类型是否适合作为Checkpoint点"""
        # 适合: 卷积层后、池化层后、BatchNorm后
        suitable_types = ['Conv2d', 'MaxPool2d', 'AvgPool2d', 'BatchNorm2d', 
                         'Linear', 'AdaptiveAvgPool2d']
        return layer_type in suitable_types
    
    def profile_model(self, model_name: str, model_id: int = 0) -> DNNModel:
        """
        分析DNN模型，提取每层特征
        
        Args:
            model_name: 模型名称 ('vgg16', 'resnet50', 'mobilenetv2')
            model_id: 模型唯一标识
            
        Returns:
            DNNModel: 模型结构信息
        """
        # 检查缓存
        cache_key = f"{model_name}_{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}. 支持: {list(self.SUPPORTED_MODELS.keys())}")
        
        config = self.SUPPORTED_MODELS[model_name]
        input_size = config['input_size']
        model = config['loader']()
        model.eval()
        
        # 使用hook收集每层信息
        layer_profiles: List[LayerProfile] = []
        hooks = []
        layer_info = {}
        
        def make_hook(name: str, module: nn.Module):
            def hook(module, input, output):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    input_shape = tuple(input[0].shape)
                else:
                    input_shape = (1,) + input_size
                    
                if isinstance(output, torch.Tensor):
                    output_shape = tuple(output.shape)
                else:
                    output_shape = (1, 1)
                
                layer_type = module.__class__.__name__
                
                # 计算FLOPs
                if isinstance(module, nn.Conv2d):
                    flops = self.count_conv_flops(module, input_shape, output_shape)
                elif isinstance(module, nn.Linear):
                    flops = self.count_linear_flops(module, input_shape, output_shape)
                elif isinstance(module, nn.BatchNorm2d):
                    flops = self.count_bn_flops(module, input_shape, output_shape)
                else:
                    # 其他层FLOPs较小，忽略
                    flops = 0.0
                
                layer_info[name] = {
                    'type': layer_type,
                    'flops': flops,
                    'output_shape': output_shape,
                    'output_size': self.get_output_size_bits(output_shape)
                }
            return hook
        
        # 注册hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                hook = module.register_forward_hook(make_hook(name, module))
                hooks.append(hook)
        
        # 前向传播
        dummy_input = torch.randn(1, *input_size)
        with torch.no_grad():
            model(dummy_input)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 构建LayerProfile列表 (只保留有意义的层)
        layer_id = 0
        for name, info in layer_info.items():
            # 过滤FLOPs为0且不是池化层的层
            if info['flops'] > 0 or 'Pool' in info['type']:
                layer_id += 1
                layer_profiles.append(LayerProfile(
                    layer_id=layer_id,
                    layer_name=name,
                    layer_type=info['type'],
                    flops=info['flops'],
                    output_size=info['output_size'],
                    output_shape=info['output_shape'],
                    is_checkpoint_suitable=self.is_checkpoint_suitable(info['type'])
                ))
        
        # 计算总FLOPs和参数量
        total_flops = sum(layer.flops for layer in layer_profiles)
        total_params = sum(p.numel() for p in model.parameters())
        
        dnn_model = DNNModel(
            model_id=model_id,
            model_name=model_name,
            input_size=input_size,
            num_layers=len(layer_profiles),
            layers=layer_profiles,
            total_flops=total_flops,
            total_params=total_params
        )
        
        # 缓存
        self._model_cache[cache_key] = dnn_model
        
        return dnn_model
    
    def get_predefined_model(self, model_name: str, model_id: int = 0) -> DNNModel:
        """
        获取预定义的模型结构 (不需要实际加载PyTorch模型)
        
        用于大规模数值模拟，基于实验文档中的模型规格
        
        Args:
            model_name: 模型名称
            model_id: 模型ID
            
        Returns:
            DNNModel: 模型结构信息
        """
        # 实验文档中的模型规格
        PREDEFINED_SPECS = {
            'vgg16': {
                'input_size': (3, 224, 224),
                'num_layers': 16,
                'total_flops': 15.5e9,  # 15.5 GFLOPs
            },
            'resnet50': {
                'input_size': (3, 224, 224),
                'num_layers': 50,
                'total_flops': 4.1e9,  # 4.1 GFLOPs
            },
            'mobilenetv2': {
                'input_size': (3, 224, 224),
                'num_layers': 53,
                'total_flops': 0.3e9,  # 0.3 GFLOPs
            },
            'yolov5s': {
                'input_size': (3, 640, 640),
                'num_layers': 21,
                'total_flops': 7.2e9,  # 7.2 GFLOPs
            }
        }
        
        if model_name not in PREDEFINED_SPECS:
            raise ValueError(f"不支持的预定义模型: {model_name}")
        
        spec = PREDEFINED_SPECS[model_name]
        num_layers = spec['num_layers']
        total_flops = spec['total_flops']
        
        # 生成均匀分布的层（简化模拟）
        flops_per_layer = total_flops / num_layers
        
        # 输出大小随层递减（模拟特征图缩小）
        C, H, W = spec['input_size']
        layers = []
        
        for i in range(num_layers):
            # 简化: 假设每5层特征图尺寸减半
            scale = 2 ** (i // 5)
            h = max(1, H // scale)
            w = max(1, W // scale)
            # 通道数递增
            c = min(512, 64 * (2 ** (i // 4)))
            
            output_shape = (1, c, h, w)
            output_size = c * h * w * 32  # bits
            
            layers.append(LayerProfile(
                layer_id=i + 1,
                layer_name=f"layer_{i+1}",
                layer_type="Conv2d" if i < num_layers - 1 else "Linear",
                flops=flops_per_layer,
                output_size=output_size,
                output_shape=output_shape,
                is_checkpoint_suitable=(i % 3 == 0)  # 每3层一个适合Checkpoint的点
            ))
        
        return DNNModel(
            model_id=model_id,
            model_name=model_name,
            input_size=spec['input_size'],
            num_layers=num_layers,
            layers=layers,
            total_flops=total_flops,
            total_params=0  # 预定义模型不计算参数量
        )


# ============ 测试用例 ============

def test_dnn_model():
    """测试DNNModel模块"""
    print("=" * 60)
    print("测试 M02: DNNModel")
    print("=" * 60)
    
    profiler = DNNProfiler()
    
    # 测试1: 分析VGG16模型
    print("\n[Test 1] 分析VGG16模型...")
    vgg16 = profiler.profile_model('vgg16', model_id=1)
    print(vgg16.summary())
    assert vgg16.num_layers > 0, "层数应大于0"
    assert vgg16.total_flops > 1e9, "VGG16 FLOPs应大于1G"
    print("  ✓ VGG16分析成功")
    
    # 测试2: 测试切分点计算
    print("\n[Test 2] 测试切分点计算...")
    cut_points = vgg16.get_cut_points()
    assert cut_points[0] == 0, "第一个切分点应为0"
    assert cut_points[-1] == vgg16.num_layers, f"最后切分点应为{vgg16.num_layers}"
    print(f"  可切分点数量: {len(cut_points)}")
    print("  ✓ 切分点计算正确")
    
    # 测试3: 测试边缘/云端计算量
    print("\n[Test 3] 测试计算量分割...")
    mid_layer = vgg16.num_layers // 2
    edge_comp = vgg16.get_edge_compute(mid_layer)
    cloud_comp = vgg16.get_cloud_compute(mid_layer)
    total_comp = edge_comp + cloud_comp
    
    print(f"  切分层: {mid_layer}")
    print(f"  边缘计算量: {edge_comp/1e9:.2f} GFLOPs")
    print(f"  云端计算量: {cloud_comp/1e9:.2f} GFLOPs")
    print(f"  总计算量: {total_comp/1e9:.2f} GFLOPs")
    
    # 验证: 边缘+云端 = 总计算量
    assert abs(total_comp - vgg16.total_flops) < 1e6, "边缘+云端应等于总计算量"
    
    # 验证: l=0时边缘为0, l=L时云端为0
    assert vgg16.get_edge_compute(0) == 0, "l=0时边缘计算应为0"
    assert vgg16.get_cloud_compute(vgg16.num_layers) == 0, "l=L时云端计算应为0"
    print("  ✓ 计算量分割正确")
    
    # 测试4: 测试传输数据量
    print("\n[Test 4] 测试传输数据量...")
    trans_mid = vgg16.get_transfer_data(mid_layer)
    trans_0 = vgg16.get_transfer_data(0)  # 全云端
    trans_L = vgg16.get_transfer_data(vgg16.num_layers)  # 全边缘
    
    print(f"  l=0 (全云端) 传输: {trans_0/8/1e6:.2f} MB")
    print(f"  l={mid_layer} (中间) 传输: {trans_mid/8/1e6:.2f} MB")
    print(f"  l={vgg16.num_layers} (全边缘) 传输: {trans_L/8/1e6:.2f} MB")
    
    assert trans_0 > 0, "全云端传输数据应大于0"
    print("  ✓ 传输数据量计算正确")
    
    # 测试5: 测试预定义模型
    print("\n[Test 5] 测试预定义模型...")
    yolo = profiler.get_predefined_model('yolov5s', model_id=4)
    print(yolo.summary())
    assert yolo.num_layers == 21, "YOLOv5s应有21层"
    assert abs(yolo.total_flops - 7.2e9) < 1e8, "YOLOv5s FLOPs应为7.2G"
    print("  ✓ 预定义模型加载正确")
    
    # 测试6: 测试Checkpoint层
    print("\n[Test 6] 测试Checkpoint层选择...")
    cp_layer = vgg16.get_checkpoint_layer(mid_layer)
    cp_size = vgg16.get_checkpoint_data_size(cp_layer)
    print(f"  切分层{mid_layer}的Checkpoint层: {cp_layer}")
    print(f"  Checkpoint数据量: {cp_size/8/1e6:.2f} MB")
    assert cp_layer <= mid_layer, "Checkpoint层应不大于切分层"
    print("  ✓ Checkpoint层选择正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dnn_model()

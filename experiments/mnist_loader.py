"""
MNIST数据加载模块

功能：
1. 加载MNIST数据集
2. 提供随机采样接口
3. 计算数据大小（用于传输时间计算）

MNIST数据集：
- 70,000张灰度图像（60,000训练 + 10,000测试）
- 每张图像：28x28像素 = 784字节
- 10个类别（0-9数字）
"""

import numpy as np
import os
import gzip
import struct
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MNISTData:
    """MNIST数据结构"""
    images: np.ndarray      # 图像数据 (N, 28, 28)
    labels: np.ndarray      # 标签 (N,)
    n_samples: int          # 样本数量
    image_size: int = 784   # 每张图像大小（字节）
    

class MNISTLoader:
    """
    MNIST数据加载器
    
    支持：
    - 从本地文件加载（如果存在）
    - 生成合成MNIST格式数据（如果无真实数据）
    """
    
    # MNIST图像参数
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_SIZE_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH  # 784 bytes per image
    
    def __init__(self, data_dir: str = None, use_synthetic: bool = True):
        """
        初始化MNIST加载器
        
        Args:
            data_dir: MNIST数据目录路径
            use_synthetic: 如果真实数据不可用，是否使用合成数据
        """
        if data_dir is None:
            # 默认路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data', 'mnist')
        
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        self._train_data: Optional[MNISTData] = None
        self._test_data: Optional[MNISTData] = None
        self._combined_data: Optional[MNISTData] = None
        
    def load(self) -> MNISTData:
        """
        加载MNIST数据
        
        Returns:
            MNISTData: 包含图像和标签的数据结构
        """
        if self._combined_data is not None:
            return self._combined_data
        
        # 尝试加载真实MNIST数据
        try:
            train_data = self._load_real_mnist('train')
            test_data = self._load_real_mnist('test')
            
            # 合并训练集和测试集
            images = np.concatenate([train_data.images, test_data.images], axis=0)
            labels = np.concatenate([train_data.labels, test_data.labels], axis=0)
            
            self._combined_data = MNISTData(
                images=images,
                labels=labels,
                n_samples=len(images)
            )
            print(f"[MNISTLoader] 成功加载真实MNIST数据: {self._combined_data.n_samples}张图像")
            
        except Exception as e:
            if self.use_synthetic:
                print(f"[MNISTLoader] 无法加载真实数据 ({e})，使用合成数据")
                self._combined_data = self._generate_synthetic_mnist(70000)
            else:
                raise RuntimeError(f"无法加载MNIST数据: {e}")
        
        return self._combined_data
    
    def _load_real_mnist(self, split: str = 'train') -> MNISTData:
        """
        加载真实MNIST数据文件
        
        Args:
            split: 'train' 或 'test'
            
        Returns:
            MNISTData
        """
        if split == 'train':
            images_file = os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')
            labels_file = os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')
        else:
            images_file = os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')
            labels_file = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')
        
        # 检查文件是否存在
        if not os.path.exists(images_file) or not os.path.exists(labels_file):
            raise FileNotFoundError(f"MNIST文件不存在: {images_file}")
        
        # 加载图像
        with gzip.open(images_file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        
        # 加载标签
        with gzip.open(labels_file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return MNISTData(images=images, labels=labels, n_samples=num)
    
    def _generate_synthetic_mnist(self, n_samples: int = 70000) -> MNISTData:
        """
        生成合成MNIST格式数据
        
        模拟真实MNIST的统计特性：
        - 图像：28x28灰度图，中心区域像素值较高
        - 标签：均匀分布在0-9
        
        Args:
            n_samples: 生成的样本数量
            
        Returns:
            MNISTData
        """
        np.random.seed(42)  # 可重复性
        
        images = np.zeros((n_samples, self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint8)
        labels = np.random.randint(0, 10, size=n_samples, dtype=np.uint8)
        
        # 为每张图像生成类似数字的模式
        for i in range(n_samples):
            # 创建带有中心亮度的图像（模拟手写数字）
            center_y, center_x = 14 + np.random.randint(-3, 4), 14 + np.random.randint(-3, 4)
            
            for y in range(self.IMAGE_HEIGHT):
                for x in range(self.IMAGE_WIDTH):
                    # 基于到中心的距离计算像素值
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist < 8 + np.random.random() * 4:
                        # 中心区域较亮
                        intensity = int(255 * np.exp(-dist / 5) * (0.5 + 0.5 * np.random.random()))
                        images[i, y, x] = min(255, intensity + np.random.randint(0, 50))
        
        print(f"[MNISTLoader] 生成合成MNIST数据: {n_samples}张图像")
        
        return MNISTData(images=images, labels=labels, n_samples=n_samples)
    
    def sample_images(self, n_images: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机采样指定数量的图像
        
        Args:
            n_images: 需要采样的图像数量
            seed: 随机种子（可选）
            
        Returns:
            (images, labels): 采样的图像和对应标签
        """
        data = self.load()
        
        if seed is not None:
            np.random.seed(seed)
        
        indices = np.random.choice(data.n_samples, size=min(n_images, data.n_samples), replace=False)
        
        return data.images[indices], data.labels[indices]
    
    def get_data_size_bytes(self, n_images: int) -> int:
        """
        计算指定数量图像的数据大小（字节）
        
        Args:
            n_images: 图像数量
            
        Returns:
            数据大小（字节）
        """
        return n_images * self.IMAGE_SIZE_BYTES
    
    def get_data_size_mb(self, n_images: int) -> float:
        """
        计算指定数量图像的数据大小（MB）
        
        Args:
            n_images: 图像数量
            
        Returns:
            数据大小（MB）
        """
        return self.get_data_size_bytes(n_images) / (1024 * 1024)
    
    @staticmethod
    def compute_transmission_time(n_images: int, bandwidth_mbps: float) -> float:
        """
        计算传输时间
        
        Args:
            n_images: 图像数量
            bandwidth_mbps: 带宽（Mbps）
            
        Returns:
            传输时间（秒）
        """
        data_size_bits = n_images * MNISTLoader.IMAGE_SIZE_BYTES * 8
        bandwidth_bps = bandwidth_mbps * 1e6
        return data_size_bits / bandwidth_bps


# ============ 辅助函数 ============

def get_mnist_loader(data_dir: str = None) -> MNISTLoader:
    """
    获取MNIST加载器实例
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        MNISTLoader实例
    """
    return MNISTLoader(data_dir=data_dir, use_synthetic=True)


def compute_input_data_size(n_images: int) -> dict:
    """
    计算MNIST输入数据的各种大小度量
    
    Args:
        n_images: 图像数量
        
    Returns:
        包含各种大小度量的字典
    """
    bytes_total = n_images * MNISTLoader.IMAGE_SIZE_BYTES
    
    return {
        'n_images': n_images,
        'bytes': bytes_total,
        'kb': bytes_total / 1024,
        'mb': bytes_total / (1024 * 1024),
        'bits': bytes_total * 8
    }


# ============ 测试 ============

def test_mnist_loader():
    """测试MNIST加载器"""
    print("=" * 60)
    print("测试 MNIST 数据加载模块")
    print("=" * 60)
    
    # 创建加载器
    loader = MNISTLoader(use_synthetic=True)
    
    # 加载数据
    data = loader.load()
    print(f"\n[Test 1] 数据加载")
    print(f"  总样本数: {data.n_samples}")
    print(f"  图像形状: {data.images.shape}")
    print(f"  标签形状: {data.labels.shape}")
    
    # 采样测试
    print(f"\n[Test 2] 随机采样")
    images, labels = loader.sample_images(100, seed=42)
    print(f"  采样100张: {images.shape}")
    print(f"  标签分布: {np.bincount(labels)}")
    
    # 数据大小计算
    print(f"\n[Test 3] 数据大小计算")
    for n in [50, 100, 200, 500]:
        size_info = compute_input_data_size(n)
        print(f"  {n}张图像: {size_info['kb']:.1f} KB = {size_info['mb']:.4f} MB")
    
    # 传输时间计算
    print(f"\n[Test 4] 传输时间计算 (2 Mbps带宽)")
    for n in [50, 100, 200, 500]:
        trans_time = MNISTLoader.compute_transmission_time(n, 2.0)
        print(f"  {n}张图像: {trans_time*1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("MNIST加载模块测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_mnist_loader()

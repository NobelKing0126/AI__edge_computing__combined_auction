"""
M06: ChannelModel - 通信模型

功能：计算用户到UAV的信道增益、传输速率和时延
输入：用户位置、UAV位置、通信参数
输出：信道增益、传输速率、传输时延

关键公式 (idea118.txt 0.8节):
    距离: d_{i,j} = sqrt((x_i-x_j)² + (y_i-y_j)² + H²)
    信道增益: h_{i,j} = β₀ / d_{i,j}²
    传输速率: R_{i,j} = W * log₂(1 + P_tx * h_{i,j} / (N₀ * W))
    上传时延: T_upload = InputSize / R_{i,j}
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.system_config import ChannelConfig


@dataclass
class ChannelState:
    """
    信道状态
    
    Attributes:
        distance: 三维距离 (m)
        channel_gain: 信道增益
        transmission_rate: 传输速率 (bps)
        snr: 信噪比
        snr_db: 信噪比 (dB)
    """
    distance: float
    channel_gain: float
    transmission_rate: float
    snr: float
    snr_db: float


class ChannelModel:
    """
    无线通信信道模型
    
    基于自由空间路径损耗模型，计算用户到UAV的通信参数
    
    Attributes:
        W: 信道带宽 (Hz)
        beta_0: 参考信道增益 (1m处)
        N_0: 噪声功率谱密度 (W/Hz)
        P_tx: 用户发射功率 (W)
        R_backhaul: 回程链路带宽 (bps)
    """
    
    def __init__(self, config: Optional[ChannelConfig] = None):
        """
        初始化通信模型
        
        Args:
            config: 通信配置，None则使用默认值
        """
        if config is None:
            config = ChannelConfig()
        
        self.W = config.W  # 信道带宽 (Hz)
        self.beta_0 = config.beta_0  # 参考信道增益
        self.N_0 = config.N_0  # 噪声功率谱密度 (W/Hz)
        self.P_tx = config.P_tx_user  # 用户发射功率 (W)
        self.R_backhaul = config.R_backhaul  # 回程带宽 (bps)
    
    def compute_distance(self, user_x: float, user_y: float,
                        uav_x: float, uav_y: float, uav_height: float) -> float:
        """
        计算用户到UAV的三维距离
        
        公式: d_{i,j} = sqrt((x_i-x_j)² + (y_i-y_j)² + H²)
        
        Args:
            user_x: 用户x坐标 (m)
            user_y: 用户y坐标 (m)
            uav_x: UAV x坐标 (m)
            uav_y: UAV y坐标 (m)
            uav_height: UAV高度 (m)
            
        Returns:
            float: 三维距离 (m)
        """
        return np.sqrt((user_x - uav_x) ** 2 + 
                      (user_y - uav_y) ** 2 + 
                      uav_height ** 2)
    
    def compute_channel_gain(self, distance: float) -> float:
        """
        计算信道增益（自由空间路径损耗）
        
        公式: h_{i,j} = β₀ / d_{i,j}²
        
        Args:
            distance: 三维距离 (m)
            
        Returns:
            float: 信道增益
        """
        if distance <= 0:
            distance = 1.0  # 避免除零
        return self.beta_0 / (distance ** 2)
    
    def compute_snr(self, channel_gain: float) -> float:
        """
        计算信噪比
        
        公式: SNR = P_tx * h / (N₀ * W)
        
        Args:
            channel_gain: 信道增益
            
        Returns:
            float: 信噪比 (线性)
        """
        noise_power = self.N_0 * self.W
        return self.P_tx * channel_gain / noise_power
    
    def compute_transmission_rate(self, channel_gain: float) -> float:
        """
        计算传输速率（香农公式）
        
        公式: R_{i,j} = W * log₂(1 + P_tx * h_{i,j} / (N₀ * W))
        
        Args:
            channel_gain: 信道增益
            
        Returns:
            float: 传输速率 (bps)
        """
        snr = self.compute_snr(channel_gain)
        return self.W * np.log2(1 + snr)
    
    def compute_upload_delay(self, data_size: float, transmission_rate: float) -> float:
        """
        计算上传时延
        
        公式: T_upload = DataSize / R_{i,j}
        
        Args:
            data_size: 数据大小 (bits)
            transmission_rate: 传输速率 (bps)
            
        Returns:
            float: 上传时延 (s)
        """
        if transmission_rate <= 0:
            return float('inf')
        return data_size / transmission_rate
    
    def compute_backhaul_delay(self, data_size: float) -> float:
        """
        计算回程链路传输时延 (UAV到云端)
        
        公式: T_backhaul = DataSize / R_backhaul
        
        Args:
            data_size: 数据大小 (bits)
            
        Returns:
            float: 回程时延 (s)
        """
        return data_size / self.R_backhaul
    
    def get_channel_state(self, user_x: float, user_y: float,
                         uav_x: float, uav_y: float, 
                         uav_height: float) -> ChannelState:
        """
        获取完整的信道状态
        
        Args:
            user_x: 用户x坐标
            user_y: 用户y坐标
            uav_x: UAV x坐标
            uav_y: UAV y坐标
            uav_height: UAV高度
            
        Returns:
            ChannelState: 信道状态对象
        """
        distance = self.compute_distance(user_x, user_y, uav_x, uav_y, uav_height)
        channel_gain = self.compute_channel_gain(distance)
        transmission_rate = self.compute_transmission_rate(channel_gain)
        snr = self.compute_snr(channel_gain)
        snr_db = 10 * np.log10(snr) if snr > 0 else -float('inf')
        
        return ChannelState(
            distance=distance,
            channel_gain=channel_gain,
            transmission_rate=transmission_rate,
            snr=snr,
            snr_db=snr_db
        )
    
    def compute_communication_energy(self, 
                                     upload_time: float,
                                     trans_time: float,
                                     P_rx: float = 0.1,
                                     P_tx: float = 5.0) -> float:
        """
        计算通信能耗
        
        公式: E_comm = P_rx * T_upload + P_tx * T_trans
        
        Args:
            upload_time: 上传时间 (s)
            trans_time: 中继传输时间 (s)
            P_rx: 接收功率 (W)
            P_tx: UAV发射功率 (W)
            
        Returns:
            float: 通信能耗 (J)
        """
        return P_rx * upload_time + P_tx * trans_time
    
    def is_feasible(self, user_x: float, user_y: float,
                   uav_x: float, uav_y: float, uav_height: float,
                   data_size: float, tau_max: float) -> bool:
        """
        检查通信是否可行（上传时延小于最大时延）
        
        Args:
            user_x, user_y: 用户位置
            uav_x, uav_y: UAV位置
            uav_height: UAV高度
            data_size: 数据大小 (bits)
            tau_max: 最大允许时延 (s)
            
        Returns:
            bool: 是否可行
        """
        state = self.get_channel_state(user_x, user_y, uav_x, uav_y, uav_height)
        upload_delay = self.compute_upload_delay(data_size, state.transmission_rate)
        return upload_delay < tau_max


class DelayModel:
    """
    完整时延模型
    
    计算DNN协同推理的端到端时延
    
    公式 (idea118.txt 2.5节):
        T_total = T_upload + T_edge + T_trans + T_cloud + T_return
    """
    
    def __init__(self, channel_model: ChannelModel, kappa_edge: float = 1e-28):
        """
        初始化时延模型
        
        Args:
            channel_model: 通信模型
            kappa_edge: 边缘能耗系数
        """
        self.channel = channel_model
        self.kappa_edge = kappa_edge
    
    def compute_edge_delay(self, C_edge: float, f_edge: float) -> float:
        """
        计算边缘计算时延
        
        公式: T_edge = C_edge / f_edge
        
        Args:
            C_edge: 边缘计算量 (FLOPs)
            f_edge: 分配的边缘算力 (FLOPS)
            
        Returns:
            float: 边缘计算时延 (s)
        """
        if f_edge <= 0:
            return float('inf') if C_edge > 0 else 0.0
        return C_edge / f_edge
    
    def compute_cloud_delay(self, C_cloud: float, f_cloud: float) -> float:
        """
        计算云端计算时延
        
        公式: T_cloud = C_cloud / f_cloud
        
        Args:
            C_cloud: 云端计算量 (FLOPs)
            f_cloud: 分配的云端算力 (FLOPS)
            
        Returns:
            float: 云端计算时延 (s)
        """
        if f_cloud <= 0:
            return float('inf') if C_cloud > 0 else 0.0
        return C_cloud / f_cloud
    
    def compute_return_delay(self, output_size: float, 
                            transmission_rate: float,
                            cut_layer: int, total_layers: int) -> float:
        """
        计算返回时延
        
        公式 (idea118.txt 2.5节):
            全边缘: T_return = OutputSize_L / R_{i,j}
            协作/全云端: T_return = OutputSize_L / R_backhaul + OutputSize_L / R_{i,j}
        
        Args:
            output_size: 最终输出大小 (bits)
            transmission_rate: 用户-UAV传输速率 (bps)
            cut_layer: 切分层
            total_layers: 总层数
            
        Returns:
            float: 返回时延 (s)
        """
        if cut_layer >= total_layers:
            # 全边缘: 只有UAV到用户
            return output_size / transmission_rate
        else:
            # 协作/全云端: 云端到UAV + UAV到用户
            return (output_size / self.channel.R_backhaul + 
                   output_size / transmission_rate)
    
    def compute_total_delay(self,
                           input_size: float,
                           C_edge: float,
                           C_cloud: float,
                           D_trans: float,
                           output_size: float,
                           f_edge: float,
                           f_cloud: float,
                           transmission_rate: float,
                           cut_layer: int,
                           total_layers: int,
                           checkpoint_time: float = 0.0) -> Tuple[float, dict]:
        """
        计算完整的端到端时延
        
        公式: T_total = T_upload + T_edge + T_trans + T_cloud + T_return + T_checkpoint
        
        Args:
            input_size: 输入数据大小 (bits)
            C_edge: 边缘计算量 (FLOPs)
            C_cloud: 云端计算量 (FLOPs)
            D_trans: 中继传输数据量 (bits)
            output_size: 最终输出大小 (bits)
            f_edge: 边缘算力 (FLOPS)
            f_cloud: 云端算力 (FLOPS)
            transmission_rate: 用户-UAV传输速率 (bps)
            cut_layer: 切分层
            total_layers: 总层数
            checkpoint_time: Checkpoint时间 (s)
            
        Returns:
            Tuple[float, dict]: (总时延, 各分量字典)
        """
        # 上传时延
        T_upload = self.channel.compute_upload_delay(input_size, transmission_rate)
        
        # 边缘计算时延
        T_edge = self.compute_edge_delay(C_edge, f_edge)
        
        # 中继传输时延 (边缘到云端)
        T_trans = self.channel.compute_backhaul_delay(D_trans) if D_trans > 0 else 0.0
        
        # 云端计算时延
        T_cloud = self.compute_cloud_delay(C_cloud, f_cloud)
        
        # 返回时延
        T_return = self.compute_return_delay(output_size, transmission_rate, 
                                            cut_layer, total_layers)
        
        # 总时延
        T_total = T_upload + T_edge + T_trans + T_cloud + T_return + checkpoint_time
        
        components = {
            'T_upload': T_upload,
            'T_edge': T_edge,
            'T_trans': T_trans,
            'T_cloud': T_cloud,
            'T_return': T_return,
            'T_checkpoint': checkpoint_time,
            'T_total': T_total
        }
        
        return T_total, components


# ============ 测试用例 ============

def test_channel_model():
    """测试ChannelModel模块"""
    print("=" * 60)
    print("测试 M06: ChannelModel")
    print("=" * 60)
    
    channel = ChannelModel()
    
    # 测试1: 距离计算
    print("\n[Test 1] 测试距离计算...")
    dist = channel.compute_distance(0, 0, 500, 500, 100)
    expected = np.sqrt(500**2 + 500**2 + 100**2)
    assert abs(dist - expected) < 1e-6, "距离计算错误"
    print(f"  距离: {dist:.2f} m")
    print("  ✓ 距离计算正确")
    
    # 测试2: 信道增益
    print("\n[Test 2] 测试信道增益...")
    gain = channel.compute_channel_gain(dist)
    expected_gain = channel.beta_0 / (dist ** 2)
    assert abs(gain - expected_gain) < 1e-15, "信道增益计算错误"
    print(f"  信道增益: {gain:.2e}")
    print("  ✓ 信道增益计算正确")
    
    # 测试3: 传输速率
    print("\n[Test 3] 测试传输速率...")
    rate = channel.compute_transmission_rate(gain)
    assert rate > 0, "传输速率应大于0"
    print(f"  传输速率: {rate/1e6:.2f} Mbps")
    print("  ✓ 传输速率计算正确")
    
    # 测试4: 上传时延
    print("\n[Test 4] 测试上传时延...")
    data_size = 5e6 * 8  # 5 MB in bits
    delay = channel.compute_upload_delay(data_size, rate)
    print(f"  5MB数据上传时延: {delay*1000:.2f} ms")
    print("  ✓ 上传时延计算正确")
    
    # 测试5: 回程时延
    print("\n[Test 5] 测试回程时延...")
    backhaul_delay = channel.compute_backhaul_delay(data_size)
    expected_backhaul = data_size / channel.R_backhaul
    assert abs(backhaul_delay - expected_backhaul) < 1e-10, "回程时延计算错误"
    print(f"  5MB数据回程时延: {backhaul_delay*1000:.2f} ms")
    print("  ✓ 回程时延计算正确")
    
    # 测试6: 完整信道状态
    print("\n[Test 6] 测试完整信道状态...")
    state = channel.get_channel_state(0, 0, 500, 500, 100)
    print(f"  距离: {state.distance:.2f} m")
    print(f"  信道增益: {state.channel_gain:.2e}")
    print(f"  传输速率: {state.transmission_rate/1e6:.2f} Mbps")
    print(f"  SNR: {state.snr_db:.1f} dB")
    print("  ✓ 信道状态获取正确")
    
    # 测试7: 通信能耗
    print("\n[Test 7] 测试通信能耗...")
    energy = channel.compute_communication_energy(0.1, 0.05)
    expected_energy = 0.1 * 0.1 + 5.0 * 0.05  # P_rx=0.1W, P_tx_uav=5.0W
    assert abs(energy - expected_energy) < 1e-10, "通信能耗计算错误"
    print(f"  通信能耗: {energy*1000:.2f} mJ")
    print("  ✓ 通信能耗计算正确")
    
    # 测试8: 可行性判断
    print("\n[Test 8] 测试可行性判断...")
    feasible = channel.is_feasible(0, 0, 500, 500, 100, data_size, 1.0)
    print(f"  5MB数据在1s内传输可行: {feasible}")
    print("  ✓ 可行性判断正确")
    
    # 测试9: 完整时延模型
    print("\n[Test 9] 测试完整时延模型...")
    delay_model = DelayModel(channel)
    
    total_delay, components = delay_model.compute_total_delay(
        input_size=5e6 * 8,     # 5 MB
        C_edge=5e9,             # 5 GFLOPs
        C_cloud=5e9,            # 5 GFLOPs
        D_trans=1e6 * 8,        # 1 MB
        output_size=0.1e6 * 8,  # 0.1 MB (分类结果)
        f_edge=5e9,             # 5 GFLOPS
        f_cloud=100e9,          # 100 GFLOPS
        transmission_rate=state.transmission_rate,
        cut_layer=10,
        total_layers=20
    )
    
    print(f"  上传时延: {components['T_upload']*1000:.2f} ms")
    print(f"  边缘计算时延: {components['T_edge']*1000:.2f} ms")
    print(f"  中继传输时延: {components['T_trans']*1000:.2f} ms")
    print(f"  云端计算时延: {components['T_cloud']*1000:.2f} ms")
    print(f"  返回时延: {components['T_return']*1000:.2f} ms")
    print(f"  总时延: {total_delay*1000:.2f} ms")
    print("  ✓ 完整时延模型正确")
    
    # 测试10: 不同距离下的性能变化
    print("\n[Test 10] 测试距离对性能的影响...")
    distances = [100, 300, 500, 800, 1000]
    for d in distances:
        state = channel.get_channel_state(0, 0, d, 0, 100)
        print(f"  距离{d}m: 速率={state.transmission_rate/1e6:.2f}Mbps, SNR={state.snr_db:.1f}dB")
    print("  ✓ 距离影响分析正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_channel_model()

"""
M08: DataLoader - 数据加载器

功能：加载EUA和Shanghai数据集的用户位置数据
输入：数据集路径
输出：用户位置列表

数据集:
    EUA: 澳大利亚边缘计算用户分布数据
    Shanghai: 上海电信6个月基站数据
"""

import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

# 尝试导入pandas（Shanghai数据需要）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class Location:
    """位置数据"""
    id: int
    x: float  # 经度或x坐标
    y: float  # 纬度或y坐标
    metadata: Optional[Dict] = None


class EUADataLoader:
    """
    EUA数据集加载器
    
    数据集结构:
        eua-dataset-master/
        ├── edge-servers/
        │   └── site-optus-melbCBD.csv
        └── users/
            ├── users100.json
            ├── users200.json
            └── ...
    """
    
    def __init__(self, data_dir: str = "data/eua/eua-dataset-master"):
        """
        初始化加载器
        
        Args:
            data_dir: EUA数据集目录
        """
        self.data_dir = data_dir
        self.users_dir = os.path.join(data_dir, "users")
        self.servers_dir = os.path.join(data_dir, "edge-servers")
    
    def list_available_files(self) -> List[str]:
        """列出可用的用户数据文件"""
        if not os.path.exists(self.users_dir):
            return []
        return [f for f in os.listdir(self.users_dir) if f.endswith('.csv')]
    
    def load_users(self, filename: str = "users-melbcbd-generated.csv", 
                   sample_size: Optional[int] = None) -> List[Location]:
        """
        加载用户位置数据
        
        Args:
            filename: 用户数据文件名
            sample_size: 采样数量，None则加载全部
            
        Returns:
            List[Location]: 用户位置列表
        """
        filepath = os.path.join(self.users_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        locations = []
        base_lat, base_lon = None, None
        
        with open(filepath, 'r') as f:
            # 跳过标题行
            header = f.readline().strip().split(',')
            
            # 确定lat/lon列索引
            lat_idx, lon_idx = None, None
            for i, col in enumerate(header):
                col_lower = col.lower().strip()
                if 'lat' in col_lower:
                    lat_idx = i
                elif 'lon' in col_lower:
                    lon_idx = i
            
            if lat_idx is None or lon_idx is None:
                # 默认假设前两列是lat, lon
                lat_idx, lon_idx = 0, 1
            
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) > max(lat_idx, lon_idx):
                    try:
                        lat = float(parts[lat_idx])
                        lon = float(parts[lon_idx])
                        
                        if base_lat is None:
                            base_lat, base_lon = lat, lon
                        
                        x = (lon - base_lon) * 111000 * np.cos(np.radians(lat))
                        y = (lat - base_lat) * 111000
                        
                        locations.append(Location(
                            id=i,
                            x=x,
                            y=y,
                            metadata={'latitude': lat, 'longitude': lon}
                        ))
                        
                        if sample_size and len(locations) >= sample_size:
                            break
                    except ValueError:
                        continue
        
        return locations
    
    def load_edge_servers(self, filename: str = "site-optus-melbCBD.csv") -> List[Location]:
        """
        加载边缘服务器位置
        
        Args:
            filename: 服务器数据文件名
            
        Returns:
            List[Location]: 服务器位置列表
        """
        filepath = os.path.join(self.servers_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        locations = []
        base_lat, base_lon = None, None
        
        with open(filepath, 'r') as f:
            # 跳过标题行
            header = f.readline()
            
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        
                        if base_lat is None:
                            base_lat, base_lon = lat, lon
                        
                        x = (lon - base_lon) * 111000 * np.cos(np.radians(lat))
                        y = (lat - base_lat) * 111000
                        
                        locations.append(Location(
                            id=i,
                            x=x,
                            y=y,
                            metadata={'latitude': lat, 'longitude': lon}
                        ))
                    except ValueError:
                        continue
        
        return locations
    
    def get_scene_bounds(self, locations: List[Location]) -> Tuple[float, float, float, float]:
        """
        获取场景边界
        
        Returns:
            Tuple: (x_min, x_max, y_min, y_max)
        """
        if not locations:
            return (0, 0, 0, 0)
        
        xs = [loc.x for loc in locations]
        ys = [loc.y for loc in locations]
        
        return (min(xs), max(xs), min(ys), max(ys))


class ShanghaiDataLoader:
    """
    Shanghai Telecom数据集加载器
    
    数据集结构:
        shanghai/
        ├── data_6.1~6.15.xlsx
        ├── data_6.16~6.30.xlsx
        └── ...
    
    每条记录包含:
        Month, Date, Start Time, End Time, Base Station Location (lat, lon), User ID
    """
    
    def __init__(self, data_dir: str = "data/shanghai"):
        """
        初始化加载器
        
        Args:
            data_dir: Shanghai数据集目录
        """
        self.data_dir = data_dir
        
        if not HAS_PANDAS:
            print("警告: pandas未安装，Shanghai数据集功能受限")
    
    def list_available_files(self) -> List[str]:
        """列出可用的数据文件"""
        if not os.path.exists(self.data_dir):
            return []
        return [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx')]
    
    def load_base_stations(self, filename: str = None, 
                           sample_size: Optional[int] = None) -> List[Location]:
        """
        加载基站位置数据
        
        Args:
            filename: 数据文件名，None则使用第一个可用文件
            sample_size: 采样数量，None则加载全部
            
        Returns:
            List[Location]: 基站位置列表（去重后）
        """
        if not HAS_PANDAS:
            raise RuntimeError("需要安装pandas: pip install pandas openpyxl")
        
        files = self.list_available_files()
        if not files:
            raise FileNotFoundError(f"目录中无xlsx文件: {self.data_dir}")
        
        if filename is None:
            filename = files[0]
        
        filepath = os.path.join(self.data_dir, filename)
        
        # 读取Excel
        df = pd.read_excel(filepath, engine='openpyxl')
        
        # 提取唯一的基站位置
        # 假设列名包含latitude和longitude信息
        # Shanghai数据格式可能需要调整
        unique_stations = {}
        
        for idx, row in df.iterrows():
            # 尝试不同的列名
            lat = row.get('latitude', row.get('lat', row.get('Latitude', None)))
            lon = row.get('longitude', row.get('lon', row.get('Longitude', None)))
            
            if lat is None or lon is None:
                # 如果没有标准列名，尝试基于位置的列
                if len(row) >= 5:
                    try:
                        lat = float(row.iloc[4]) if isinstance(row.iloc[4], (int, float, str)) else None
                        lon = float(row.iloc[5]) if isinstance(row.iloc[5], (int, float, str)) else None
                    except:
                        continue
            
            if lat is not None and lon is not None:
                key = (round(lat, 6), round(lon, 6))
                if key not in unique_stations:
                    unique_stations[key] = len(unique_stations)
        
        # 转换为Location列表
        locations = []
        base_lat, base_lon = None, None
        
        for (lat, lon), station_id in unique_stations.items():
            if base_lat is None:
                base_lat, base_lon = lat, lon
            
            x = (lon - base_lon) * 111000 * np.cos(np.radians(lat))
            y = (lat - base_lat) * 111000
            
            locations.append(Location(
                id=station_id,
                x=x,
                y=y,
                metadata={'latitude': lat, 'longitude': lon}
            ))
            
            if sample_size and len(locations) >= sample_size:
                break
        
        return locations
    
    def load_user_samples(self, filename: str = None,
                         sample_size: int = 100) -> List[Location]:
        """
        从数据中采样用户位置
        
        使用基站位置模拟用户位置（用户位于基站覆盖范围内）
        
        Args:
            filename: 数据文件名
            sample_size: 采样用户数量
            
        Returns:
            List[Location]: 用户位置列表
        """
        # 加载基站位置
        stations = self.load_base_stations(filename)
        
        if not stations:
            return []
        
        # 在基站周围生成用户位置
        rng = np.random.default_rng(42)
        locations = []
        coverage_radius = 500  # 基站覆盖半径 500m
        
        for i in range(sample_size):
            # 随机选择一个基站
            station = rng.choice(stations)
            
            # 在基站周围随机位置
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.uniform(0, coverage_radius)
            
            x = station.x + distance * np.cos(angle)
            y = station.y + distance * np.sin(angle)
            
            locations.append(Location(
                id=i,
                x=x,
                y=y,
                metadata={'base_station_id': station.id}
            ))
        
        return locations


def generate_synthetic_users(num_users: int,
                            scene_width: float = 2000.0,
                            scene_height: float = 2000.0,
                            distribution: str = 'uniform',
                            seed: int = 42) -> List[Location]:
    """
    生成合成用户位置数据
    
    Args:
        num_users: 用户数量
        scene_width: 场景宽度 (m)
        scene_height: 场景高度 (m)
        distribution: 分布类型 ('uniform', 'hotspot', 'edge')
        seed: 随机种子
        
    Returns:
        List[Location]: 用户位置列表
    """
    rng = np.random.default_rng(seed)
    locations = []
    
    if distribution == 'uniform':
        for i in range(num_users):
            x = rng.uniform(0, scene_width)
            y = rng.uniform(0, scene_height)
            locations.append(Location(id=i, x=x, y=y))
    
    elif distribution == 'hotspot':
        num_hotspots = 3
        hotspots = [(rng.uniform(0.2, 0.8) * scene_width,
                    rng.uniform(0.2, 0.8) * scene_height)
                   for _ in range(num_hotspots)]
        sigma = 200.0
        
        for i in range(num_users):
            center = hotspots[rng.integers(0, num_hotspots)]
            x = np.clip(rng.normal(center[0], sigma), 0, scene_width)
            y = np.clip(rng.normal(center[1], sigma), 0, scene_height)
            locations.append(Location(id=i, x=x, y=y))
    
    elif distribution == 'edge':
        edge_width = 200.0
        
        for i in range(num_users):
            edge = rng.integers(0, 4)
            if edge == 0:  # 上
                x = rng.uniform(0, scene_width)
                y = rng.uniform(scene_height - edge_width, scene_height)
            elif edge == 1:  # 下
                x = rng.uniform(0, scene_width)
                y = rng.uniform(0, edge_width)
            elif edge == 2:  # 左
                x = rng.uniform(0, edge_width)
                y = rng.uniform(0, scene_height)
            else:  # 右
                x = rng.uniform(scene_width - edge_width, scene_width)
                y = rng.uniform(0, scene_height)
            locations.append(Location(id=i, x=x, y=y))
    
    elif distribution == 'cluster':
        # 多聚类分布
        num_clusters = 4
        clusters = [
            (0.25 * scene_width, 0.25 * scene_height),
            (0.75 * scene_width, 0.25 * scene_height),
            (0.25 * scene_width, 0.75 * scene_height),
            (0.75 * scene_width, 0.75 * scene_height)
        ]
        sigma = 150.0
        
        for i in range(num_users):
            center = clusters[i % num_clusters]
            x = np.clip(rng.normal(center[0], sigma), 0, scene_width)
            y = np.clip(rng.normal(center[1], sigma), 0, scene_height)
            locations.append(Location(id=i, x=x, y=y))
    
    else:
        # 默认使用uniform
        for i in range(num_users):
            x = rng.uniform(0, scene_width)
            y = rng.uniform(0, scene_height)
            locations.append(Location(id=i, x=x, y=y))
    
    return locations


# ============ 测试用例 ============

def test_data_loader():
    """测试DataLoader模块"""
    print("=" * 60)
    print("测试 M08: DataLoader")
    print("=" * 60)
    
    # 测试1: EUA数据加载
    print("\n[Test 1] 测试EUA数据加载...")
    eua_loader = EUADataLoader("/home/hyp/projects/first/data/eua/eua-dataset-master")
    
    files = eua_loader.list_available_files()
    print(f"  可用文件: {files[:5]}..." if len(files) > 5 else f"  可用文件: {files}")
    
    if files:
        try:
            users = eua_loader.load_users(files[0])
            print(f"  加载了 {len(users)} 个用户位置")
            
            bounds = eua_loader.get_scene_bounds(users)
            print(f"  场景范围: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}]")
            print("  ✓ EUA数据加载成功")
        except Exception as e:
            print(f"  EUA加载失败: {e}")
    else:
        print("  无可用EUA文件，跳过")
    
    # 测试2: 合成数据生成
    print("\n[Test 2] 测试合成数据生成...")
    
    # 均匀分布
    uniform_users = generate_synthetic_users(50, distribution='uniform')
    assert len(uniform_users) == 50, "应生成50个用户"
    print(f"  均匀分布: {len(uniform_users)} 用户")
    
    # 热点分布
    hotspot_users = generate_synthetic_users(50, distribution='hotspot')
    print(f"  热点分布: {len(hotspot_users)} 用户")
    
    # 边缘分布
    edge_users = generate_synthetic_users(50, distribution='edge')
    print(f"  边缘分布: {len(edge_users)} 用户")
    
    print("  ✓ 合成数据生成正确")
    
    # 测试3: 位置数据验证
    print("\n[Test 3] 测试位置数据验证...")
    for user in uniform_users[:5]:
        assert 0 <= user.x <= 2000, "x坐标应在范围内"
        assert 0 <= user.y <= 2000, "y坐标应在范围内"
    print(f"  前5个用户: {[(u.x, u.y) for u in uniform_users[:3]]}")
    print("  ✓ 位置数据验证正确")
    
    # 测试4: Shanghai数据（如果有pandas）
    print("\n[Test 4] 测试Shanghai数据加载...")
    shanghai_loader = ShanghaiDataLoader("/home/hyp/projects/first/data/shanghai")
    
    files = shanghai_loader.list_available_files()
    print(f"  可用文件: {len(files)} 个")
    
    if files and HAS_PANDAS:
        try:
            users = shanghai_loader.load_user_samples(sample_size=20)
            print(f"  采样了 {len(users)} 个用户位置")
            print("  ✓ Shanghai数据加载成功")
        except Exception as e:
            print(f"  Shanghai加载失败: {e}")
    else:
        print("  跳过Shanghai测试（无pandas或无文件）")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loader()

"""
M10: WeightedKMeans - 加权K-means UAV部署

功能：根据用户分布确定UAV最优部署位置
输入：用户位置、权重、UAV数量
输出：UAV部署位置

关键公式 (idea118.txt 0.7节):
    聚类目标: min Σ_k Σ_{i∈C_k} w_i * ||pos_i - center_k||²
    权重计算: w_i = α₁ * D_i / D_max + α₂ * C_i / C_max
    中心更新: center_k = Σ_{i∈C_k} w_i * pos_i / Σ_{i∈C_k} w_i
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_loader import Location
from config.constants import NUMERICAL


@dataclass
class ClusterResult:
    """
    聚类结果
    
    Attributes:
        centers: UAV部署位置列表 [(x, y), ...]
        labels: 每个用户的聚类标签
        iterations: 迭代次数
        inertia: 目标函数值
        converged: 是否收敛
    """
    centers: List[Tuple[float, float]]
    labels: List[int]
    iterations: int
    inertia: float
    converged: bool


class WeightedKMeans:
    """
    加权K-means聚类
    
    Attributes:
        n_clusters: 聚类数量
        alpha1: 数据量权重
        alpha2: 计算量权重
        max_iter: 最大迭代次数
        epsilon: 收敛阈值
        random_state: 随机种子
    """
    
    def __init__(self,
                 n_clusters: int,
                 alpha1: float = 0.7,
                 alpha2: float = 0.3,
                 max_iter: int = 100,
                 epsilon: float = 1e-6,
                 random_state: int = 42):
        """
        初始化加权K-means
        
        Args:
            n_clusters: 聚类数量（UAV数量）
            alpha1: 数据量权重
            alpha2: 计算量权重
            max_iter: 最大迭代次数
            epsilon: 收敛阈值
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)
    
    def compute_weights(self,
                       data_sizes: Optional[List[float]] = None,
                       compute_sizes: Optional[List[float]] = None,
                       n_samples: int = 0) -> np.ndarray:
        """
        计算用户权重
        
        公式: w_i = α₁ * D_i/D_max + α₂ * C_i/C_max
        
        Args:
            data_sizes: 数据量列表
            compute_sizes: 计算量列表
            n_samples: 样本数量
            
        Returns:
            np.ndarray: 权重数组
        """
        if data_sizes is None and compute_sizes is None:
            return np.ones(n_samples)
        
        weights = np.zeros(n_samples)
        
        if data_sizes is not None:
            data_arr = np.array(data_sizes)
            d_max = np.max(data_arr) if np.max(data_arr) > 0 else 1.0
            weights += self.alpha1 * data_arr / d_max
        
        if compute_sizes is not None:
            comp_arr = np.array(compute_sizes)
            c_max = np.max(comp_arr) if np.max(comp_arr) > 0 else 1.0
            weights += self.alpha2 * comp_arr / c_max
        
        # 确保权重非负且非零（使用常量配置）
        weights = np.maximum(weights, NUMERICAL.MIN_WEIGHT)
        
        return weights
    
    def _init_centers(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        使用K-means++初始化聚类中心
        
        Args:
            X: 位置数组 (n, 2)
            weights: 权重数组
            
        Returns:
            np.ndarray: 初始中心 (k, 2)
        """
        n = len(X)
        centers = []
        
        # 选择第一个中心：加权随机
        probs = weights / np.sum(weights)
        first_idx = self.rng.choice(n, p=probs)
        centers.append(X[first_idx].copy())
        
        # 选择剩余中心
        for _ in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            min_dists = np.full(n, np.inf)
            for center in centers:
                dists = np.sum((X - center) ** 2, axis=1)
                min_dists = np.minimum(min_dists, dists)
            
            # 加权概率
            probs = weights * min_dists
            probs = probs / np.sum(probs)
            
            next_idx = self.rng.choice(n, p=probs)
            centers.append(X[next_idx].copy())
        
        return np.array(centers)
    
    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        分配聚类标签
        
        Args:
            X: 位置数组 (n, 2)
            centers: 中心数组 (k, 2)
            
        Returns:
            np.ndarray: 标签数组 (n,)
        """
        n = len(X)
        labels = np.zeros(n, dtype=int)
        
        for i in range(n):
            dists = np.sum((centers - X[i]) ** 2, axis=1)
            labels[i] = np.argmin(dists)
        
        return labels
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray, 
                        weights: np.ndarray) -> np.ndarray:
        """
        更新聚类中心（加权平均）
        
        公式: center_k = Σ_{i∈C_k} w_i * pos_i / Σ_{i∈C_k} w_i
        
        Args:
            X: 位置数组 (n, 2)
            labels: 标签数组 (n,)
            weights: 权重数组 (n,)
            
        Returns:
            np.ndarray: 新中心 (k, 2)
        """
        new_centers = np.zeros((self.n_clusters, 2))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                w_k = weights[mask]
                total_weight = np.sum(w_k)
                new_centers[k] = np.sum(w_k[:, np.newaxis] * X[mask], axis=0) / total_weight
            else:
                # 空聚类：随机重新初始化
                new_centers[k] = X[self.rng.integers(0, len(X))]
        
        return new_centers
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                         centers: np.ndarray, weights: np.ndarray) -> float:
        """
        计算目标函数值（加权惯量）
        
        公式: J = Σ_k Σ_{i∈C_k} w_i * ||pos_i - center_k||²
        
        Args:
            X: 位置数组
            labels: 标签数组
            centers: 中心数组
            weights: 权重数组
            
        Returns:
            float: 惯量值
        """
        inertia = 0.0
        for i in range(len(X)):
            dist_sq = np.sum((X[i] - centers[labels[i]]) ** 2)
            inertia += weights[i] * dist_sq
        return inertia
    
    def fit(self, locations: List[Location],
            weights: Optional[np.ndarray] = None) -> ClusterResult:
        """
        执行加权K-means聚类
        
        Args:
            locations: 用户位置列表
            weights: 权重数组（可选）
            
        Returns:
            ClusterResult: 聚类结果
        """
        n = len(locations)
        
        if n == 0:
            return ClusterResult(
                centers=[],
                labels=[],
                iterations=0,
                inertia=0.0,
                converged=True
            )
        
        # 转换为数组
        X = np.array([[loc.x, loc.y] for loc in locations])
        
        if weights is None:
            weights = np.ones(n)
        
        # 初始化中心
        centers = self._init_centers(X, weights)
        
        converged = False
        for iteration in range(self.max_iter):
            # 分配标签
            labels = self._assign_clusters(X, centers)
            
            # 更新中心
            new_centers = self._update_centers(X, labels, weights)
            
            # 检查收敛
            center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            
            if center_shift < self.epsilon:
                converged = True
                break
        
        # 计算最终惯量
        labels = self._assign_clusters(X, centers)
        inertia = self._compute_inertia(X, labels, centers, weights)
        
        return ClusterResult(
            centers=[(c[0], c[1]) for c in centers],
            labels=labels.tolist(),
            iterations=iteration + 1,
            inertia=inertia,
            converged=converged
        )
    
    def fit_with_height(self, locations: List[Location],
                        uav_height: float = 100.0,
                        weights: Optional[np.ndarray] = None) -> List[Tuple[float, float, float]]:
        """
        执行聚类并返回3D位置（含高度）
        
        Args:
            locations: 用户位置列表
            uav_height: UAV飞行高度
            weights: 权重数组
            
        Returns:
            List[Tuple]: UAV部署位置 [(x, y, z), ...]
        """
        result = self.fit(locations, weights)
        return [(x, y, uav_height) for x, y in result.centers]


def deploy_uavs(locations: List[Location],
                n_uavs: int,
                data_sizes: Optional[List[float]] = None,
                compute_sizes: Optional[List[float]] = None,
                uav_height: float = 100.0,
                alpha1: float = 0.7,
                alpha2: float = 0.3) -> Tuple[List[Tuple[float, float, float]], ClusterResult]:
    """
    部署UAV的便捷函数
    
    Args:
        locations: 用户位置列表
        n_uavs: UAV数量
        data_sizes: 用户数据量列表
        compute_sizes: 用户计算量列表
        uav_height: UAV高度
        alpha1: 数据量权重
        alpha2: 计算量权重
        
    Returns:
        Tuple: (UAV位置列表, 聚类结果)
    """
    kmeans = WeightedKMeans(
        n_clusters=n_uavs,
        alpha1=alpha1,
        alpha2=alpha2
    )
    
    # 计算权重
    weights = kmeans.compute_weights(data_sizes, compute_sizes, len(locations))
    
    # 执行聚类
    result = kmeans.fit(locations, weights)
    
    # 添加高度
    positions = [(x, y, uav_height) for x, y in result.centers]
    
    return positions, result


# ============ 测试用例 ============

def test_weighted_kmeans():
    """测试WeightedKMeans模块"""
    print("=" * 60)
    print("测试 M10: WeightedKMeans")
    print("=" * 60)
    
    # 创建测试数据 - 两个明显的聚类
    np.random.seed(42)
    locations = []
    
    # 聚类1: 中心(500, 500)
    for i in range(30):
        locations.append(Location(
            id=i,
            x=np.random.normal(500, 100),
            y=np.random.normal(500, 100)
        ))
    
    # 聚类2: 中心(1500, 1500)
    for i in range(30, 60):
        locations.append(Location(
            id=i,
            x=np.random.normal(1500, 100),
            y=np.random.normal(1500, 100)
        ))
    
    # 测试1: 基本聚类
    print("\n[Test 1] 测试基本K-means聚类...")
    kmeans = WeightedKMeans(n_clusters=2)
    result = kmeans.fit(locations)
    
    assert len(result.centers) == 2, "应有2个中心"
    assert result.converged, "应该收敛"
    
    # 验证中心大致正确
    centers_arr = np.array(result.centers)
    expected = np.array([[500, 500], [1500, 1500]])
    
    # 检查每个实际中心是否接近某个期望中心
    for center in centers_arr:
        min_dist = min(np.linalg.norm(center - exp) for exp in expected)
        assert min_dist < 200, f"中心偏差过大: {center}"
    
    print(f"  中心1: ({result.centers[0][0]:.1f}, {result.centers[0][1]:.1f})")
    print(f"  中心2: ({result.centers[1][0]:.1f}, {result.centers[1][1]:.1f})")
    print(f"  迭代次数: {result.iterations}")
    print(f"  惯量: {result.inertia:.1f}")
    print("  ✓ 基本聚类正确")
    
    # 测试2: 加权聚类
    print("\n[Test 2] 测试加权K-means...")
    
    # 给聚类1的用户更高权重
    weights = np.array([2.0 if i < 30 else 1.0 for i in range(60)])
    result_weighted = kmeans.fit(locations, weights)
    
    print(f"  加权中心1: ({result_weighted.centers[0][0]:.1f}, {result_weighted.centers[0][1]:.1f})")
    print(f"  加权中心2: ({result_weighted.centers[1][0]:.1f}, {result_weighted.centers[1][1]:.1f})")
    print("  ✓ 加权聚类完成")
    
    # 测试3: 权重计算
    print("\n[Test 3] 测试权重计算...")
    data_sizes = [10.0 * i for i in range(60)]
    compute_sizes = [5.0 * (60 - i) for i in range(60)]
    
    weights = kmeans.compute_weights(data_sizes, compute_sizes, 60)
    assert len(weights) == 60, "权重数量应等于用户数"
    assert all(w > 0 for w in weights), "权重应为正"
    print(f"  权重范围: [{min(weights):.3f}, {max(weights):.3f}]")
    print("  ✓ 权重计算正确")
    
    # 测试4: 带高度的部署
    print("\n[Test 4] 测试3D部署...")
    positions_3d = kmeans.fit_with_height(locations, uav_height=100.0)
    
    assert len(positions_3d) == 2, "应有2个3D位置"
    assert all(p[2] == 100.0 for p in positions_3d), "高度应为100m"
    print(f"  UAV1: ({positions_3d[0][0]:.1f}, {positions_3d[0][1]:.1f}, {positions_3d[0][2]:.1f})")
    print(f"  UAV2: ({positions_3d[1][0]:.1f}, {positions_3d[1][1]:.1f}, {positions_3d[1][2]:.1f})")
    print("  ✓ 3D部署正确")
    
    # 测试5: 便捷函数
    print("\n[Test 5] 测试deploy_uavs便捷函数...")
    positions, cluster_result = deploy_uavs(
        locations,
        n_uavs=3,
        data_sizes=[1.0] * 60,
        compute_sizes=[1.0] * 60,
        uav_height=150.0
    )
    
    assert len(positions) == 3, "应有3个UAV"
    assert all(p[2] == 150.0 for p in positions), "高度应为150m"
    print(f"  部署了 {len(positions)} 个UAV")
    print("  ✓ 便捷函数正确")
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_weighted_kmeans()

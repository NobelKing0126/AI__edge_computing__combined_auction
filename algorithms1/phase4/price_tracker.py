"""
价格历史记录模块

功能：
1. 追踪UAV计算资源价格随时间的变化
2. 记录利用率与价格的关系
3. 支持可视化分析

用于小规模实验的价格动态折线图
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy


@dataclass
class PriceSnapshot:
    """价格快照"""
    time_step: int
    timestamp: float               # 实际时间戳
    uav_prices: Dict[int, float]   # {uav_id: price}
    uav_utilizations: Dict[int, float]  # {uav_id: utilization}
    avg_price: float
    avg_utilization: float
    
    # 任务相关
    tasks_processed: int
    tasks_successful: int


class PriceTracker:
    """
    价格追踪器
    
    记录UAV计算资源价格的动态变化
    """
    
    def __init__(self, n_uavs: int, initial_price: float = 1.0):
        """
        初始化价格追踪器
        
        Args:
            n_uavs: UAV数量
            initial_price: 初始价格
        """
        self.n_uavs = n_uavs
        self.initial_price = initial_price
        
        # 当前状态
        self.current_prices: Dict[int, float] = {i: initial_price for i in range(n_uavs)}
        self.current_utilizations: Dict[int, float] = {i: 0.0 for i in range(n_uavs)}
        
        # 历史记录
        self.snapshots: List[PriceSnapshot] = []
        self.time_step = 0
        
        # 累计统计
        self.total_tasks_processed = 0
        self.total_tasks_successful = 0
    
    def record_snapshot(self, 
                        prices: Dict[int, float] = None,
                        utilizations: Dict[int, float] = None,
                        tasks_processed: int = 0,
                        tasks_successful: int = 0,
                        timestamp: float = None):
        """
        记录当前价格快照
        
        Args:
            prices: UAV价格字典 (如果为None，使用当前价格)
            utilizations: UAV利用率字典
            tasks_processed: 本轮处理的任务数
            tasks_successful: 本轮成功的任务数
            timestamp: 时间戳
        """
        if prices is not None:
            self.current_prices = copy.deepcopy(prices)
        
        if utilizations is not None:
            self.current_utilizations = copy.deepcopy(utilizations)
        
        # 计算平均值
        avg_price = np.mean(list(self.current_prices.values()))
        avg_util = np.mean(list(self.current_utilizations.values()))
        
        # 更新累计统计
        self.total_tasks_processed += tasks_processed
        self.total_tasks_successful += tasks_successful
        
        snapshot = PriceSnapshot(
            time_step=self.time_step,
            timestamp=timestamp if timestamp else float(self.time_step),
            uav_prices=copy.deepcopy(self.current_prices),
            uav_utilizations=copy.deepcopy(self.current_utilizations),
            avg_price=avg_price,
            avg_utilization=avg_util,
            tasks_processed=self.total_tasks_processed,
            tasks_successful=self.total_tasks_successful
        )
        
        self.snapshots.append(snapshot)
        self.time_step += 1
    
    def update_price(self, uav_id: int, new_price: float):
        """更新单个UAV的价格"""
        if uav_id in self.current_prices:
            self.current_prices[uav_id] = new_price
    
    def update_utilization(self, uav_id: int, utilization: float):
        """更新单个UAV的利用率"""
        if uav_id in self.current_utilizations:
            self.current_utilizations[uav_id] = utilization
    
    def get_price_history(self) -> Dict[str, List]:
        """
        获取价格历史数据（用于绘图）
        
        Returns:
            {
                'time_steps': [...],
                'uav_prices': {uav_id: [...]},
                'avg_price': [...],
                'utilizations': {uav_id: [...]},
                'avg_utilization': [...]
            }
        """
        if not self.snapshots:
            return {
                'time_steps': [],
                'uav_prices': {},
                'avg_price': [],
                'utilizations': {},
                'avg_utilization': []
            }
        
        time_steps = [s.time_step for s in self.snapshots]
        avg_prices = [s.avg_price for s in self.snapshots]
        avg_utils = [s.avg_utilization for s in self.snapshots]
        
        # 每个UAV的价格历史
        uav_prices = {}
        uav_utils = {}
        for uav_id in range(self.n_uavs):
            uav_prices[uav_id] = [s.uav_prices.get(uav_id, self.initial_price) 
                                  for s in self.snapshots]
            uav_utils[uav_id] = [s.uav_utilizations.get(uav_id, 0.0) 
                                for s in self.snapshots]
        
        return {
            'time_steps': time_steps,
            'uav_prices': uav_prices,
            'avg_price': avg_prices,
            'utilizations': uav_utils,
            'avg_utilization': avg_utils
        }
    
    def get_convergence_metrics(self) -> Dict:
        """
        计算价格收敛指标
        
        Returns:
            收敛相关指标
        """
        if len(self.snapshots) < 2:
            return {
                'converged': False,
                'final_avg_price': self.initial_price,
                'price_variance': 0.0,
                'convergence_step': -1
            }
        
        avg_prices = [s.avg_price for s in self.snapshots]
        
        # 计算最后几步的方差
        last_n = min(5, len(avg_prices))
        recent_variance = np.var(avg_prices[-last_n:])
        
        # 判断是否收敛（方差小于阈值）
        convergence_threshold = 0.01
        converged = recent_variance < convergence_threshold
        
        # 找到收敛点（如果收敛）
        convergence_step = -1
        if converged:
            for i in range(len(avg_prices) - last_n, -1, -1):
                window = avg_prices[i:i+last_n]
                if np.var(window) > convergence_threshold:
                    convergence_step = i + last_n
                    break
            if convergence_step == -1:
                convergence_step = 0
        
        return {
            'converged': converged,
            'final_avg_price': avg_prices[-1],
            'price_variance': recent_variance,
            'convergence_step': convergence_step,
            'total_steps': len(self.snapshots)
        }
    
    def get_price_statistics(self) -> Dict:
        """
        获取价格统计信息
        
        Returns:
            价格统计字典
        """
        if not self.snapshots:
            return {}
        
        all_prices = []
        for s in self.snapshots:
            all_prices.extend(s.uav_prices.values())
        
        final_prices = list(self.snapshots[-1].uav_prices.values())
        
        return {
            'initial_avg_price': self.snapshots[0].avg_price,
            'final_avg_price': self.snapshots[-1].avg_price,
            'min_price': min(all_prices),
            'max_price': max(all_prices),
            'price_range': max(all_prices) - min(all_prices),
            'final_price_std': np.std(final_prices),
            'price_change_pct': (self.snapshots[-1].avg_price - self.snapshots[0].avg_price) 
                               / self.snapshots[0].avg_price * 100
        }
    
    def reset(self):
        """重置追踪器"""
        self.current_prices = {i: self.initial_price for i in range(self.n_uavs)}
        self.current_utilizations = {i: 0.0 for i in range(self.n_uavs)}
        self.snapshots = []
        self.time_step = 0
        self.total_tasks_processed = 0
        self.total_tasks_successful = 0


class MultiExperimentPriceTracker:
    """
    多实验价格追踪器
    
    用于对比不同配置下的价格变化
    """
    
    def __init__(self):
        self.trackers: Dict[str, PriceTracker] = {}
        self.experiment_configs: Dict[str, Dict] = {}
    
    def add_tracker(self, name: str, n_uavs: int, config: Dict = None):
        """添加一个追踪器"""
        self.trackers[name] = PriceTracker(n_uavs)
        self.experiment_configs[name] = config or {}
    
    def get_tracker(self, name: str) -> Optional[PriceTracker]:
        """获取指定追踪器"""
        return self.trackers.get(name)
    
    def get_all_histories(self) -> Dict[str, Dict]:
        """获取所有追踪器的历史数据"""
        return {name: tracker.get_price_history() 
                for name, tracker in self.trackers.items()}
    
    def get_comparison_data(self) -> Dict:
        """
        获取对比数据（用于多折线图）
        
        Returns:
            {
                'names': [...],
                'final_prices': [...],
                'convergence_steps': [...],
                'price_variances': [...]
            }
        """
        names = list(self.trackers.keys())
        final_prices = []
        convergence_steps = []
        price_variances = []
        
        for name in names:
            metrics = self.trackers[name].get_convergence_metrics()
            final_prices.append(metrics['final_avg_price'])
            convergence_steps.append(metrics['convergence_step'])
            price_variances.append(metrics['price_variance'])
        
        return {
            'names': names,
            'final_prices': final_prices,
            'convergence_steps': convergence_steps,
            'price_variances': price_variances
        }


# ============ 动态定价更新函数 ============

def compute_dynamic_price(current_price: float,
                          utilization: float,
                          target_utilization: float = 0.7,
                          price_sensitivity: float = 0.1,
                          min_price: float = 0.5,
                          max_price: float = 2.0) -> float:
    """
    计算动态更新后的价格
    
    基于利用率调整价格：
    - 利用率高于目标 → 提高价格
    - 利用率低于目标 → 降低价格
    
    Args:
        current_price: 当前价格
        utilization: 当前利用率 (0-1)
        target_utilization: 目标利用率
        price_sensitivity: 价格调整灵敏度
        min_price: 最低价格
        max_price: 最高价格
        
    Returns:
        更新后的价格
    """
    # 计算利用率偏差
    deviation = utilization - target_utilization
    
    # 价格调整
    price_adjustment = price_sensitivity * deviation
    new_price = current_price * (1 + price_adjustment)
    
    # 限制价格范围
    new_price = max(min_price, min(max_price, new_price))
    
    return new_price


def batch_update_prices(prices: Dict[int, float],
                        utilizations: Dict[int, float],
                        target_utilization: float = 0.7,
                        price_sensitivity: float = 0.1) -> Dict[int, float]:
    """
    批量更新所有UAV的价格
    
    Args:
        prices: 当前价格字典
        utilizations: 当前利用率字典
        target_utilization: 目标利用率
        price_sensitivity: 价格灵敏度
        
    Returns:
        更新后的价格字典
    """
    new_prices = {}
    for uav_id, price in prices.items():
        util = utilizations.get(uav_id, 0.0)
        new_prices[uav_id] = compute_dynamic_price(
            price, util, target_utilization, price_sensitivity
        )
    return new_prices


# ============ 测试 ============

def test_price_tracker():
    """测试价格追踪器"""
    print("=" * 60)
    print("测试 价格历史记录模块")
    print("=" * 60)
    
    # 创建追踪器
    n_uavs = 5
    tracker = PriceTracker(n_uavs, initial_price=1.0)
    
    # 模拟价格动态变化
    print("\n[Test 1] 模拟价格动态变化")
    np.random.seed(42)
    
    for step in range(20):
        # 模拟利用率变化
        utilizations = {i: np.random.uniform(0.3, 0.9) for i in range(n_uavs)}
        
        # 更新价格
        new_prices = batch_update_prices(
            tracker.current_prices,
            utilizations,
            target_utilization=0.7,
            price_sensitivity=0.15
        )
        
        # 记录快照
        tracker.record_snapshot(
            prices=new_prices,
            utilizations=utilizations,
            tasks_processed=np.random.randint(5, 15),
            tasks_successful=np.random.randint(3, 12)
        )
    
    print(f"  记录了 {len(tracker.snapshots)} 个快照")
    
    # 获取历史数据
    print("\n[Test 2] 获取价格历史")
    history = tracker.get_price_history()
    print(f"  时间步数: {len(history['time_steps'])}")
    print(f"  UAV数量: {len(history['uav_prices'])}")
    print(f"  平均价格范围: {min(history['avg_price']):.3f} - {max(history['avg_price']):.3f}")
    
    # 获取收敛指标
    print("\n[Test 3] 收敛指标")
    convergence = tracker.get_convergence_metrics()
    print(f"  是否收敛: {convergence['converged']}")
    print(f"  最终平均价格: {convergence['final_avg_price']:.3f}")
    print(f"  价格方差: {convergence['price_variance']:.4f}")
    
    # 获取统计信息
    print("\n[Test 4] 价格统计")
    stats = tracker.get_price_statistics()
    print(f"  初始平均价格: {stats['initial_avg_price']:.3f}")
    print(f"  最终平均价格: {stats['final_avg_price']:.3f}")
    print(f"  价格变化: {stats['price_change_pct']:.1f}%")
    print(f"  价格范围: {stats['min_price']:.3f} - {stats['max_price']:.3f}")
    
    # 测试多实验追踪
    print("\n[Test 5] 多实验追踪")
    multi_tracker = MultiExperimentPriceTracker()
    
    for n_users in [10, 20, 30]:
        name = f"{n_users}用户"
        multi_tracker.add_tracker(name, n_uavs, {'n_users': n_users})
        
        # 模拟该配置下的价格变化
        tracker = multi_tracker.get_tracker(name)
        for step in range(15):
            # 用户数越多，利用率越高
            base_util = 0.3 + 0.4 * (n_users / 50)
            utilizations = {i: np.random.uniform(base_util - 0.1, base_util + 0.2) 
                           for i in range(n_uavs)}
            new_prices = batch_update_prices(tracker.current_prices, utilizations)
            tracker.record_snapshot(prices=new_prices, utilizations=utilizations)
    
    comparison = multi_tracker.get_comparison_data()
    print(f"  实验配置: {comparison['names']}")
    print(f"  最终价格: {[f'{p:.3f}' for p in comparison['final_prices']]}")
    
    print("\n" + "=" * 60)
    print("价格历史记录模块测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_price_tracker()

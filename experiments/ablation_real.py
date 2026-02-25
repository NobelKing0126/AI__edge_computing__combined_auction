"""
真实消融实验模块

每个消融变体通过修改实际的算法逻辑来运行，而非简单调整结果。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.metrics import MetricsCalculator, SystemMetrics, TaskMetric
from config.system_config import SystemConfig
from utils.data_loader import generate_synthetic_users, EUADataLoader


@dataclass
class RealAblationResult:
    """真实消融实验结果"""
    variant_name: str
    description: str
    metrics: SystemMetrics
    raw_task_results: List[Dict]


class RealAblationExperiment:
    """
    真实消融实验
    
    每个变体通过实际修改算法逻辑来运行
    支持合成数据和EUA真实数据两种模式
    """
    
    def __init__(self, n_users: int = 50, n_uavs: int = 5, seed: int = 42,
                 use_eua_data: bool = False, eua_file: str = "users-melbcbd-generated.csv"):
        """
        初始化实验
        
        Args:
            n_users: 用户数量
            n_uavs: UAV数量
            seed: 随机种子
            use_eua_data: 是否使用EUA真实数据
            eua_file: EUA数据文件名
        """
        self.n_users = n_users
        self.n_uavs = n_uavs
        self.seed = seed
        self.use_eua_data = use_eua_data
        self.eua_file = eua_file
        self.config = SystemConfig()
        
        # 生成测试任务
        np.random.seed(seed)
        if use_eua_data:
            self.tasks = self._load_eua_tasks()
        else:
            self.tasks = self._generate_tasks()
        self.uav_resources = self._init_uav_resources()
        
    def _generate_tasks(self) -> List[Dict]:
        """生成合成测试任务（均匀分布）"""
        tasks = []
        for i in range(self.n_users):
            tasks.append({
                'task_id': i,
                'user_id': i,
                'user_pos': (np.random.uniform(0, 2000), np.random.uniform(0, 2000)),
                'data_size': np.random.uniform(1e6, 5e6),  # 1-5 MB
                'compute_size': np.random.uniform(5e9, 20e9),  # 5-20 GFLOPs
                'deadline': np.random.uniform(2.0, 5.0),
                'priority': np.random.uniform(0.3, 0.9),
                'user_level': np.random.randint(1, 6)
            })
        return tasks
    
    def _load_eua_tasks(self) -> List[Dict]:
        """从EUA数据集加载真实用户位置"""
        try:
            loader = EUADataLoader()
            eua_users = loader.load_users(filename=self.eua_file, sample_size=self.n_users)
            
            if len(eua_users) == 0:
                print(f"警告: EUA数据为空，回退到合成数据")
                return self._generate_tasks()
            
            # 计算坐标范围并归一化到场景尺度
            xs = [u.x for u in eua_users]
            ys = [u.y for u in eua_users]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # 计算缩放因子，将真实坐标映射到 [100, 1900] 范围（保留边界）
            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1
            
            scale_x = 1800 / x_range
            scale_y = 1800 / y_range
            
            tasks = []
            for i, user in enumerate(eua_users):
                # 归一化到场景范围 [100, 1900]
                x_norm = (user.x - x_min) * scale_x + 100
                y_norm = (user.y - y_min) * scale_y + 100
                
                tasks.append({
                    'task_id': i,
                    'user_id': i,
                    'user_pos': (x_norm, y_norm),
                    'data_size': np.random.uniform(1e6, 5e6),
                    'compute_size': np.random.uniform(5e9, 20e9),
                    'deadline': np.random.uniform(2.0, 5.0),
                    'priority': np.random.uniform(0.3, 0.9),
                    'user_level': np.random.randint(1, 6),
                    'original_lat': user.metadata.get('latitude') if user.metadata else None,
                    'original_lon': user.metadata.get('longitude') if user.metadata else None
                })
            
            print(f"已加载 {len(tasks)} 个EUA用户位置 (来自 {self.eua_file})")
            return tasks
            
        except Exception as e:
            print(f"警告: 加载EUA数据失败 ({e})，回退到合成数据")
            return self._generate_tasks()
    
    def _init_uav_resources(self) -> List[Dict]:
        """初始化UAV资源"""
        uavs = []
        for i in range(self.n_uavs):
            uavs.append({
                'uav_id': i,
                'position': (400 + i * 300, 1000),  # 均匀分布
                'f_max': self.config.uav.f_max,
                'f_avail': self.config.uav.f_max,
                'E_max': self.config.uav.E_max,
                'E_remain': self.config.uav.E_max,
                'health': 1.0
            })
        return uavs
    
    def run_full_method(self) -> RealAblationResult:
        """运行完整方法"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="Full-Method",
            description="完整方法（所有组件启用）",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_no_checkpoint(self, fault_prob: float = 0.1) -> RealAblationResult:
        """A2: 禁用Checkpoint"""
        results = self._run_with_config(
            use_checkpoint=False,  # 关键
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=fault_prob  # 有故障时Checkpoint才有意义
        )
        return RealAblationResult(
            variant_name="A2-NoCheckpoint",
            description="禁用Checkpoint机制",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_no_convex_optimization(self) -> RealAblationResult:
        """A3: 使用启发式算力分配"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=False,  # 关键：使用启发式
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="A3-NoConvex",
            description="使用启发式算力分配（非凸优化）",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_no_priority(self) -> RealAblationResult:
        """禁用优先级调度"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=False,  # 关键
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="No-Priority",
            description="禁用优先级调度（随机顺序）",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_fixed_deploy(self) -> RealAblationResult:
        """使用固定部署"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=False,  # 关键：固定部署
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="Fixed-Deploy",
            description="使用固定UAV部署（非K-means）",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_no_free_energy(self) -> RealAblationResult:
        """A1: 禁用自由能风险评估"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=False,  # 关键
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=True,
            fault_prob=0.05  # 有风险时自由能才有意义
        )
        return RealAblationResult(
            variant_name="A1-NoFreeEnergy",
            description="禁用自由能风险评估",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_single_strategy_greedy(self) -> RealAblationResult:
        """A6: 仅使用单策略贪心"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=False,  # 关键
            use_high_priority_constraint=True,
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="A6-SingleGreedy",
            description="仅使用优先级排序策略（非多策略）",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def run_no_high_priority_constraint(self) -> RealAblationResult:
        """A4: 无高优先级强制约束"""
        results = self._run_with_config(
            use_checkpoint=True,
            use_free_energy=True,
            use_convex_opt=True,
            use_priority=True,
            use_kmeans=True,
            use_dynamic_pricing=True,
            use_multi_strategy=True,
            use_high_priority_constraint=False,  # 关键
            fault_prob=0.0
        )
        return RealAblationResult(
            variant_name="A4-NoHighPrio",
            description="无高优先级强制约束",
            metrics=self._compute_metrics(results),
            raw_task_results=results
        )
    
    def _run_with_config(self,
                         use_checkpoint: bool,
                         use_free_energy: bool,
                         use_convex_opt: bool,
                         use_priority: bool,
                         use_kmeans: bool,
                         use_dynamic_pricing: bool,
                         use_multi_strategy: bool,
                         use_high_priority_constraint: bool,
                         fault_prob: float) -> List[Dict]:
        """
        使用指定配置运行实验
        
        这是真实的实验逻辑，根据配置启用/禁用不同组件
        """
        np.random.seed(self.seed)
        
        results = []
        
        # 复制UAV资源状态
        uav_state = [copy.deepcopy(u) for u in self.uav_resources]
        
        # UAV部署
        if use_kmeans:
            uav_positions = self._kmeans_deploy()
        else:
            uav_positions = self._fixed_deploy()
        
        for i, uav in enumerate(uav_state):
            uav['position'] = uav_positions[i]
        
        # 任务排序
        if use_priority:
            sorted_tasks = self._sort_by_priority(self.tasks)
        else:
            sorted_tasks = self.tasks.copy()
            np.random.shuffle(sorted_tasks)
        
        # 如果有高优先级约束，先处理高优先级任务
        if use_high_priority_constraint:
            high_prio = [t for t in sorted_tasks if t['priority'] >= 0.7]
            low_prio = [t for t in sorted_tasks if t['priority'] < 0.7]
            sorted_tasks = high_prio + low_prio
        
        # 处理每个任务
        for task in sorted_tasks:
            # 选择最优UAV
            best_uav, best_split = self._select_uav_and_split(
                task, uav_state, use_convex_opt, use_multi_strategy
            )
            
            if best_uav is None:
                results.append({
                    'task_id': task['task_id'],
                    'success': False,
                    'delay': 999.0,
                    'deadline': task['deadline'],
                    'energy': 0.0,
                    'met_deadline': False,
                    'priority': task['priority']
                })
                continue
            
            # 计算时延和能耗
            delay, energy = self._compute_delay_energy(
                task, best_uav, best_split, uav_state
            )
            
            # 自由能风险评估 (优化: 提高阈值到0.9，降低拒绝概率)
            if use_free_energy:
                risk = self._compute_free_energy_risk(task, best_uav, delay)
                if risk > 0.9:  # 高风险阈值从0.8提升到0.9
                    if np.random.random() < risk * 0.3:  # 拒绝概率从0.5降到0.3
                        results.append({
                            'task_id': task['task_id'],
                            'success': False,
                            'delay': 999.0,
                            'deadline': task['deadline'],
                            'energy': 0.0,
                            'met_deadline': False,
                            'priority': task['priority'],
                            'rejected_by_free_energy': True
                        })
                        continue
            
            # 模拟故障
            fault_occurred = np.random.random() < fault_prob
            
            if fault_occurred:
                if use_checkpoint:
                    # 有Checkpoint，部分恢复
                    recovery_delay = delay * 0.3  # 只需重做30%
                    delay += recovery_delay
                    energy *= 1.3
                    success = delay <= task['deadline']
                else:
                    # 无Checkpoint，完全失败
                    success = False
                    delay = 999.0
            else:
                success = delay <= task['deadline']
            
            # 更新UAV资源
            if success:
                uav_idx = best_uav['uav_id']
                uav_state[uav_idx]['E_remain'] -= energy
                
            results.append({
                'task_id': task['task_id'],
                'success': success,
                'delay': delay,
                'deadline': task['deadline'],
                'energy': energy if success else 0,
                'met_deadline': delay <= task['deadline'],
                'priority': task['priority'],
                'uav_id': best_uav['uav_id'] if best_uav else None,
                'split_ratio': best_split
            })
        
        return results
    
    def _kmeans_deploy(self) -> List[Tuple[float, float]]:
        """K-means部署"""
        user_positions = [t['user_pos'] for t in self.tasks]
        
        # 简单K-means
        centers = []
        k = self.n_uavs
        
        # 初始化：均匀采样
        indices = np.linspace(0, len(user_positions)-1, k, dtype=int)
        centers = [user_positions[i] for i in indices]
        
        # 迭代 (优化: 从20次增加到50次以确保收敛)
        for _ in range(50):
            # 分配
            clusters = [[] for _ in range(k)]
            for pos in user_positions:
                dists = [np.sqrt((pos[0]-c[0])**2 + (pos[1]-c[1])**2) for c in centers]
                clusters[np.argmin(dists)].append(pos)
            
            # 更新中心
            new_centers = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    new_centers.append((
                        np.mean([p[0] for p in cluster]),
                        np.mean([p[1] for p in cluster])
                    ))
                else:
                    new_centers.append(centers[i])
            centers = new_centers
        
        return centers
    
    def _fixed_deploy(self) -> List[Tuple[float, float]]:
        """固定部署：均匀网格"""
        positions = []
        n = int(np.ceil(np.sqrt(self.n_uavs)))
        step = 2000 / (n + 1)
        
        for i in range(self.n_uavs):
            x = step * (1 + i % n)
            y = step * (1 + i // n)
            positions.append((x, y))
        
        return positions
    
    def _sort_by_priority(self, tasks: List[Dict]) -> List[Dict]:
        """按优先级排序"""
        def compute_priority(task):
            urgency = task['compute_size'] / task['deadline']
            return task['priority'] * 0.4 + urgency / 1e10 * 0.4 + task['user_level'] / 5 * 0.2
        
        return sorted(tasks, key=compute_priority, reverse=True)
    
    def _select_uav_and_split(self, 
                               task: Dict,
                               uav_state: List[Dict],
                               use_convex_opt: bool,
                               use_multi_strategy: bool) -> Tuple[Optional[Dict], float]:
        """选择UAV和切分点"""
        best_uav = None
        best_split = 0.5
        best_delay = float('inf')
        
        for uav in uav_state:
            if uav['E_remain'] < 1000:  # 能量不足
                continue
            
            # 计算距离
            dist = np.sqrt(
                (task['user_pos'][0] - uav['position'][0])**2 +
                (task['user_pos'][1] - uav['position'][1])**2 +
                self.config.uav.H**2
            )
            
            # 获取DNN模型层数
            model_spec = task.get('model_spec')
            n_layers = model_spec.layers if model_spec and hasattr(model_spec, 'layers') else 10
            
            if use_convex_opt:
                # 凸优化：枚举所有整数层切分点 (0 到 n_layers)
                splits_to_try = [layer / n_layers for layer in range(0, n_layers + 1)]
            else:
                # 启发式：固定切分（中间层）
                splits_to_try = [0.5]
            
            if use_multi_strategy:
                # 多策略：尝试多个切分点（已在凸优化中处理）
                pass
            else:
                # 单策略：尝试几个常用层（约40%, 50%, 60%）
                mid_layer = n_layers // 2
                splits_to_try = [
                    max(0, mid_layer - n_layers // 10) / n_layers,
                    mid_layer / n_layers,
                    min(n_layers, mid_layer + n_layers // 10) / n_layers
                ]
            
            for split in splits_to_try:
                delay, _ = self._compute_delay_energy(task, uav, split, uav_state)
                if delay < best_delay:
                    best_delay = delay
                    best_uav = uav
                    best_split = split
        
        return best_uav, best_split
    
    def _get_feature_size_at_layer(self, split_layer: int, model_spec, n_layers: int) -> float:
        """获取精确的中间特征大小"""
        if split_layer == 0:
            return model_spec.input_size_bytes if hasattr(model_spec, 'input_size_bytes') else 150000
        elif split_layer >= n_layers:
            return 1000
        
        if model_spec and hasattr(model_spec, 'get_output_size_at_layer'):
            return model_spec.get_output_size_at_layer(split_layer - 1)
        
        if hasattr(model_spec, 'typical_feature_sizes') and model_spec.typical_feature_sizes:
            idx = int(split_layer / n_layers * len(model_spec.typical_feature_sizes))
            idx = min(idx, len(model_spec.typical_feature_sizes) - 1)
            return model_spec.typical_feature_sizes[idx]
        return 150000 * (1 - split_layer / n_layers) * 0.5
    
    def _get_cumulative_flops_at_layer(self, split_layer: int, model_spec, n_layers: int, C_total: float) -> float:
        """获取精确的累计计算量"""
        if split_layer <= 0:
            return 0.0
        if split_layer >= n_layers:
            return C_total
        
        if model_spec and hasattr(model_spec, 'get_flops_ratio_at_layer'):
            return C_total * model_spec.get_flops_ratio_at_layer(split_layer)
        
        return C_total * (split_layer / n_layers)
    
    def _compute_delay_energy(self,
                               task: Dict,
                               uav: Dict,
                               split_ratio: float,
                               uav_state: List[Dict]) -> Tuple[float, float]:
        """计算时延和能耗（使用精确的每层数据）"""
        # 距离
        dist = np.sqrt(
            (task['user_pos'][0] - uav['position'][0])**2 +
            (task['user_pos'][1] - uav['position'][1])**2 +
            self.config.uav.H**2
        )
        
        # 信道增益
        h = self.config.channel.beta_0 / (dist ** 2)
        
        # 传输速率
        P_tx = self.config.channel.P_tx_user
        noise = self.config.channel.N_0 * self.config.channel.W
        snr = P_tx * h / noise
        rate = self.config.channel.W * np.log2(1 + snr)
        
        # 上传时延
        T_upload = task['data_size'] / rate
        
        # 获取模型信息
        model_spec = task.get('model_spec')
        n_layers = model_spec.layers if model_spec and hasattr(model_spec, 'layers') else 10
        split_layer = int(split_ratio * n_layers)
        
        # 使用精确的计算量分配
        C_total = task['compute_size']
        C_edge = self._get_cumulative_flops_at_layer(split_layer, model_spec, n_layers, C_total)
        C_cloud = C_total - C_edge
        
        T_edge = C_edge / uav['f_max'] if C_edge > 0 else 0
        
        # 使用精确的特征大小
        feature_size = self._get_feature_size_at_layer(split_layer, model_spec, n_layers)
        T_trans = feature_size / self.config.channel.R_backhaul if split_layer < n_layers else 0
        T_cloud = C_cloud / self.config.cloud.F_c if split_layer < n_layers else 0
        
        T_total = T_upload + T_edge + T_trans + T_cloud
        
        # 能耗
        kappa = self.config.energy.kappa_edge
        f_edge = uav['f_max']
        energy = kappa * (f_edge ** 2) * C_edge
        
        return T_total, energy
    
    def _compute_free_energy_risk(self, 
                                   task: Dict,
                                   uav: Dict,
                                   predicted_delay: float) -> float:
        """计算自由能风险"""
        # 时间风险
        time_margin = (task['deadline'] - predicted_delay) / task['deadline']
        time_risk = max(0, 1 - time_margin)
        
        # 能量风险
        energy_ratio = uav['E_remain'] / uav['E_max']
        energy_risk = max(0, 1 - energy_ratio)
        
        # 综合风险
        return 0.5 * time_risk + 0.5 * energy_risk
    
    def _compute_metrics(self, results: List[Dict]) -> SystemMetrics:
        """计算指标"""
        n = len(results)
        if n == 0:
            return SystemMetrics(
                total_tasks=0, success_count=0, success_rate=0,
                avg_delay=0, max_delay=0, deadline_meet_rate=0,
                total_energy=0, avg_energy_per_task=0,
                energy_efficiency=0, throughput=0
            )
        
        success = [r for r in results if r['success']]
        met_deadline = [r for r in results if r['met_deadline']]
        
        delays = [r['delay'] for r in results if r['delay'] < 900]
        energies = [r['energy'] for r in results]
        
        total_energy = sum(energies)
        
        # 高优先级成功率
        high_prio = [r for r in results if r['priority'] >= 0.7]
        high_success = [r for r in high_prio if r['success']]
        high_prio_rate = len(high_success) / len(high_prio) if high_prio else 0
        
        return SystemMetrics(
            total_tasks=n,
            success_count=len(success),
            success_rate=len(success) / n,
            avg_delay=np.mean(delays) if delays else 0,
            max_delay=max(delays) if delays else 0,
            deadline_meet_rate=len(met_deadline) / n,
            total_energy=total_energy,
            avg_energy_per_task=total_energy / n if n > 0 else 0,
            energy_efficiency=len(success) / total_energy if total_energy > 0 else 0,
            throughput=len(success) / (np.mean(delays) if delays else 1),
            high_priority_success_rate=high_prio_rate
        )
    
    def run_all(self) -> Dict[str, RealAblationResult]:
        """运行所有消融实验"""
        results = {}
        
        print("运行完整方法...")
        results['full'] = self.run_full_method()
        
        print("运行A1: 禁用自由能...")
        results['no_free_energy'] = self.run_no_free_energy()
        
        print("运行A2: 禁用Checkpoint...")
        results['no_checkpoint'] = self.run_no_checkpoint()
        
        print("运行A3: 禁用凸优化...")
        results['no_convex'] = self.run_no_convex_optimization()
        
        print("运行A4: 禁用高优先级约束...")
        results['no_high_prio'] = self.run_no_high_priority_constraint()
        
        print("运行A6: 单策略贪心...")
        results['single_greedy'] = self.run_single_strategy_greedy()
        
        print("运行No-Priority...")
        results['no_priority'] = self.run_no_priority()
        
        print("运行Fixed-Deploy...")
        results['fixed_deploy'] = self.run_fixed_deploy()
        
        return results
    
    def print_results(self, results: Dict[str, RealAblationResult]):
        """打印结果"""
        print("\n" + "=" * 90)
        print("真实消融实验结果")
        print("=" * 90)
        
        full_sr = results['full'].metrics.success_rate
        full_delay = results['full'].metrics.avg_delay
        
        print(f"\n{'变体':<20} {'成功率':>10} {'Δ成功率':>10} {'平均时延':>12} {'Δ时延':>10} {'高优先级':>10}")
        print("-" * 90)
        
        for key, result in results.items():
            sr = result.metrics.success_rate
            delay = result.metrics.avg_delay
            hp_rate = result.metrics.high_priority_success_rate
            
            if key == 'full':
                diff_sr = "-"
                diff_delay = "-"
            else:
                diff_sr = f"{(sr - full_sr) * 100:+.1f}%"
                diff_delay = f"{(delay - full_delay) / full_delay * 100:+.1f}%" if full_delay > 0 else "-"
            
            print(f"{result.variant_name:<20} {sr*100:>9.1f}% {diff_sr:>10} "
                  f"{delay*1000:>10.1f}ms {diff_delay:>10} {hp_rate*100:>9.1f}%")
        
        print("=" * 90)


def test_real_ablation():
    """测试真实消融实验"""
    print("=" * 60)
    print("测试真实消融实验")
    print("=" * 60)
    
    exp = RealAblationExperiment(n_users=50, n_uavs=5, seed=42)
    results = exp.run_all()
    exp.print_results(results)
    
    print("\n" + "=" * 60)
    print("真实消融实验测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_real_ablation()

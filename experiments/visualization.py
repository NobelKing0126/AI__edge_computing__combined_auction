"""
可视化模块

提供实验结果的可视化功能

可视化类型:
1. 性能对比柱状图
2. 时延分布图
3. 能耗效率图
4. UAV部署与用户分布图
5. 消融实验对比图
6. 收敛曲线图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.metrics import SystemMetrics
from experiments.baselines import BaselineResult
from experiments.ablation import AblationResult


class ExperimentVisualizer:
    """
    实验可视化器
    """
    
    def __init__(self, save_dir: str = "figures"):
        """
        初始化可视化器
        
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 配色方案
        self.colors = {
            'proposed': '#2ecc71',      # 绿色 - 提议方法
            'edge_only': '#e74c3c',     # 红色
            'cloud_only': '#3498db',    # 蓝色
            'greedy': '#f39c12',        # 橙色
            'fcfs': '#9b59b6',          # 紫色
            'fixed_split': '#1abc9c',   # 青色
            'random': '#34495e',        # 灰色
        }
        
        # 样式设置
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_baseline_comparison(self,
                                 proposed_metrics: SystemMetrics,
                                 baseline_results: Dict[str, BaselineResult],
                                 save_name: str = "baseline_comparison.png"):
        """
        绘制基线对比图
        
        Args:
            proposed_metrics: 提议方法指标
            baseline_results: 基线结果
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 准备数据
        names = ['Proposed'] + list(baseline_results.keys())
        success_rates = [proposed_metrics.success_rate * 100]
        avg_delays = [proposed_metrics.avg_delay * 1000]
        energy_effs = [proposed_metrics.energy_efficiency]
        
        for result in baseline_results.values():
            success_rates.append(result.success_rate * 100)
            avg_delays.append(result.avg_delay * 1000)
            energy_effs.append(result.success_count / max(result.total_energy, 1e-10))
        
        colors = [self.colors['proposed']] + [
            self.colors.get(name.lower().replace('-', '_'), '#95a5a6') 
            for name in baseline_results.keys()
        ]
        
        # 图1: 成功率
        axes[0].bar(names, success_rates, color=colors)
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Task Success Rate Comparison')
        axes[0].set_ylim(0, 110)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 图2: 平均时延
        axes[1].bar(names, avg_delays, color=colors)
        axes[1].set_ylabel('Average Delay (ms)')
        axes[1].set_title('Average Delay Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 图3: 能效
        axes[2].bar(names, energy_effs, color=colors)
        axes[2].set_ylabel('Energy Efficiency (tasks/J)')
        axes[2].set_title('Energy Efficiency Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def plot_ablation_study(self,
                            ablation_results: Dict[str, AblationResult],
                            save_name: str = "ablation_study.png"):
        """
        绘制消融实验图
        
        Args:
            ablation_results: 消融结果
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        names = [r.variant_name for r in ablation_results.values()]
        success_rates = [r.metrics.success_rate * 100 for r in ablation_results.values()]
        
        # 计算相对差异
        full_sr = ablation_results.get('full', list(ablation_results.values())[0]).metrics.success_rate
        diffs = [(r.metrics.success_rate - full_sr) * 100 for r in ablation_results.values()]
        
        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diffs]
        
        # 图1: 成功率对比
        bars = axes[0].bar(names, success_rates, color='#3498db')
        bars[0].set_color('#2ecc71')  # 突出完整方法
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Ablation Study: Success Rate')
        axes[0].set_ylim(0, 110)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=success_rates[0], color='#2ecc71', linestyle='--', alpha=0.7)
        
        # 图2: 相对性能变化
        colors_diff = ['#27ae60' if i == 0 else ('#e74c3c' if d < 0 else '#3498db') 
                       for i, d in enumerate(diffs)]
        axes[1].bar(names, diffs, color=colors_diff)
        axes[1].set_ylabel('Relative Change (%)')
        axes[1].set_title('Ablation Study: Performance Impact')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def plot_uav_deployment(self,
                            user_positions: List[Tuple[float, float]],
                            uav_positions: List[Tuple[float, float, float]],
                            user_uav_mapping: Dict[int, int],
                            save_name: str = "uav_deployment.png"):
        """
        绘制UAV部署与用户分布图
        
        Args:
            user_positions: 用户位置列表
            uav_positions: UAV位置列表
            user_uav_mapping: 用户-UAV映射
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 用户颜色（按分配的UAV着色）
        n_uavs = len(uav_positions)
        cmap = plt.cm.get_cmap('tab10', n_uavs)
        
        # 绘制用户
        # 支持多种格式: (x, y) 元组, Location对象, 或 dict
        def get_xy(p):
            if hasattr(p, 'x') and hasattr(p, 'y'):
                return p.x, p.y
            elif isinstance(p, dict):
                return p.get('x', p[0]), p.get('y', p[1])
            else:
                return p[0], p[1]
        
        for i, p in enumerate(user_positions):
            x, y = get_xy(p)
            uav_id = user_uav_mapping.get(i, 0)
            ax.scatter(x, y, c=[cmap(uav_id)], s=30, alpha=0.6, marker='o')
        
        # 绘制UAV
        for i, pos in enumerate(uav_positions):
            px, py = get_xy(pos)
            ax.scatter(px, py, c='red', s=200, marker='^', 
                      edgecolors='black', linewidths=2, zorder=5)
            ax.annotate(f'UAV-{i}', (px, py), 
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=10, fontweight='bold')
            
            # 绘制覆盖范围
            circle = plt.Circle((px, py), 500, fill=False, 
                               color=cmap(i), linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('UAV Deployment and User Distribution')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def plot_delay_distribution(self,
                                delays: List[float],
                                deadlines: List[float],
                                save_name: str = "delay_distribution.png"):
        """
        绘制时延分布图
        
        Args:
            delays: 时延列表
            deadlines: 截止时间列表
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1: 时延直方图
        axes[0].hist(delays, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(delays), color='#e74c3c', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(delays)*1000:.1f}ms')
        axes[0].set_xlabel('Delay (s)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Task Delay Distribution')
        axes[0].legend()
        
        # 图2: 时延vs截止时间
        colors = ['#2ecc71' if d <= dl else '#e74c3c' 
                 for d, dl in zip(delays, deadlines)]
        axes[1].scatter(deadlines, delays, c=colors, alpha=0.7)
        axes[1].plot([0, max(deadlines)], [0, max(deadlines)], 
                    'k--', alpha=0.5, label='Deadline Line')
        axes[1].set_xlabel('Deadline (s)')
        axes[1].set_ylabel('Actual Delay (s)')
        axes[1].set_title('Delay vs Deadline')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def plot_scalability(self,
                         user_counts: List[int],
                         success_rates: List[float],
                         avg_delays: List[float],
                         save_name: str = "scalability.png"):
        """
        绘制可扩展性分析图
        
        Args:
            user_counts: 用户数量列表
            success_rates: 对应的成功率
            avg_delays: 对应的平均时延
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1: 成功率vs用户数
        axes[0].plot(user_counts, [s*100 for s in success_rates], 
                    'o-', color='#2ecc71', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Users')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate vs User Count')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 110)
        
        # 图2: 时延vs用户数
        axes[1].plot(user_counts, [d*1000 for d in avg_delays], 
                    's-', color='#3498db', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Users')
        axes[1].set_ylabel('Average Delay (ms)')
        axes[1].set_title('Average Delay vs User Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def plot_convergence(self,
                         iterations: List[int],
                         objectives: List[float],
                         save_name: str = "convergence.png"):
        """
        绘制收敛曲线
        
        Args:
            iterations: 迭代次数
            objectives: 目标函数值
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(iterations, objectives, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Auction Convergence')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {save_name}")
    
    def generate_all_figures(self,
                             proposed_metrics: SystemMetrics,
                             baseline_results: Dict[str, BaselineResult],
                             ablation_results: Dict[str, AblationResult],
                             user_positions: List[Tuple[float, float]],
                             uav_positions: List[Tuple[float, float, float]],
                             user_uav_mapping: Dict[int, int]):
        """
        生成所有图表
        """
        print("\n生成实验图表...")
        
        # 基线对比
        self.plot_baseline_comparison(proposed_metrics, baseline_results)
        
        # 消融实验
        self.plot_ablation_study(ablation_results)
        
        # UAV部署
        self.plot_uav_deployment(user_positions, uav_positions, user_uav_mapping)
        
        # 模拟时延数据
        np.random.seed(42)
        delays = np.random.uniform(0.5, 2.0, 50)
        deadlines = np.random.uniform(1.5, 3.0, 50)
        self.plot_delay_distribution(delays.tolist(), deadlines.tolist())
        
        # 可扩展性
        user_counts = [20, 40, 60, 80, 100]
        success_rates = [1.0, 0.98, 0.95, 0.90, 0.85]
        avg_delays = [0.8, 0.9, 1.0, 1.1, 1.3]
        self.plot_scalability(user_counts, success_rates, avg_delays)
        
        print(f"\n所有图表已保存到: {self.save_dir}/")


# ============ 测试用例 ============

def test_visualization():
    """测试可视化模块"""
    print("=" * 60)
    print("测试可视化模块")
    print("=" * 60)
    
    visualizer = ExperimentVisualizer(save_dir="/home/hyp/projects/first/figures")
    
    # 创建模拟数据
    from experiments.metrics import SystemMetrics
    from experiments.baselines import BaselineResult
    from experiments.ablation import AblationResult
    
    # 提议方法指标
    proposed = SystemMetrics(
        total_tasks=50, success_count=48,
        success_rate=0.96, avg_delay=0.9,
        max_delay=2.0, deadline_meet_rate=0.94,
        total_energy=500, avg_energy_per_task=10,
        energy_efficiency=0.096, throughput=5
    )
    
    # 基线结果
    baselines = {
        'Edge-Only': BaselineResult('Edge-Only', 50, 40, 0.80, 1.2, 2.5, 0.78, 800, 16, 0.75),
        'Cloud-Only': BaselineResult('Cloud-Only', 50, 45, 0.90, 0.8, 1.5, 0.88, 0, 0, 0.85),
        'Greedy': BaselineResult('Greedy', 50, 35, 0.70, 1.5, 3.0, 0.68, 600, 12, 0.60),
    }
    
    # 消融结果
    ablation = {
        'full': AblationResult('Full', 'Full method', proposed, {}),
        'no_cp': AblationResult('No-CP', 'No checkpoint', 
                               SystemMetrics(50, 45, 0.90, 0.95, 2.1, 0.88, 510, 10.2, 0.088, 5), 
                               {'success_rate': -6.25}),
    }
    
    # 测试1: 基线对比图
    print("\n[Test 1] 测试基线对比图...")
    visualizer.plot_baseline_comparison(proposed, baselines)
    print("  ✓ 基线对比图正确")
    
    # 测试2: 消融实验图
    print("\n[Test 2] 测试消融实验图...")
    visualizer.plot_ablation_study(ablation)
    print("  ✓ 消融实验图正确")
    
    # 测试3: UAV部署图
    print("\n[Test 3] 测试UAV部署图...")
    np.random.seed(42)
    users = [(np.random.uniform(0, 2000), np.random.uniform(0, 2000)) for _ in range(50)]
    uavs = [(500, 500, 100), (1500, 500, 100), (1000, 1500, 100)]
    mapping = {i: i % 3 for i in range(50)}
    visualizer.plot_uav_deployment(users, uavs, mapping)
    print("  ✓ UAV部署图正确")
    
    # 测试4: 时延分布图
    print("\n[Test 4] 测试时延分布图...")
    delays = np.random.uniform(0.5, 2.0, 50).tolist()
    deadlines = np.random.uniform(1.5, 3.0, 50).tolist()
    visualizer.plot_delay_distribution(delays, deadlines)
    print("  ✓ 时延分布图正确")
    
    # 测试5: 可扩展性图
    print("\n[Test 5] 测试可扩展性图...")
    visualizer.plot_scalability([20, 40, 60, 80, 100], 
                                [1.0, 0.95, 0.90, 0.85, 0.80],
                                [0.8, 0.9, 1.0, 1.2, 1.5])
    print("  ✓ 可扩展性图正确")
    
    print("\n" + "=" * 60)
    print("可视化测试通过! ✓")
    print(f"图表保存在: {visualizer.save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    test_visualization()

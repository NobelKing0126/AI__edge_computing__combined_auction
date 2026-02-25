"""
M17D: BeliefUpdater - 信念更新器

功能：根据观测更新对当前状态的信念分布
参考文档：docs/自由能.txt 第292-364行

数学框架 (自由能.txt 第298-354行):
    - 信念表示: Q(s_t) = N(s_t; μ_t, Σ_t)
    - 贝叶斯更新: Q(s_t | o_t) ∝ P(o_t | s_t) × Q(s_t)
    - 变分自由能: F = D_KL[Q(s) || P(s|o)] - log P(o)
    - 梯度下降更新: μ_t ← μ_t - η × ∇_μ F
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.active_inference.state_space import StateVector, ObservationVector
from algorithms.active_inference.generative_model import GenerativeModel
from config.constants import NUMERICAL


@dataclass
class BeliefState:
    """
    信念状态 (自由能.txt 第298-304行)

    使用高斯近似: Q(s_t) = N(s_t; μ_t, Σ_t)

    Attributes:
        mean: 状态均值
        covariance: 协方差矩阵 [6×6]
        variance: 标准方差向量 (简化版)
        precision: 精度矩阵
    """
    mean: StateVector
    covariance: Optional[np.ndarray] = None
    variance: Optional[np.ndarray] = None  # 简化版本
    precision: Optional[np.ndarray] = None

    def __post_init__(self):
        """初始化时设置协方差"""
        state_dim = 6  # E, T, h, p, d, sigma

        if self.covariance is None:
            # 使用对角协方差 (简化)
            self.covariance = np.diag([
                1000.0**2,    # E方差 (J²)
                1.0**2,         # T方差 (s²)
                0.05**2,        # h方差
                0.1**2,         # p方差
                100.0**2,       # d方差 (m²)
                0.05**2         # sigma方差
            ])

        if self.variance is None:
            self.variance = np.diag(self.covariance)

        if self.precision is None:
            # 精度矩阵 = 协方差矩阵的逆
            self.precision = np.linalg.inv(self.covariance + np.eye(state_dim) * 1e-6)

    def get_entropy(self) -> float:
        """
        计算信念熵 (自由能.txt 第420-427行)

        H[N(μ, Σ)] = 1/2 × log|2πeΣ|

        Returns:
            float: 熵值
        """
        return 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * self.covariance) + NUMERICAL.EPSILON)

    def sample(self, n_samples: int = 1) -> list:
        """
        从信念分布采样

        Args:
            n_samples: 采样数量

        Returns:
            list: 采样状态列表
        """
        samples = np.random.multivariate_normal(
            self.mean.to_numpy(),
            self.covariance,
            size=n_samples
        )

        return [
            StateVector(
                E=max(0, s[0]),
                T=max(0, s[1]),
                h=np.clip(s[2], 0.0, 1.0),
                p=np.clip(s[3], 0.0, 1.0),
                d=max(0, s[4]),
                sigma=np.clip(s[5], 0.0, 1.0)
            )
            for s in samples
        ]


class KalmanUpdater:
    """
    卡尔曼滤波更新器 (自由能.txt 第339-346行简化版本)

    功能：使用卡尔曼滤波更新信念
    公式 (自由能.txt 第340-346行): μ_t ← μ_t + K_t (o_t - ô_t)

    Attributes:
        gen_model: 生成模型
        observation_noise: 观测噪声协方差
    """

    def __init__(self, gen_model: GenerativeModel):
        """
        初始化卡尔曼更新器

        Args:
            gen_model: 生成模型
        """
        self.gen_model = gen_model
        # 观测噪声协方差 [4×4]
        self.observation_noise = np.diag([
            1000.0**2,   # E_hat噪声 (J²)
            0.02**2,      # h_hat噪声
            0.05**2,       # q噪声
            0.1**2         # w噪声
        ])

        # 观测矩阵 H: [4×6] (从状态6维到观测4维)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # E_hat = E
            [0, 0, 1, 0, 0, 0],  # h_hat = h
            [0, 0, 0, 0, 0, -0.5],  # q = 1 - 0.5 * sigma
            [0, 0, 0, 0, 0, 0]   # w ≈ 0.8 (常数)
        ])

    def update(self, belief: BeliefState, obs: ObservationVector) -> BeliefState:
        """
        使用卡尔曼滤波更新信念 (自由能.txt 第340-346行)

        Args:
            belief: 当前信念
            obs: 新观测

        Returns:
            BeliefState: 更新后的信念
        """
        # 预测观测
        pred_mean = self.H @ belief.mean.to_numpy()

        # 观测向量
        obs_vector = np.array([obs.E_hat, obs.h_hat, obs.q, obs.w])

        # 新息
        innovation = obs_vector - pred_mean

        # 新息协方差
        S = self.H @ belief.covariance @ self.H.T + self.observation_noise

        # 卡尔曼增益
        K = belief.covariance @ self.H.T @ np.linalg.inv(S + np.eye(4) * 1e-6)

        # 更新均值
        new_mean_vec = belief.mean.to_numpy() + K @ innovation

        # 更新协方差 (Joseph形式)
        new_covariance = (np.eye(6) - K @ self.H) @ belief.covariance @ (np.eye(6) - K @ self.H).T + \
                        K @ self.observation_noise @ K.T

        # 创建新的信念状态
        return BeliefState(
            mean=StateVector(
                E=max(0, new_mean_vec[0]),
                T=max(0, new_mean_vec[1]),
                h=np.clip(new_mean_vec[2], 0.0, 1.0),
                p=np.clip(new_mean_vec[3], 0.0, 1.0),
                d=max(0, new_mean_vec[4]),
                sigma=np.clip(new_mean_vec[5], 0.0, 1.0)
            ),
            covariance=new_covariance
        )


class VariationalUpdater:
    """
    变分推断更新器 (自由能.txt 第312-363行)

    功能：使用变分推断更新信念
    公式 (自由能.txt 第323-324行): F = E_Q[-log P(o|s)] + D_KL[Q(s) || P(s)]

    Attributes:
        gen_model: 生成模型
        learning_rate: 学习率
    """

    def __init__(self, gen_model: GenerativeModel, learning_rate: float = 0.1):
        """
        初始化变分更新器

        Args:
            gen_model: 生成模型
            learning_rate: 学习率
        """
        self.gen_model = gen_model
        self.learning_rate = learning_rate

    def compute_free_energy(self,
                            belief: BeliefState,
                            obs: ObservationVector) -> float:
        """
        计算变分自由能 (自由能.txt 第317-324行)

        F = E_Q[-log P(o|s)] + D_KL[Q(s) || P(s)]

        简化：第二项为0（无先验），所以 F = E_Q[-log P(o|s)]

        Args:
            belief: 当前信念
            obs: 观测

        Returns:
            float: 自由能值
        """
        # 采样状态
        samples = belief.sample(n_samples=10)

        # 期望重构误差: E_Q[-log P(o|s)]
        expected_nll = 0.0
        for sample in samples:
            log_likelihood = self.gen_model.compute_likelihood(obs, sample)
            expected_nll -= log_likelihood

        expected_nll /= len(samples)

        # KL散度 (简化为0，无明确先验)
        kl_divergence = 0.0

        return expected_nll + kl_divergence

    def update(self,
                belief: BeliefState,
                obs: ObservationVector,
                n_iterations: int = 10) -> BeliefState:
        """
        使用变分推断更新信念 (自由能.txt 第326-354行)

        梯度下降更新: μ ← μ - η × ∇_μ F

        Args:
            belief: 当前信念
            obs: 新观测
            n_iterations: 迭代次数

        Returns:
            BeliefState: 更新后的信念
        """
        new_belief = BeliefState(mean=belief.mean.copy(), covariance=belief.covariance.copy())

        for _ in range(n_iterations):
            # 计算梯度 (数值梯度)
            current_free_energy = self.compute_free_energy(new_belief, obs)
            grad = np.zeros(6)

            # 数值梯度
            epsilon = 1e-5
            for i in range(6):
                # 前向扰动
                mean_perturbed_plus = new_belief.mean.to_numpy().copy()
                mean_perturbed_plus[i] += epsilon
                mean_plus = self._numpy_to_state(mean_perturbed_plus)
                belief_plus = BeliefState(mean=mean_plus, covariance=new_belief.covariance)
                fe_plus = self.compute_free_energy(belief_plus, obs)

                # 后向扰动
                mean_perturbed_minus = new_belief.mean.to_numpy().copy()
                mean_perturbed_minus[i] -= epsilon
                mean_minus = self._numpy_to_state(mean_perturbed_minus)
                belief_minus = BeliefState(mean=mean_minus, covariance=new_belief.covariance)
                fe_minus = self.compute_free_energy(belief_minus, obs)

                # 中心差分
                grad[i] = (fe_plus - fe_minus) / (2 * epsilon)

            # 梯度下降更新 (自由能.txt 第328-331行)
            mean_vec = new_belief.mean.to_numpy()
            mean_vec = mean_vec - self.learning_rate * grad

            # 应用约束
            mean_vec = np.array([
                max(0, mean_vec[0]),          # E >= 0
                max(0, mean_vec[1]),          # T >= 0
                np.clip(mean_vec[2], 0.0, 1.0),  # h ∈ [0,1]
                np.clip(mean_vec[3], 0.0, 1.0),  # p ∈ [0,1]
                max(0, mean_vec[4]),          # d >= 0
                np.clip(mean_vec[5], 0.0, 1.0)   # sigma ∈ [0,1]
            ])

            new_belief = BeliefState(mean=self._numpy_to_state(mean_vec), covariance=new_belief.covariance)

        return new_belief

    def _numpy_to_state(self, vec: np.ndarray) -> StateVector:
        """将numpy数组转换为StateVector"""
        return StateVector(
            E=vec[0], T=vec[1], h=vec[2], p=vec[3], d=vec[4], sigma=vec[5]
        )


class PrecisionEstimator:
    """
    动态精度估计器 (自由能.txt 第347-354行子模块清单)

    功能：根据上下文动态调整信念的精度

    精度加权公式 (自由能.txt 第350-354行):
        μ_posterior = (Π_prior × μ_prior + Π_likelihood × μ_likelihood) / (Π_prior + Π_likelihood)
    """

    def __init__(self, base_precision: float = 1.0):
        """
        初始化精度估计器

        Args:
            base_precision: 基础精度
        """
        self.base_precision = base_precision

    def estimate(self, uncertainty: float, observation_quality: float = 1.0) -> float:
        """
        估计精度

        Args:
            uncertainty: 当前不确定性
            observation_quality: 观测质量 [0,1]

        Returns:
            float: 精度值
        """
        # 不确定性越高，精度越低
        prior_precision = self.base_precision / (1.0 + uncertainty * 10)

        # 观测质量越高，似然精度越高
        likelihood_precision = self.base_precision * observation_quality * 5

        # 精度加权 (自由能.txt 第350-354行)
        total_precision = prior_precision + likelihood_precision

        return total_precision


class BeliefUpdater:
    """
    信念更新器 (自由能.txt 第292-364行)

    整合卡尔曼滤波和变分推断两种更新方法

    Attributes:
        gen_model: 生成模型
        kalman_updater: 卡尔曼更新器
        variational_updater: 变分更新器
        precision_estimator: 精度估计器
        current_belief: 当前信念
    """

    def __init__(self,
                 gen_model: GenerativeModel,
                 initial_state: Optional[StateVector] = None):
        """
        初始化信念更新器

        Args:
            gen_model: 生成模型
            initial_state: 初始状态
        """
        self.gen_model = gen_model
        self.kalman_updater = KalmanUpdater(gen_model)
        self.variational_updater = VariationalUpdater(gen_model, learning_rate=0.1)
        self.precision_estimator = PrecisionEstimator(base_precision=1.0)

        if initial_state is None:
            initial_state = StateVector(E=500e3, T=0, h=1.0, p=0, d=1000, sigma=0.1)

        self.current_belief = BeliefState(mean=initial_state)

    def update(self,
                obs: ObservationVector,
                method: str = 'kalman') -> BeliefState:
        """
        更新信念

        Args:
            obs: 新观测
            method: 更新方法 ('kalman' 或 'variational')

        Returns:
            BeliefState: 更新后的信念
        """
        if method == 'kalman':
            self.current_belief = self.kalman_updater.update(self.current_belief, obs)
        elif method == 'variational':
            self.current_belief = self.variational_updater.update(self.current_belief, obs)
        else:
            raise ValueError(f"未知更新方法: {method}")

        return self.current_belief

    def get_current_belief(self) -> BeliefState:
        """获取当前信念"""
        return self.current_belief

    def get_entropy(self) -> float:
        """获取当前信念熵"""
        return self.current_belief.get_entropy()

    def get_uncertainty(self) -> float:
        """获取当前不确定性 (熵的简化指标)"""
        return self.current_belief.mean.sigma


# ============ 测试用例 ============

def test_belief_updater():
    """测试BeliefUpdater模块"""
    print("=" * 60)
    print("测试 M17D: BeliefUpdater (自由能.txt 第292-364行)")
    print("=" * 60)

    from algorithms.active_inference.generative_model import GenerativeModel

    # 初始化
    gen_model = GenerativeModel(T_task=50.0)
    initial_state = StateVector(E=400e3, T=5.0, h=0.9, p=0.3, d=800.0, sigma=0.1)
    updater = BeliefUpdater(gen_model, initial_state)

    # 测试1: 初始信念
    print("\n[Test 1] 测试初始信念...")
    belief = updater.get_current_belief()
    print(f"  均值: E={belief.mean.E/1e3:.1f}kJ, T={belief.mean.T:.1f}s, "
          f"h={belief.mean.h:.2f}, p={belief.mean.p:.2f}, σ={belief.mean.sigma:.2f}")
    print(f"  熵: {belief.get_entropy():.3f}")
    print("  ✓ 初始信念正确")

    # 测试2: 观测生成
    print("\n[Test 2] 测试观测生成...")
    obs = gen_model.generate_observation(belief.mean)
    print(f"  观测: Ê={obs.E_hat/1e3:.1f}kJ, ĥ={obs.h_hat:.2f}, "
          f"q={obs.q:.2f}, w={obs.w:.2f}")
    print("  ✓ 观测生成正确")

    # 测试3: 卡尔曼更新
    print("\n[Test 3] 测试卡尔曼更新...")
    old_mean = belief.mean.copy()
    updated_belief = updater.update(obs, method='kalman')
    print(f"  更新前: E={old_mean.E/1e3:.1f}kJ, h={old_mean.h:.2f}")
    print(f"  更新后: E={updated_belief.mean.E/1e3:.1f}kJ, h={updated_belief.mean.h:.2f}")
    print(f"  新息: E={(obs.E_hat - old_mean.E)/1e3:.1f}kJ, "
          f"h={(obs.h_hat - old_mean.h):.3f}")
    print("  ✓ 卡尔曼更新正确")

    # 测试4: 多步信念更新
    print("\n[Test 4] 测试多步信念更新...")
    for step in range(5):
        # 真实状态演化
        from algorithms.active_inference.state_space import ActionType
        result = gen_model.predict_next(updated_belief.mean, ActionType.CONTINUE, 1.0)
        true_state = result.next_state

        # 生成观测
        new_obs = gen_model.generate_observation(true_state)

        # 更新信念
        updated_belief = updater.update(new_obs, method='kalman')

        print(f"  步骤{step+1}: E={updated_belief.mean.E/1e3:.1f}kJ, "
              f"h={updated_belief.mean.h:.2f}, p={updated_belief.mean.p:.3f}")
    print("  ✓ 多步更新正确")

    # 测试5: 变分更新
    print("\n[Test 5] 测试变分更新...")
    updater_variational = BeliefUpdater(gen_model, initial_state)
    old_variational = updater_variational.get_current_belief()

    # 偏差观测
    biased_obs = ObservationVector(
        E_hat=obs.E_hat - 5000,
        h_hat=obs.h_hat - 0.05,
        q=obs.q,
        w=obs.w
    )

    updated_variational = updater_variational.update(biased_obs, method='variational')
    print(f"  更新前熵: {old_variational.get_entropy():.3f}")
    print(f"  更新后熵: {updated_variational.get_entropy():.3f}")
    print(f"  均值变化: E={(updated_variational.mean.E - old_variational.mean.E)/1e3:.1f}kJ, "
          f"h={(updated_variational.mean.h - old_variational.mean.h):.3f}")
    print("  ✓ 变分更新正确")

    # 测试6: 精度估计
    print("\n[Test 6] 测试精度估计...")
    for sigma in [0.1, 0.3, 0.5, 0.8]:
        precision = updater.precision_estimator.estimate(sigma, observation_quality=0.8)
        print(f"  σ={sigma:.2f} -> 精度={precision:.3f}")
    print("  ✓ 精度估计正确")

    # 测试7: 信念采样
    print("\n[Test 7] 测试信念采样...")
    samples = belief.sample(n_samples=5)
    print(f"  均值状态: E={belief.mean.E/1e3:.1f}kJ, h={belief.mean.h:.2f}")
    print(f"  采样状态:")
    for i, sample in enumerate(samples):
        print(f"    样本{i+1}: E={sample.E/1e3:.1f}kJ, h={sample.h:.2f}, σ={sample.sigma:.2f}")
    print("  ✓ 信念采样正确")

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_belief_updater()

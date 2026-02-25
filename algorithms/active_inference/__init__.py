"""
Active Inference Package - 主动推理框架

实现文档 docs/自由能.txt 中定义的完整主动推理框架

模块列表:
    M17A: StateSpace - 状态空间定义
    M17B: GenerativeModel - 生成模型
    M17C: TrajectoryPredictor - 轨迹预测器
    M17D: BeliefUpdater - 信念更新器
    M17E: FreeEnergyCalculator - 自由能计算器 (扩展四分量)
    M17F: ActionSelector - 行动选择器
    M17G: PerceptionActionLoop - 感知-行动循环

使用示例:
    from algorithms.active_inference import (
        StateVector, ActionType, PerceptionActionLoop
    )

    # 创建循环
    initial_state = StateVector(E=400e3, T=0, h=0.9, p=0, d=800, sigma=0.1)
    loop = PerceptionActionLoop(initial_state=initial_state)

    # 运行循环
    history = loop.run(max_steps=100, verbose=True)
    summary = loop.get_summary()
"""

# M17A: StateSpace
from algorithms.active_inference.state_space import (
    StateVector,
    ObservationVector,
    ActionType,
    ActionEffect,
    StateBounds,
    StateNormalizer,
    ActionSet
)

# M17B: GenerativeModel
from algorithms.active_inference.generative_model import (
    GenerativeModel,
    TransitionModel,
    LikelihoodModel,
    TransitionResult,
    NoiseParameters
)

# M17C: TrajectoryPredictor
from algorithms.active_inference.trajectory_predictor import (
    TrajectoryPredictor,
    TrajectoryState,
    Trajectory,
    DeterministicPredictor,
    UncertaintyPropagator,
    MonteCarloSampler
)

# M17D: BeliefUpdater
from algorithms.active_inference.belief_updater import (
    BeliefUpdater,
    BeliefState,
    KalmanUpdater,
    VariationalUpdater,
    PrecisionEstimator
)

# M17E: FreeEnergyCalculator
from algorithms.active_inference.free_energy import (
    FreeEnergyCalculator,
    FourComponentCalculator,
    ExpectedFreeEnergyCalculator,
    InstantFreeEnergy,
    ExpectedFreeEnergy,
    RiskLevel
)

# M17F: ActionSelector
from algorithms.active_inference.action_selector import (
    ActionSelector,
    GreedySelector,
    SoftmaxSelector,
    ConstraintChecker,
    RollingHorizonOptimizer,
    ActionEvaluation
)

# M17G: PerceptionActionLoop
from algorithms.active_inference.perception_action_loop import (
    PerceptionActionLoop,
    LoopPhase,
    LoopState,
    ExecutionResult,
    TimeBudgetManager,
    ObservationInterface,
    ExecutionInterface
)

__all__ = [
    # M17A: StateSpace
    'StateVector',
    'ObservationVector',
    'ActionType',
    'ActionEffect',
    'StateBounds',
    'StateNormalizer',
    'ActionSet',

    # M17B: GenerativeModel
    'GenerativeModel',
    'TransitionModel',
    'LikelihoodModel',
    'TransitionResult',
    'NoiseParameters',

    # M17C: TrajectoryPredictor
    'TrajectoryPredictor',
    'TrajectoryState',
    'Trajectory',
    'DeterministicPredictor',
    'UncertaintyPropagator',
    'MonteCarloSampler',

    # M17D: BeliefUpdater
    'BeliefUpdater',
    'BeliefState',
    'KalmanUpdater',
    'VariationalUpdater',
    'PrecisionEstimator',

    # M17E: FreeEnergyCalculator
    'FreeEnergyCalculator',
    'FourComponentCalculator',
    'ExpectedFreeEnergyCalculator',
    'InstantFreeEnergy',
    'ExpectedFreeEnergy',
    'RiskLevel',

    # M17F: ActionSelector
    'ActionSelector',
    'GreedySelector',
    'SoftmaxSelector',
    'ConstraintChecker',
    'RollingHorizonOptimizer',
    'ActionEvaluation',

    # M17G: PerceptionActionLoop
    'PerceptionActionLoop',
    'LoopPhase',
    'LoopState',
    'ExecutionResult',
    'TimeBudgetManager',
    'ObservationInterface',
    'ExecutionInterface',
]

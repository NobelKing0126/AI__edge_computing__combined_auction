"""
Test script to verify all Active Inference modules can be imported
"""

def test_imports():
    """Test all module imports"""
    print("=" * 60)
    print("Testing Active Inference Module Imports")
    print("=" * 60)

    errors = []

    # Test M17A: StateSpace
    print("\n[1/7] Testing M17A: StateSpace...")
    try:
        from algorithms.active_inference import (
            StateVector, ActionType, ActionSet, StateNormalizer
        )
        state = StateVector(E=400e3, T=10.0, h=0.9, p=0.5, d=800, sigma=0.1)
        action_set = ActionSet()
        print(f"  State: E={state.E/1e3:.1f}kJ, T={state.T:.1f}s, h={state.h:.2f}")
        print(f"  Actions: {len(action_set.get_all_actions())}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17A")

    # Test M17B: GenerativeModel
    print("\n[2/7] Testing M17B: GenerativeModel...")
    try:
        from algorithms.active_inference import (
            GenerativeModel, TransitionModel, LikelihoodModel
        )
        gen_model = GenerativeModel(T_task=50.0)
        result = gen_model.predict_next(state, ActionType.CONTINUE, 1.0)
        print(f"  Next state: E={result.next_state.E/1e3:.1f}kJ, h={result.next_state.h:.2f}")
        print(f"  Energy consumed: {result.energy_consumed:.1f}J")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17B")

    # Test M17C: TrajectoryPredictor
    print("\n[3/7] Testing M17C: TrajectoryPredictor...")
    try:
        from algorithms.active_inference import (
            TrajectoryPredictor, Trajectory, DeterministicPredictor
        )
        traj_pred = TrajectoryPredictor(gen_model)
        policy = [ActionType.CONTINUE] * 5
        traj, stats = traj_pred.predict(state, policy, method='deterministic')
        print(f"  Trajectory length: {len(traj)}")
        print(f"  Final progress: {traj.final_progress:.3f}")
        print(f"  Mean uncertainty: {stats['mean_uncertainty']:.3f}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17C")

    # Test M17D: BeliefUpdater
    print("\n[4/7] Testing M17D: BeliefUpdater...")
    try:
        from algorithms.active_inference import (
            BeliefUpdater, BeliefState, KalmanUpdater
        )
        belief_updater = BeliefUpdater(gen_model, state)
        belief = belief_updater.get_current_belief()
        print(f"  Belief mean: E={belief.mean.E/1e3:.1f}kJ, h={belief.mean.h:.2f}")
        print(f"  Entropy: {belief.get_entropy():.3f}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17D")

    # Test M17E: FreeEnergyCalculator
    print("\n[5/7] Testing M17E: FreeEnergyCalculator...")
    try:
        from algorithms.active_inference import (
            FreeEnergyCalculator, FourComponentCalculator, InstantFreeEnergy
        )
        fe_calc = FreeEnergyCalculator(use_four_component=True)
        instant_fe = fe_calc.compute_instant(
            state, E_required=100e3, T_remaining_required=30.0
        )
        print(f"  F_total: {instant_fe.F_total:.2f}")
        print(f"  F_energy: {instant_fe.F_energy:.2f}")
        print(f"  F_time: {instant_fe.F_time:.2f}")
        print(f"  F_health: {instant_fe.F_health:.2f}")
        print(f"  F_progress: {instant_fe.F_progress:.2f}")
        print(f"  Risk level: {instant_fe.risk_level.name}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17E")

    # Test M17F: ActionSelector
    print("\n[6/7] Testing M17F: ActionSelector...")
    try:
        from algorithms.active_inference import (
            ActionSelector, GreedySelector, SoftmaxSelector
        )
        action_selector = ActionSelector(traj_pred, fe_calc)
        action_selector.set_evaluation_params(
            E_required=100e3, T_remaining_required=30.0
        )
        choice = action_selector.select_action(state)
        print(f"  Selected action: {choice.action.value}")
        print(f"  G_value: {choice.G_value:.2f}")
        print(f"  Confidence: {choice.action_confidence:.2f}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17F")

    # Test M17G: PerceptionActionLoop
    print("\n[7/7] Testing M17G: PerceptionActionLoop...")
    try:
        from algorithms.active_inference import (
            PerceptionActionLoop, LoopPhase, TimeBudgetManager
        )
        loop = PerceptionActionLoop(initial_state=state, H=3)
        print(f"  Initial belief: E={loop.loop_state.belief.mean.E/1e3:.1f}kJ")
        print(f"  Time manager budget: {loop.time_manager.T_budget:.2f}s")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        errors.append("M17G")

    # Summary
    print("\n" + "=" * 60)
    print("Import Test Summary")
    print("=" * 60)
    print(f"Total modules tested: 7")
    print(f"Passed: {7 - len(errors)}")
    print(f"Failed: {len(errors)}")

    if errors:
        print(f"Failed modules: {errors}")
    else:
        print("All modules imported successfully!")

    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)

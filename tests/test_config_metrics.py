from rl_research.config import load_experiment_config
from rl_research.evaluation.metrics import summarize_eval


def test_config_loads_eval_and_cost_blocks() -> None:
    config = load_experiment_config("configs/train/hopper_v5_abrupt.yaml")

    assert config.train.env_id == "Hopper-v5"
    assert config.eval.eval_episodes == 3
    assert config.cost.include_reward_ctrl is True


def test_zero_update_reward_per_update_is_not_rankable() -> None:
    record = summarize_eval(
        pretrain_returns=[10.0],
        deploy_returns=[8.0, 9.0],
        update_cost=0,
        replay_samples=0,
        wallclock_update_time=0.0,
        reference_return=10.0,
    )

    assert record.reward_per_update is None
    assert record.normalized_budgeted_score is not None


def test_negative_reward_recovery_metric_has_valid_range() -> None:
    record = summarize_eval(
        pretrain_returns=[-100.0],
        deploy_returns=[-120.0, -95.0],
        update_cost=10,
        replay_samples=640,
        wallclock_update_time=0.1,
        reference_return=-100.0,
    )

    assert 0.0 <= record.recovery_auc <= 1.0
    assert record.steps_to_recovery == 1

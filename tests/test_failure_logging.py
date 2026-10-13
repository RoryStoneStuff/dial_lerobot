from types import SimpleNamespace
import numpy as np
from dial_lerobot.config import Config
from dial_lerobot.evaluator import Evaluator
from dial_lerobot.tasks.lerobot_task import COST_NAMES


def test_first_nonfinite_update_returns_empty_prefix_and_failure_reason():
    e = object.__new__(Evaluator)
    e.cfg = Config()
    state = SimpleNamespace(time=np.array(0.), qpos=np.array([0., 0., .72]), qvel=np.zeros(18))
    e.reset = lambda seed: (state, None)
    row = {'time': np.array(.02), 'qpos': np.full(19, np.nan), 'qvel': np.zeros(18),
           'ctrl': np.zeros(12), 'action': np.zeros(12), 'cost_terms': np.zeros(len(COST_NAMES)),
           'plan_cost_min': np.array(0.), 'plan_cost_scaled_std': np.array(0.),
           'plan_weight_ess': np.array(1.),
           'body_velocity': np.zeros(3), 'target_velocity': np.zeros(2),
           'feet': np.zeros((2, 3)), 'healthy': np.array(False)}
    e.step = lambda state, params: (state, params, row)
    summary, arrays = e.episode(45)
    assert summary['failure_reason'] == 'nonfinite'
    assert summary['failure_time_s'] == .02
    assert summary['updates'] == 0
    assert summary['sim_seconds'] == 0
    assert arrays['qpos'].shape == (0, 19)
    assert summary['failure_diagnostic']['failed_fields']['qpos']['nonfinite_count'] == 19
    assert summary['failure_diagnostic']['update'] == 1

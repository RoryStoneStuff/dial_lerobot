"""One configuration shared by headless evaluation, DIAL viewer and co-design."""

from dataclasses import asdict, dataclass, field, fields, replace
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RobotConfig:
    zoo_root: str | None = None
    timestep: float = 0.02
    control_mode: str = "torque"
    solver_iterations: int = 100
    ls_iterations: int = 50
    soft_joint_limit_factor: float = 0.9
    initial_clearance: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    target_vx: float = 0.5
    target_vy: float = 0.0
    ramp_up_time: float = 2.0
    gait_weight: float = 5.0
    ref_tracking_weight: float = 3.0
    velocity_weight: float = 4.0
    upright_weight: float = 5.0
    height_weight: float = 1.0
    hip_lateral_weight: float = 6.0
    contact_weight: float = 0.05
    yaw_weight: float = 0.1
    angular_velocity_weight: float = 1.0
    energy_weight: float = 0.01
    unhealthy_penalty: float = 10.0
    duty: float = 0.50
    cadence: float = 1.0
    swing_height: float = 0.05
    hip_amplitude: float = 0.25
    knee_amplitude: float = 0.35
    # Berkeley coordinates, translated to the zoo CAD zero pose in the task.
    qstand_hip: float = -0.20
    qstand_knee: float = 0.40
    source_rest_height: float = 0.85
    source_healthy_z_min: float = 0.6
    rest_height: float | None = None
    healthy_z_min: float | None = None
    healthy_up_dot: float = 0.8
    terminal_height_weight: float = 5.0
    terminal_velocity_weight: float = 2.0
    terminal_upright_weight: float = 10.0


@dataclass(frozen=True)
class MPCConfig:
    plan_horizon: float = 0.5
    num_knots: int = 4
    num_samples: int = 1024
    noise_level: float = 0.3
    temperature: float = 0.5
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    iterations: int = 1
    spline_type: str = "zero"
    # Original evaluator replans every physics step. Set to 4 for 50 Hz.
    control_decimation: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    episode_steps: int = 300  # MPC updates; dt * decimation seconds per update.
    num_episodes: int = 2
    height_jitter: float = 0.02
    terminate_on_fall: bool = True
    seed: int = 0


@dataclass(frozen=True)
class DesignConfig:
    sigma0: float = 0.05
    lower_bounds: tuple[float, ...] = (0.05, 0.05, 0.1)
    upper_bounds: tuple[float, ...] = (0.4, 0.4, 8.0)
    maxiter: int = 80
    popsize: int = 12
    seed: int = 42


@dataclass(frozen=True)
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    design: DesignConfig = field(default_factory=DesignConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for section in (self.robot, self.cost, self.mpc, self.evaluation, self.design):
            for f in fields(section):
                value = getattr(section, f.name)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{f.name} must be finite")
        positive = {
            "timestep": self.robot.timestep,
            "plan_horizon": self.mpc.plan_horizon,
            "temperature": self.mpc.temperature,
            "noise_level": self.mpc.noise_level,
            "beta_opt_iter": self.mpc.beta_opt_iter,
            "beta_horizon": self.mpc.beta_horizon,
            "cadence": self.cost.cadence,
            "source_rest_height": self.cost.source_rest_height,
            "sigma0": self.design.sigma0,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "solver_iterations": self.robot.solver_iterations,
            "ls_iterations": self.robot.ls_iterations,
            "num_knots": self.mpc.num_knots,
            "num_samples": self.mpc.num_samples,
            "iterations": self.mpc.iterations,
            "control_decimation": self.mpc.control_decimation,
            "episode_steps": self.evaluation.episode_steps,
            "num_episodes": self.evaluation.num_episodes,
            "maxiter": self.design.maxiter,
            "popsize": self.design.popsize,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mpc.num_knots < 2 or self.design.popsize < 2:
            raise ValueError("num_knots and popsize must be at least 2")
        if self.mpc.spline_type not in ("zero", "linear"):
            raise ValueError("Use zero or linear splines to preserve bounded control inputs")
        if self.robot.control_mode not in ("torque", "position"):
            raise ValueError("control_mode must be torque or position")
        if self.mpc.plan_horizon < self.robot.timestep * self.mpc.control_decimation:
            raise ValueError("Planning horizon must cover at least one MPC update")
        if not 0 < self.cost.duty < 1:
            raise ValueError("duty must be between 0 and 1")
        if not 0 < self.robot.soft_joint_limit_factor <= 1:
            raise ValueError("soft_joint_limit_factor must be in (0, 1]")
        if self.evaluation.height_jitter < 0 or self.robot.initial_clearance < 0:
            raise ValueError("Clearance and height jitter must be nonnegative")
        if len(self.design.lower_bounds) != 3 or len(self.design.upper_bounds) != 3:
            raise ValueError("Design bounds require thigh length, shin length and density proxy")
        if any(not math.isfinite(lo + hi) or not 0 < lo < hi for lo, hi in
               zip(self.design.lower_bounds, self.design.upper_bounds)):
            raise ValueError("Design bounds must be finite, positive and ordered")
        return self


def from_dict(data: dict) -> Config:
    cfg = Config()
    allowed = {f.name for f in fields(cfg)}
    if unknown := data.keys() - allowed:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section, values in data.items():
        if section == "design":
            values = dict(values)
            for name in ("lower_bounds", "upper_bounds"):
                if name in values:
                    values[name] = tuple(values[name])
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **values)})
    return cfg.validate()


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config().validate()
    with Path(path).open("rb") as f:
        return from_dict(tomllib.load(f))

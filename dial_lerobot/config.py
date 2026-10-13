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
    # Unitree H1 MJCF default for actuated joints.
    joint_damping: float = 2.0
    # Collision geometry used by the feet. ``native`` keeps the zoo's six
    # capsule approximation; ``h1_pair`` uses two H1-style support rails per
    # foot while retaining the LeRobot foot body and visual mesh.
    foot_contact_model: str = "native"
    soft_joint_limit_factor: float = 0.9
    initial_clearance: float = 0.0
    preserve_link_masses: bool = False


@dataclass(frozen=True)
class CostConfig:
    target_vx: float = 0.5
    target_vy: float = 0.0
    # Optional commanded yaw rate in rad/s. Zero preserves translation-only behavior.
    target_wz: float = 0.0
    ramp_up_time: float = 2.0
    gait_weight: float = 5.0
    ref_tracking_weight: float = 3.0
    velocity_weight: float = 4.0
    upright_weight: float = 5.0
    height_weight: float = 1.0
    hip_lateral_weight: float = 6.0
    contact_weight: float = 0.05
    contact_tolerance: float = 0.002
    contact_height_scale: float = 0.02
    stance_slip_weight: float = 0.0
    stance_slip_scale: float = 0.2
    gait_start_delay: float = 0.0
    yaw_weight: float = 0.1
    angular_velocity_weight: float = 1.0
    yaw_rate_weight: float = 0.0
    # Source H1 locomotion includes a small flat-foot/orientation term.
    foot_level_weight: float = 0.0
    # Weight for the Cartesian forward foot-placement objective.
    foot_placement_weight: float = 0.0
    energy_weight: float = 0.01
    # Disabled auxiliary terms. Air-time is a schedule-only placeholder;
    # contact_event is an extra LeRobot term, absent from source H1 walking.
    air_time_weight: float = 0.0
    position_weight: float = 0.0
    # Penalize whole-body COM leaving the horizontal support bounds of the
    # actual foot collision geometry. Keep zero for locomotion; enable for
    # standing experiments so stepping is not blocked by this term.
    com_support_weight: float = 0.0
    com_support_margin: float = 0.01
    com_support_scale: float = 0.1
    contact_event_weight: float = 0.0
    alive_weight: float = 0.0
    unhealthy_penalty: float = 10.0
    duty: float = 0.50
    cadence: float = 1.0
    swing_height: float = 0.05
    hip_amplitude: float = 0.25
    knee_amplitude: float = 0.35
    # Grounded LeRobot standing pose in task joint coordinates.
    qstand_hip: float = -0.1
    qstand_knee: float = -0.05
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
    initial_diffusion_iterations: int = 1
    trajectory_diffuse_factor: float = 0.2
    normalize_costs_for_temperature: bool = False
    append_mean_sample: bool = False
    fix_initial_knot: bool = False
    spline_type: str = "zero"
    # Reconstruct the shifted action sequence into knots, as in the H1 repo.
    shift_reconstruction: bool = False
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
    fixed_density_proxy: float | None = None


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
            "trajectory_diffuse_factor": self.mpc.trajectory_diffuse_factor,
            "cadence": self.cost.cadence,
            "contact_height_scale": self.cost.contact_height_scale,
            "stance_slip_scale": self.cost.stance_slip_scale,
            "com_support_scale": self.cost.com_support_scale,
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
            "initial_diffusion_iterations": self.mpc.initial_diffusion_iterations,
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
        if not 0 < self.mpc.trajectory_diffuse_factor <= 1:
            raise ValueError("trajectory_diffuse_factor must be in (0, 1]")
        if self.mpc.spline_type not in ("zero", "linear", "quadratic", "quadratic_dial"):
            raise ValueError("Use zero, linear, quadratic, or quadratic_dial splines")
        if self.mpc.spline_type in ("quadratic", "quadratic_dial") and self.mpc.num_knots < 3:
            raise ValueError("quadratic interpolation requires at least 3 knots")
        if self.robot.control_mode not in ("torque", "position"):
            raise ValueError("control_mode must be torque or position")
        if self.robot.foot_contact_model not in ("native", "h1_pair"):
            raise ValueError("foot_contact_model must be native or h1_pair")
        if self.mpc.plan_horizon < self.robot.timestep * self.mpc.control_decimation:
            raise ValueError("Planning horizon must cover at least one MPC update")
        if not 0 < self.cost.duty < 1:
            raise ValueError("duty must be between 0 and 1")
        if not 0 < self.robot.soft_joint_limit_factor <= 1:
            raise ValueError("soft_joint_limit_factor must be in (0, 1]")
        if self.evaluation.height_jitter < 0 or self.robot.initial_clearance < 0:
            raise ValueError("Clearance and height jitter must be nonnegative")
        for name in ("contact_weight", "contact_tolerance", "stance_slip_weight", "gait_start_delay",
                     "air_time_weight", "position_weight", "contact_event_weight", "alive_weight",
                     "foot_level_weight", "foot_placement_weight", "com_support_weight", "com_support_margin",
                     "yaw_rate_weight"):
            if getattr(self.cost, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (len(self.design.lower_bounds) not in (3, 4)
                or len(self.design.upper_bounds) != len(self.design.lower_bounds)):
            raise ValueError("Design bounds require thigh, shin, density proxy, and optional hip width")
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

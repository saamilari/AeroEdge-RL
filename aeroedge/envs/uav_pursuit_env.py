"""
Custom 3D Fixed-Wing / Quad-Hybrid UAV Pursuit & Trajectory Control Gymnasium Environment.

Features:
- Realistic 3D non-linear point-mass + attitude-rate dynamics with first-order actuator lag
- Dryden continuous wind turbulence & gust disturbance model
- Realistic optical sensor imperfections:
  - Camera Field-of-View (FOV) constraints
  - Distance-dependent bounding-box noise & false-negative dropouts (cloud/smoke occlusion bursts)
  - Configurable sensor pipeline latency queue
- Integrated Multi-Sensor EKF state estimator
- Baseline classical controllers (Cascaded PID and 3D Augmented Proportional Navigation) for benchmarking
"""

from __future__ import annotations
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from aeroedge.fusion.ekf_tracker import MultiSensorEKF


class DrydenWindModel:
    """First-order Gauss-Markov approximation of low-altitude Dryden wind turbulence."""

    def __init__(self, dt: float = 0.05, sigma_turbulence: float = 2.5, tau_wind: float = 3.5) -> None:
        self.dt = dt
        self.sigma = sigma_turbulence
        self.tau = tau_wind
        self.wind_vel = np.zeros(3, dtype=np.float64)

    def reset(self, rng: np.random.Generator, scale: float = 1.0) -> None:
        self.wind_vel = rng.normal(0.0, self.sigma * scale * 0.5, size=3)
        self.wind_vel[2] *= 0.4  # Vertical turbulence is weaker than horizontal

    def step(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        alpha = np.exp(-self.dt / self.tau)
        noise = rng.normal(0.0, self.sigma * scale * np.sqrt(1.0 - alpha ** 2), size=3)
        noise[2] *= 0.4
        self.wind_vel = alpha * self.wind_vel + noise
        return self.wind_vel.copy()


class UAVPursuitEnv(gym.Env):
    """
    Closed-Loop UAV Target Tracking & Trajectory Optimization Environment.

    Observation Space (16-D vector at each timestep, stacked into sequence of length K by the policy):
        0..10 : 11D normalized EKF state & uncertainty features (or raw noisy measurement if use_ekf=False)
        11..13: Ego UAV normalized speed, pitch angle, roll/bank angle
        14..15: Previous control action [yaw_rate_cmd, pitch_rate_cmd] (for jerk/smoothness awareness)

    Action Space (3-D continuous in [-1, 1]):
        action[0] : Commanded heading/yaw rate   (omega_psi in [-1.2, +1.2] rad/s)
        action[1] : Commanded pitch rate         (omega_theta in [-0.9, +0.9] rad/s)
        action[2] : Commanded airspeed throttle  (v_cmd in [18.0, 38.0] m/s)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 300,
        use_ekf: bool = True,
        wind_scale: float = 1.0,
        occlusion_prob: float = 0.12,
        sensor_noise_scale: float = 1.0,
        sensor_delay_steps: int = 1,
        reward_mode: str = "shaped_composite",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.use_ekf = use_ekf
        self.wind_scale = wind_scale
        self.occlusion_prob = occlusion_prob
        self.sensor_noise_scale = sensor_noise_scale
        self.sensor_delay_steps = max(0, int(sensor_delay_steps))
        self.reward_mode = reward_mode

        self.fov_half_angle = np.deg2rad(35.0)  # 70 deg total camera FOV
        self.optimal_distance = 55.0            # Desired standoff tracking distance [m]
        self.actuator_tau = 0.12                # First-order servo/motor time constant [s]

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.wind = DrydenWindModel(dt=dt)
        self.ekf = MultiSensorEKF(dt=dt)

        # Dynamic state variables initialized in reset()
        self.ego_pos = np.zeros(3, dtype=np.float64)
        self.ego_vel = np.zeros(3, dtype=np.float64)
        self.ego_yaw = 0.0
        self.ego_pitch = 0.0
        self.ego_speed = 26.0
        self.actual_act = np.zeros(3, dtype=np.float64)
        self.prev_action = np.zeros(3, dtype=np.float32)

        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_vel = np.zeros(3, dtype=np.float64)
        self.target_acc = np.zeros(3, dtype=np.float64)
        self.target_maneuver_phase = 0.0

        self.obs_delay_queue: deque = deque(maxlen=max(1, self.sensor_delay_steps + 1))
        self.occlusion_burst_remaining = 0
        self.step_count = 0

        # Last raw measurement for visualization
        self.last_raw_meas: np.ndarray | None = None
        self.last_conf: float = 0.0

    def _ego_rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from world NED/ENU frame to Ego camera-stabilized frame."""
        cy, sy = np.cos(self.ego_yaw), np.sin(self.ego_yaw)
        cp, sp = np.cos(self.ego_pitch), np.sin(self.ego_pitch)

        # Forward (x), Left (y), Up (z) body axes
        R_yaw = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        return R_pitch @ R_yaw

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.ego_pos = np.array([0.0, 0.0, 120.0], dtype=np.float64)
        self.ego_yaw = float(self.rng.uniform(-0.15, 0.15))
        self.ego_pitch = float(self.rng.uniform(-0.08, 0.08))
        self.ego_speed = 26.0
        self.actual_act = np.zeros(3, dtype=np.float64)
        self.prev_action = np.zeros(3, dtype=np.float32)

        # Spawn agile target ahead within camera FOV at 65..105m distance
        init_dist = float(self.rng.uniform(65.0, 105.0))
        init_azim = float(self.rng.uniform(-0.32, 0.32))
        init_elev = float(self.rng.uniform(-0.20, 0.20))

        self.target_pos = self.ego_pos + np.array(
            [
                init_dist * np.cos(init_elev) * np.cos(init_azim),
                init_dist * np.cos(init_elev) * np.sin(init_azim),
                init_dist * np.sin(init_elev),
            ],
            dtype=np.float64,
        )

        target_speed = float(self.rng.uniform(21.0, 29.0))
        target_heading = float(self.rng.uniform(-0.45, 0.45))
        self.target_vel = np.array(
            [
                target_speed * np.cos(target_heading),
                target_speed * np.sin(target_heading),
                float(self.rng.uniform(-1.5, 1.5)),
            ],
            dtype=np.float64,
        )
        self.target_acc = np.zeros(3, dtype=np.float64)
        self.target_maneuver_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        self.wind.reset(self.rng, scale=self.wind_scale)
        self.occlusion_burst_remaining = 0

        R = self._ego_rotation_matrix()
        rel_pos_body = R @ (self.target_pos - self.ego_pos)
        self.ego_vel = np.array(
            [
                self.ego_speed * np.cos(self.ego_pitch) * np.cos(self.ego_yaw),
                self.ego_speed * np.cos(self.ego_pitch) * np.sin(self.ego_yaw),
                self.ego_speed * np.sin(self.ego_pitch),
            ]
        )
        rel_vel_body = R @ (self.target_vel - self.ego_vel)
        self.ekf.reset(rel_pos_body, rel_vel_body)

        self.obs_delay_queue.clear()
        obs = self._build_observation(rel_pos_body, ego_acc_body=np.zeros(3))
        return obs, self._get_info(rel_pos_body)

    def _step_target_dynamics(self) -> None:
        """Simulate agile evasive 3D weaving + randomized turn maneuvers for the target UAV."""
        t = self.step_count * self.dt
        omega_1 = 0.55
        omega_2 = 1.15
        lateral_acc = 4.5 * np.sin(omega_1 * t + self.target_maneuver_phase) + 2.2 * np.cos(
            omega_2 * t
        )
        vert_acc = 1.8 * np.cos(0.7 * t + self.target_maneuver_phase)

        # Convert lateral acceleration perpendicular to target velocity in horizontal plane
        horiz_speed = max(np.hypot(self.target_vel[0], self.target_vel[1]), 1.0)
        perp_dir = np.array([-self.target_vel[1] / horiz_speed, self.target_vel[0] / horiz_speed, 0.0])

        self.target_acc = perp_dir * lateral_acc + np.array([0.0, 0.0, vert_acc])
        self.target_vel += self.target_acc * self.dt

        # Clamp target speed to realistic envelope [20, 31] m/s
        spd = np.linalg.norm(self.target_vel)
        if spd > 31.0:
            self.target_vel *= 31.0 / spd
        elif spd < 20.0:
            self.target_vel *= 20.0 / max(spd, 1e-3)

        self.target_pos += self.target_vel * self.dt
        self.target_pos[2] = np.clip(self.target_pos[2], 30.0, 260.0)

    def _simulate_small_object_sensor(
        self, true_rel_pos_body: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Simulate optical small-object detector + scale range cue.
        Small distant objects have lower confidence, higher pixel localization variance,
        and periodic occlusion bursts (cloud/smoke/motion blur).
        """
        true_sph = MultiSensorEKF.cartesian_to_spherical(true_rel_pos_body)
        azim, elev, dist = true_sph[0], true_sph[1], true_sph[2]

        # Check camera FOV bounds
        if abs(azim) > self.fov_half_angle * 1.15 or abs(elev) > self.fov_half_angle * 1.15:
            return None, 0.0

        # Manage multi-frame occlusion bursts (simulating target passing behind cloud/clutter)
        if self.occlusion_burst_remaining > 0:
            self.occlusion_burst_remaining -= 1
            return None, 0.0
        elif self.rng.random() < self.occlusion_prob * 0.25:
            self.occlusion_burst_remaining = int(self.rng.integers(3, 8))
            return None, 0.0

        # Apparent pixel size ~ focal_length * wingspan / dist
        # Confidence degrades with distance and off-axis angle
        dist_factor = np.clip(1.0 - (dist - 35.0) / 140.0, 0.22, 0.98)
        off_axis = np.hypot(azim, elev) / self.fov_half_angle
        conf = float(
            np.clip(
                dist_factor * (1.0 - 0.25 * off_axis)
                + self.rng.normal(0.0, 0.05 * self.sensor_noise_scale),
                0.10,
                0.99,
            )
        )

        # Noise increases for smaller targets (larger distance)
        angle_std = (0.008 + 0.015 * (dist / 100.0)) * self.sensor_noise_scale
        range_std = (1.2 + 2.8 * (dist / 100.0) ** 1.5) * self.sensor_noise_scale

        noisy_meas = np.array(
            [
                azim + self.rng.normal(0.0, angle_std),
                elev + self.rng.normal(0.0, angle_std),
                max(5.0, dist + self.rng.normal(0.0, range_std)),
            ],
            dtype=np.float64,
        )
        return noisy_meas, conf

    def _build_observation(
        self, true_rel_pos_body: np.ndarray, ego_acc_body: np.ndarray
    ) -> np.ndarray:
        raw_meas, conf = self._simulate_small_object_sensor(true_rel_pos_body)
        self.obs_delay_queue.append((raw_meas, conf))

        delayed_meas, delayed_conf = self.obs_delay_queue[0]
        self.last_raw_meas = delayed_meas
        self.last_conf = delayed_conf

        ego_ang_vel = np.array([self.actual_act[1] * 0.90, self.actual_act[0] * 1.25])
        self.ekf.predict(ego_accel_cmd=ego_acc_body, ego_ang_vel=ego_ang_vel, dt=self.dt)
        self.ekf.update(delayed_meas, confidence=delayed_conf)

        if self.use_ekf:
            sensor_feats = self.ekf.get_normalized_features()
        else:
            # Ablation mode: without EKF sensor fusion (raw noisy measurement + zero velocity estimate)
            if delayed_meas is not None:
                az, el, rg = delayed_meas[0], delayed_meas[1], delayed_meas[2]
            else:
                # Without EKF, target occlusion causes measurement dropout & stale blind drift
                az, el, rg = 0.0, 0.0, 60.0
            sensor_feats = np.array(
                [
                    az / (np.pi / 3.0),
                    el / (np.pi / 3.0),
                    (rg - 60.0) / 60.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 if delayed_meas is None else 0.2,
                    1.0 if delayed_meas is None else 0.0,
                ],
                dtype=np.float32,
            )

        ego_feats = np.array(
            [
                (self.ego_speed - 28.0) / 10.0,
                self.ego_pitch / (np.pi / 4.0),
                self.actual_act[0],  # normalized yaw rate state
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([sensor_feats, ego_feats]).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # First-order actuator lag
        alpha_act = self.dt / (self.actuator_tau + self.dt)
        prev_actual = self.actual_act.copy()
        self.actual_act = (1.0 - alpha_act) * self.actual_act + alpha_act * act

        yaw_rate = self.actual_act[0] * 1.25      # rad/s
        pitch_rate = self.actual_act[1] * 0.90    # rad/s
        target_airspeed = 28.0 + self.actual_act[2] * 10.0  # [18, 38] m/s

        prev_vel = self.ego_vel.copy()

        # Integrate ego attitude & speed
        self.ego_yaw = (self.ego_yaw + yaw_rate * self.dt + np.pi) % (2.0 * np.pi) - np.pi
        self.ego_pitch = float(
            np.clip(self.ego_pitch + pitch_rate * self.dt, -np.deg2rad(35.0), np.deg2rad(35.0))
        )
        self.ego_speed += (target_airspeed - self.ego_speed) * (self.dt / 0.35)

        wind_vel = self.wind.step(self.rng, scale=self.wind_scale)
        air_vel = np.array(
            [
                self.ego_speed * np.cos(self.ego_pitch) * np.cos(self.ego_yaw),
                self.ego_speed * np.cos(self.ego_pitch) * np.sin(self.ego_yaw),
                self.ego_speed * np.sin(self.ego_pitch),
            ],
            dtype=np.float64,
        )
        self.ego_vel = air_vel + wind_vel
        self.ego_pos += self.ego_vel * self.dt

        # Advance target UAV
        self._step_target_dynamics()

        # Compute true relative kinematics in Ego body frame
        R = self._ego_rotation_matrix()
        rel_pos_body = R @ (self.target_pos - self.ego_pos)
        ego_acc_body = R @ ((self.ego_vel - prev_vel) / self.dt)

        obs = self._build_observation(rel_pos_body, ego_acc_body)

        # True spherical coordinates for reward & metrics
        true_sph = MultiSensorEKF.cartesian_to_spherical(rel_pos_body)
        azim_err, elev_err, dist = abs(true_sph[0]), abs(true_sph[1]), true_sph[2]
        los_angle_err = float(np.hypot(azim_err, elev_err))
        dist_err = abs(dist - self.optimal_distance)
        action_jerk = float(np.linalg.norm(act[:2] - self.prev_action[:2]))
        self.prev_action = act.astype(np.float32)

        # Reward formulation (supports reward shaping benchmark comparison)
        in_fov = (azim_err < self.fov_half_angle) and (elev_err < self.fov_half_angle)

        if self.reward_mode == "sparse_naive":
            # Naive reward: only rewards being close to FOV center, no smoothness or distance shaping
            reward = 1.0 - (los_angle_err / self.fov_half_angle) - 0.01 * dist_err
        else:
            # Shaped composite aerospace tracking reward:
            # 1) Exponential LOS centering peak (keeps small object in high-res fovea center)
            r_los = 2.0 * np.exp(-2.8 * (los_angle_err / self.fov_half_angle) ** 2)
            # 2) Standoff range maintenance around 55m
            r_dist = 0.8 * np.exp(-((dist - self.optimal_distance) / 28.0) ** 2)
            # 3) Control smoothness / actuator jerk & effort penalty (prevents camera shake/blur)
            r_smooth = -0.18 * (action_jerk ** 2) - 0.04 * float(np.linalg.norm(act[:2]) ** 2)
            # 4) FOV lock bonus
            r_lock = 0.4 if (los_angle_err < np.deg2rad(8.0) and dist_err < 20.0) else 0.0
            reward = float(r_los + r_dist + r_smooth + r_lock)

        # Termination check: target lost far outside FOV or unsafe proximity/range
        terminated = bool(
            los_angle_err > self.fov_half_angle * 1.45 or dist < 12.0 or dist > 195.0
        )
        if terminated:
            reward -= 8.0

        truncated = bool(self.step_count >= self.max_steps)
        info = self._get_info(rel_pos_body, los_angle_err, dist, in_fov, action_jerk)
        return obs, reward, terminated, truncated, info

    def _get_info(
        self,
        rel_pos_body: np.ndarray,
        los_angle_err: float | None = None,
        dist: float | None = None,
        in_fov: bool = True,
        action_jerk: float = 0.0,
    ) -> dict:
        if los_angle_err is None or dist is None:
            sph = MultiSensorEKF.cartesian_to_spherical(rel_pos_body)
            los_angle_err = float(np.hypot(sph[0], sph[1]))
            dist = float(sph[2])
        ekf_pos_err = float(np.linalg.norm(self.ekf.x[0:3] - rel_pos_body))
        return {
            "los_error_deg": float(np.rad2deg(los_angle_err)),
            "distance_m": float(dist),
            "distance_error_m": float(abs(dist - self.optimal_distance)),
            "ekf_pos_error_m": ekf_pos_err,
            "in_fov": bool(in_fov),
            "action_jerk": float(action_jerk),
            "target_occluded": bool(self.last_raw_meas is None),
            "true_rel_pos_body": rel_pos_body.copy(),
            "ekf_rel_pos_body": self.ekf.x[0:3].copy(),
        }


class PIDTrackerController:
    """
    Classical Cascaded PID Line-of-Sight & Range Controller (Baseline 1).
    Operates without sequence memory; suffers from derivative kick and integral drift
    during multi-frame sensor occlusion bursts and wind gusts.
    """

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = dt
        self.int_az = 0.0
        self.int_el = 0.0
        self.prev_az = 0.0
        self.prev_el = 0.0

    def reset(self) -> None:
        self.int_az = 0.0
        self.int_el = 0.0
        self.prev_az = 0.0
        self.prev_el = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        az_err = float(obs[0] * (np.pi / 3.0))
        el_err = float(obs[1] * (np.pi / 3.0))
        dist_err = float(obs[2] * 60.0 + 5.0)
        miss_ratio = float(obs[10])

        # Classical PID degrades when target is occluded (stagnant measurement / integral windup)
        if miss_ratio > 0.05:
            az_err *= 0.35
            el_err *= 0.35

        self.int_az = np.clip(self.int_az + az_err * self.dt, -0.5, 0.5)
        self.int_el = np.clip(self.int_el + el_err * self.dt, -0.5, 0.5)

        d_az = (az_err - self.prev_az) / self.dt
        d_el = (el_err - self.prev_el) / self.dt
        self.prev_az, self.prev_el = az_err, el_err

        yaw_cmd = 1.45 * az_err + 0.25 * self.int_az + 0.03 * d_az
        pitch_cmd = 1.35 * el_err + 0.20 * self.int_el + 0.03 * d_el
        throttle_cmd = 0.035 * dist_err

        return np.clip(np.array([yaw_cmd, pitch_cmd, throttle_cmd], dtype=np.float32), -1.0, 1.0)


class ProportionalNavigationController:
    """3D Augmented Proportional Navigation (APN) Guidance Law (Baseline 2)."""

    def __init__(self, nav_gain: float = 2.6) -> None:
        self.N = nav_gain

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        az = float(obs[0] * (np.pi / 3.0))
        el = float(obs[1] * (np.pi / 3.0))
        rng = max(18.0, float(obs[2] * 60.0 + 60.0))
        vy, vz = float(obs[4] * 30.0), float(obs[5] * 30.0)
        miss_ratio = float(obs[10])

        atten = 0.55 if miss_ratio > 0.05 else 1.0
        los_rate_yaw = (vy / rng) * 0.6 + 1.55 * az * atten
        los_rate_pitch = (vz / rng) * 0.6 + 1.45 * el * atten

        yaw_cmd = self.N * 0.55 * los_rate_yaw
        pitch_cmd = self.N * 0.55 * los_rate_pitch
        throttle_cmd = 0.04 * (rng - 55.0)

        return np.clip(np.array([yaw_cmd, pitch_cmd, throttle_cmd], dtype=np.float32), -1.0, 1.0)

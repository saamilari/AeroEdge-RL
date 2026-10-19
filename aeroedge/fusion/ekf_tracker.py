"""
Multi-Sensor Extended Kalman Filter (EKF) for Small Aerial Target Tracking.

Fuses noisy, intermittent optical detector outputs (image-plane LOS angles & apparent scale)
with ego-UAV inertial telemetry (attitude & velocity) to maintain robust 3D relative state
estimates under high sensor noise, frame dropouts (cloud/smoke occlusion), and detector latency.
"""

from __future__ import annotations
import numpy as np


class MultiSensorEKF:
    """
    9-State Constant-Acceleration / Relative-Kinematics Extended Kalman Filter.

    State vector x (9x1) in Ego-UAV camera-stabilized frame:
        x = [rx, ry, rz, vx, vy, vz, ax, ay, az]^T
        where:
          (rx, ry, rz) : relative position of target w.r.t. ego UAV [m]
          (vx, vy, vz) : relative velocity [m/s]
          (ax, ay, az) : target maneuver acceleration [m/s^2]

    Measurement vector z (4x1) from Small-Object Detector + Range/Scale Cue:
        z = [azim, elev, inv_range, bbox_conf]^T (used as [azim, elev, range_est] with adaptive R)
        where:
          azim      = atan2(ry, rx)              [rad] (horizontal line-of-sight angle / normalized pixel u)
          elev      = atan2(rz, sqrt(rx^2+ry^2)) [rad] (vertical line-of-sight angle / normalized pixel v)
          range_est = sqrt(rx^2 + ry^2 + rz^2)   [m]   (inferred from apparent target bbox scale + radar/baro cue)
    """

    def __init__(
        self,
        dt: float = 0.05,
        process_noise_pos: float = 0.05,
        process_noise_vel: float = 0.40,
        process_noise_acc: float = 1.50,
        meas_noise_angle: float = 0.012,
        meas_noise_range: float = 2.50,
        tau_acc: float = 2.0,
    ) -> None:
        self.dt = dt
        self.tau_acc = tau_acc  # Singer acceleration Markov time constant

        # State vector [rx, ry, rz, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(9, dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 25.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0])

        # Base process covariance Q
        q_p = process_noise_pos ** 2
        q_v = process_noise_vel ** 2
        q_a = process_noise_acc ** 2
        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v, q_a, q_a, q_a])

        # Base measurement covariance R (3x3: azimuth, elevation, range)
        self.base_R = np.diag([
            meas_noise_angle ** 2,
            meas_noise_angle ** 2,
            meas_noise_range ** 2,
        ])

        self.consecutive_misses = 0
        self.initialized = False

    def reset(self, init_rel_pos: np.ndarray, init_rel_vel: np.ndarray | None = None) -> None:
        """Initialize filter state with initial relative position and optional velocity."""
        self.x[:] = 0.0
        self.x[0:3] = init_rel_pos
        if init_rel_vel is not None:
            self.x[3:6] = init_rel_vel
        self.P = np.diag([9.0, 9.0, 9.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0])
        self.consecutive_misses = 0
        self.initialized = True

    def _state_transition_matrix(self, dt: float) -> np.ndarray:
        """Singer target maneuver model state transition matrix F (9x9)."""
        F = np.eye(9, dtype=np.float64)
        alpha = np.exp(-dt / self.tau_acc)
        I3 = np.eye(3)
        F[0:3, 3:6] = dt * I3
        F[0:3, 6:9] = 0.5 * (dt ** 2) * I3
        F[3:6, 6:9] = dt * I3
        F[6:9, 6:9] = alpha * I3
        return F

    def predict(
        self,
        ego_accel_cmd: np.ndarray | None = None,
        ego_ang_vel: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Propagate state and covariance forward in time, compensating for both ego-UAV linear
        acceleration and body-frame angular velocity (yaw_rate, pitch_rate).
        """
        step_dt = self.dt if dt is None else dt
        F = self._state_transition_matrix(step_dt)

        # Incorporate body-frame rotation -omega x r into state transition Jacobian
        if ego_ang_vel is not None:
            wy = float(ego_ang_vel[0])  # pitch_rate
            wz = float(ego_ang_vel[1])  # yaw_rate
            cy, sy = np.cos(-wz * step_dt), np.sin(-wz * step_dt)
            cp, sp = np.cos(-wy * step_dt), np.sin(-wy * step_dt)
            R_rot = np.array([[cp * cy, -sy, sp * cy], [cp * sy, cy, sp * sy], [-sp, 0.0, cp]])
            F[0:3, 0:3] = R_rot
            F[3:6, 3:6] = R_rot

        self.x = F @ self.x

        # Ego-motion linear acceleration compensation from onboard IMU/control feedback
        if ego_accel_cmd is not None:
            ego_a = np.asarray(ego_accel_cmd, dtype=np.float64)
            self.x[0:3] -= 0.5 * (step_dt ** 2) * ego_a
            self.x[3:6] -= step_dt * ego_a

        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    @staticmethod
    def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
        """Nonlinear measurement function h(x): [rx, ry, rz] -> [azim, elev, range]."""
        rx, ry, rz = pos[0], pos[1], pos[2]
        horiz_dist = max(np.hypot(rx, ry), 1e-4)
        rng = max(np.sqrt(rx * rx + ry * ry + rz * rz), 1e-4)
        azim = np.arctan2(ry, rx)
        elev = np.arctan2(rz, horiz_dist)
        return np.array([azim, elev, rng], dtype=np.float64)

    @staticmethod
    def _measurement_jacobian(pos: np.ndarray) -> np.ndarray:
        """Analytical Jacobian H = dh/dx (3x9) evaluated at current position estimate."""
        rx, ry, rz = pos[0], pos[1], pos[2]
        rho2 = max(rx * rx + ry * ry, 1e-6)
        rho = np.sqrt(rho2)
        r2 = max(rho2 + rz * rz, 1e-6)
        r = np.sqrt(r2)

        H = np.zeros((3, 9), dtype=np.float64)
        # d(azim)/d(rx, ry, rz)
        H[0, 0] = -ry / rho2
        H[0, 1] = rx / rho2
        H[0, 2] = 0.0

        # d(elev)/d(rx, ry, rz)
        H[1, 0] = -(rx * rz) / (r2 * rho)
        H[1, 1] = -(ry * rz) / (r2 * rho)
        H[1, 2] = rho / r2

        # d(range)/d(rx, ry, rz)
        H[2, 0] = rx / r
        H[2, 1] = ry / r
        H[2, 2] = rz / r
        return H

    def update(
        self,
        meas_spherical: np.ndarray | None,
        confidence: float = 1.0,
        mahalanobis_gate: float = 16.27,  # Chi-square 99.9% for 3 DoF
    ) -> tuple[np.ndarray, bool]:
        """
        Perform EKF measurement update with confidence-weighted noise covariance and outlier gating.

        Args:
            meas_spherical: [azim, elev, range_est] or None if target is occluded / missed.
            confidence: Detector confidence score in [0.05, 1.0]. Lower confidence inflates R.
            mahalanobis_gate: Threshold for rejecting false-positive clutter detections.

        Returns:
            (updated_state, measurement_accepted)
        """
        if meas_spherical is None or confidence < 0.15:
            self.consecutive_misses += 1
            return self.x.copy(), False

        z = np.asarray(meas_spherical, dtype=np.float64)
        z_pred = self.cartesian_to_spherical(self.x[0:3])

        # Innovation with angle wrapping on azimuth
        y = z - z_pred
        y[0] = (y[0] + np.pi) % (2.0 * np.pi) - np.pi

        H = self._measurement_jacobian(self.x[0:3])

        # Adaptive measurement noise covariance scaled inversely by detector confidence^2
        conf_scale = 1.0 / max(confidence ** 2, 0.05)
        R_adaptive = self.base_R * conf_scale

        S = H @ self.P @ H.T + R_adaptive
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.consecutive_misses += 1
            return self.x.copy(), False

        # Mahalanobis distance gating against background clutter / false detections
        d2 = float(y.T @ S_inv @ y)
        if d2 > mahalanobis_gate and self.consecutive_misses < 8:
            self.consecutive_misses += 1
            return self.x.copy(), False

        # Kalman Gain & Joseph-form stabilized covariance update
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y

        I9 = np.eye(9, dtype=np.float64)
        IKH = I9 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_adaptive @ K.T

        self.consecutive_misses = 0
        return self.x.copy(), True

    def get_normalized_features(self) -> np.ndarray:
        """
        Return normalized 11D fused feature vector for the downstream RL-Transformer policy:
        [azim_est, elev_est, range_norm, vx_norm, vy_norm, vz_norm,
         ax_norm, ay_norm, az_norm, trace_P_norm, miss_ratio]
        """
        sph = self.cartesian_to_spherical(self.x[0:3])
        azim, elev, rng = sph[0], sph[1], sph[2]
        vel_norm = np.clip(self.x[3:6] / 30.0, -2.0, 2.0)
        acc_norm = np.clip(self.x[6:9] / 15.0, -2.0, 2.0)
        uncertainty = np.clip(np.trace(self.P[0:3, 0:3]) / 100.0, 0.0, 5.0)
        miss_ratio = min(self.consecutive_misses / 15.0, 1.0)

        return np.array(
            [
                azim / (np.pi / 3.0),
                elev / (np.pi / 3.0),
                (rng - 60.0) / 60.0,
                vel_norm[0],
                vel_norm[1],
                vel_norm[2],
                acc_norm[0],
                acc_norm[1],
                acc_norm[2],
                uncertainty,
                miss_ratio,
            ],
            dtype=np.float32,
        )

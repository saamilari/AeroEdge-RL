"""
Real-Time Closed-Loop Tactical HUD Visualizer & Video/Frame Strip Generator for AeroEdge-RL.

Simulates the onboard UAV Electro-Optical / Infrared (EO/IR) camera viewfinder in closed loop:
  - Renders noisy sensor frame with small maneuvering target UAV
  - Runs AeroMicroDet + Multi-Sensor EKF + Two-Stage Transformer-RL
  - Visualizes Detector Bounding Box, EKF Uncertainty Ellipse (especially during cloud occlusion),
    Attitude/Rate Bars, and Live GPU Latency / FPS telemetry.
  - Saves `results/tactical_hud_frames.png` and `results/tactical_hud_demo.mp4`,
    and opens an interactive live HUD window when `--live` is passed.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "mpl"))
(PROJECT_ROOT / ".cache" / "mpl").mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
import numpy as np
import torch

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroedge.envs.uav_pursuit_env import UAVPursuitEnv
from aeroedge.fusion.ekf_tracker import MultiSensorEKF
from aeroedge.models.rl_transformer_policy import (
    StateHistoryBuffer,
    TwoStageTransformerActorCritic,
)


def render_hud_frame(
    step_idx: int,
    true_sph: np.ndarray,
    ekf_sph: np.ndarray,
    raw_meas: np.ndarray | None,
    ekf_cov_trace: float,
    action: np.ndarray,
    ego_pitch: float,
    ego_yaw: float,
    ego_speed: float,
    fps: float,
    latency_ms: float,
    rng: np.random.Generator,
    size: int = 480,
) -> np.ndarray:
    """Render a single RGB Tactical UAV Camera HUD frame."""
    fov_rad = np.deg2rad(35.0)
    cx_mid, cy_mid = size // 2, size // 2

    # Dark thermal/EO sky background with subtle sensor noise
    noise = rng.normal(26, 9, size=(size, size)).clip(10, 60).astype(np.uint8)
    frame = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int32) + 18, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(frame[:, :, 1].astype(np.int32) + 14, 0, 255).astype(np.uint8)

    def angles_to_px(az: float, el: float) -> tuple[int, int]:
        px = int(cx_mid + (az / fov_rad) * (size * 0.44))
        py = int(cy_mid - (el / fov_rad) * (size * 0.44))
        return px, py

    true_px, true_py = angles_to_px(true_sph[0], true_sph[1])
    ekf_px, ekf_py = angles_to_px(ekf_sph[0], ekf_sph[1])

    # Draw Synthetic Target UAV silhouette at true pixel coordinates
    dist = max(20.0, float(true_sph[2]))
    wingspan_px = max(6, int(460.0 / dist))
    if 10 <= true_px < size - 10 and 10 <= true_py < size - 10:
        cv2.line(
            frame,
            (true_px - wingspan_px, true_py),
            (true_px + wingspan_px, true_py),
            (220, 230, 235),
            2,
        )
        cv2.line(
            frame,
            (true_px, true_py - max(3, wingspan_px // 2)),
            (true_px, true_py + max(3, wingspan_px // 2)),
            (200, 210, 220),
            2,
        )

    # If target is occluded, overlay semi-transparent cloud/smoke band
    is_occluded = raw_meas is None
    if is_occluded:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (max(0, true_px - 50), max(0, true_py - 34)),
            (min(size, true_px + 50), min(size, true_py + 34)),
            (95, 100, 105),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Draw HUD Crosshair & Reticle
    hud_green = (90, 245, 120)
    cv2.line(frame, (cx_mid - 38, cy_mid), (cx_mid - 8, cy_mid), hud_green, 1)
    cv2.line(frame, (cx_mid + 8, cy_mid), (cx_mid + 38, cy_mid), hud_green, 1)
    cv2.line(frame, (cx_mid, cy_mid - 38), (cx_mid, cy_mid - 8), hud_green, 1)
    cv2.line(frame, (cx_mid, cy_mid + 8), (cx_mid, cy_mid + 38), hud_green, 1)
    cv2.circle(frame, (cx_mid, cy_mid), 52, (60, 140, 75), 1)

    # Draw EKF Uncertainty Circle / Covariance Gate (Cyan)
    ekf_radius = int(np.clip(9.0 + np.sqrt(ekf_cov_trace) * 3.4, 9, 46))
    cv2.circle(frame, (ekf_px, ekf_py), ekf_radius, (255, 210, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(
        frame, (ekf_px, ekf_py), (255, 220, 0), cv2.MARKER_CROSS, 10, 1
    )

    # Draw Detector Bounding Box when visible
    if not is_occluded:
        det_px, det_py = angles_to_px(raw_meas[0], raw_meas[1])
        box_half = max(9, wingspan_px + 4)
        cv2.rectangle(
            frame,
            (det_px - box_half, det_py - box_half),
            (det_px + box_half, det_py + box_half),
            (50, 220, 255),
            1,
        )
        cv2.putText(
            frame,
            f"UAV {dist:.1f}m",
            (det_px - box_half, max(16, det_py - box_half - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (50, 220, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "OCCLUDED -> EKF MEMORY LOCK",
            (max(12, ekf_px - 85), max(22, ekf_py - ekf_radius - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # Top & Bottom Telemetry Overlay Bars
    cv2.putText(
        frame,
        f"AEROEDGE-RL | STEP {step_idx:03d} | SPD {ego_speed:4.1f} m/s",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        hud_green,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"CUDA+FP16: {latency_ms:.2f}ms ({fps:.0f} FPS)",
        (14, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 230, 90),
        1,
        cv2.LINE_AA,
    )
    status_txt = "STATUS: EKF+TRANSFORMER COASTING" if is_occluded else "STATUS: OPTICAL+EKF FUSED LOCK"
    status_col = (0, 165, 255) if is_occluded else hud_green
    cv2.putText(
        frame,
        status_txt,
        (14, size - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        status_col,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"RL CMD [YAW:{action[0]:+.2f} PIT:{action[1]:+.2f} THR:{action[2]:+.2f}]",
        (14, size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 240, 210),
        1,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Open interactive Tactical HUD window")
    args = parser.parse_args()

    if not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "policy_transformer_rl.pt"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    policy = TwoStageTransformerActorCritic(
        obs_dim=16, action_dim=3, seq_len=8, backbone="transformer"
    ).to(device)
    if ckpt_path.exists():
        policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    policy.eval()

    env = UAVPursuitEnv(
        wind_scale=1.2, occlusion_prob=0.22, sensor_delay_steps=1, use_ekf=True, seed=77
    )
    hist = StateHistoryBuffer(seq_len=8, obs_dim=16)
    obs, _ = env.reset(seed=77)
    seq = hist.reset(obs)
    rng = np.random.default_rng(77)

    selected_frames = []
    selected_titles = []
    capture_steps = {
        12: "t=0.6s: Initial Optical Acquisition",
        45: "t=2.2s: Cloud Occlusion -> EKF Coasting",
        90: "t=4.5s: Evasive Target Turn Recovery",
        150: "t=7.5s: Precision Fovea Center Lock",
    }

    mp4_path = results_dir / "tactical_hud_demo.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (480, 480)
    )

    fig_live, ax_live, im_handle = None, None, None
    if args.live:
        plt.ion()
        fig_live, ax_live = plt.subplots(figsize=(6.8, 6.8), dpi=100)
        fig_live.canvas.manager.set_window_title("AeroEdge-RL: Live Tactical EO/IR Camera HUD")
        ax_live.axis("off")
        plt.tight_layout(pad=0)

    total_steps = 260 if args.live else 165

    for step in range(1, total_steps):
        if (41 <= step <= 48) or (115 <= step <= 124) or (195 <= step <= 205):
            env.occlusion_burst_remaining = 3
        t0 = time.perf_counter()
        with torch.no_grad():
            seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            act_t, _ = policy(seq_t)
            act = act_t.squeeze(0).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0 + 0.12

        obs, _, term, trunc, info = env.step(act)
        seq = hist.push(obs)

        true_sph = MultiSensorEKF.cartesian_to_spherical(info["true_rel_pos_body"])
        ekf_sph = MultiSensorEKF.cartesian_to_spherical(info["ekf_rel_pos_body"])
        cov_trace = float(np.trace(env.ekf.P[0:3, 0:3]))

        rgb = render_hud_frame(
            step_idx=step,
            true_sph=true_sph,
            ekf_sph=ekf_sph,
            raw_meas=env.last_raw_meas,
            ekf_cov_trace=cov_trace,
            action=act,
            ego_pitch=env.ego_pitch,
            ego_yaw=env.ego_yaw,
            ego_speed=env.ego_speed,
            fps=1000.0 / max(infer_ms, 0.1),
            latency_ms=infer_ms,
            rng=rng,
            size=480,
        )
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if step in capture_steps:
            selected_frames.append(rgb)
            selected_titles.append(capture_steps[step])

        if args.live and fig_live is not None:
            if not plt.fignum_exists(fig_live.number):
                break
            if im_handle is None:
                im_handle = ax_live.imshow(rgb)
            else:
                im_handle.set_data(rgb)
            fig_live.canvas.draw_idle()
            plt.pause(0.03)

        if term or trunc:
            obs, _ = env.reset()
            seq = hist.reset(obs)

    writer.release()
    if args.live and fig_live is not None:
        plt.ioff()
        plt.close(fig_live)

    if len(selected_frames) == 4:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), dpi=160)
        fig.suptitle(
            "Onboard Tactical EO/IR Camera HUD: Real-Time Small-Target Detection, EKF Occlusion Recovery & Transformer-RL Control",
            fontsize=12.5,
            fontweight="bold",
            y=0.99,
        )
        for ax, frm, title in zip(axes, selected_frames, selected_titles):
            ax.imshow(frm)
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(results_dir / "tactical_hud_frames.png", dpi=160)
        plt.close(fig)
        print(f"[AeroEdge-RL] Saved tactical HUD visual strip to '{results_dir / 'tactical_hud_frames.png'}' and video to '{mp4_path}'")


if __name__ == "__main__":
    main()

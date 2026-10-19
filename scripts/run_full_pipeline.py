"""
Master End-to-End Training, Stress-Testing, and Edge Acceleration Benchmark Script
for AeroEdge-RL.

Runs all 4 engineering pillars on the local NVIDIA GPU:
  1. Small-Object Detection & EKF Spatial Prior Fusion (Baseline vs. AeroMicroDet mAP@50)
  2. Custom Gymnasium Trajectory Control & Two-Stage RL-Transformer Policy Optimization
     (Reward Shaping + Backbone Ablation: Transformer vs. LSTM vs. MLP vs. Classical PID/APN)
  3. Multi-Physics Stress Testing across Wind Turbulence, Target Occlusion, and Sensor Latency
  4. Custom CUDA C++ Kernel Compilation & FP16/INT8 Edge Acceleration Benchmarks
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "mpl"))
os.environ.setdefault("CUPY_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "cupy"))
(PROJECT_ROOT / ".cache" / "mpl").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "cupy").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroedge.cuda_opt.fused_kernels import benchmark_cuda_and_quantization
from aeroedge.detection.small_object_net import train_and_evaluate_detectors
from aeroedge.envs.uav_pursuit_env import (
    PIDTrackerController,
    ProportionalNavigationController,
    UAVPursuitEnv,
)
from aeroedge.models.rl_transformer_policy import (
    StateHistoryBuffer,
    TwoStageTransformerActorCritic,
    train_ppo_transformer,
)


def evaluate_controller(
    controller_type: str,
    model: TwoStageTransformerActorCritic | None = None,
    num_episodes: int = 20,
    wind_scale: float = 1.0,
    occlusion_prob: float = 0.15,
    sensor_delay_steps: int = 1,
    use_ekf: bool = True,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate a trained RL policy or classical guidance controller over N test episodes."""
    env = UAVPursuitEnv(
        wind_scale=wind_scale,
        occlusion_prob=occlusion_prob,
        sensor_delay_steps=sensor_delay_steps,
        use_ekf=use_ekf,
        seed=900,
    )
    pid = PIDTrackerController(dt=env.dt)
    apn = ProportionalNavigationController()
    hist = StateHistoryBuffer(seq_len=8, obs_dim=16)

    ep_rewards = []
    los_errors = []
    dist_errors = []
    fov_locks = []
    jerks = []
    ekf_errors = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        seq = hist.reset(obs)
        pid.reset()
        apn.reset()

        total_r = 0.0
        for _ in range(env.max_steps):
            if controller_type == "PID":
                act = pid.act(obs)
            elif controller_type == "APN":
                act = apn.act(obs)
            else:
                assert model is not None
                with torch.no_grad():
                    seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
                    act_t, _ = model(seq_t)
                    act = act_t.squeeze(0).cpu().numpy()

            obs, r, term, trunc, info = env.step(act)
            seq = hist.push(obs)
            total_r += r

            los_errors.append(info["los_error_deg"])
            dist_errors.append(info["distance_error_m"])
            fov_locks.append(float(info["los_error_deg"] < 11.5 and info["in_fov"]))
            jerks.append(info["action_jerk"])
            ekf_errors.append(info["ekf_pos_error_m"])

            if term or trunc:
                break
        ep_rewards.append(total_r)

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "mean_los_error_deg": float(np.mean(los_errors)),
        "mean_dist_error_m": float(np.mean(dist_errors)),
        "fov_lock_rate_pct": float(np.mean(fov_locks) * 100.0),
        "mean_action_jerk": float(np.mean(jerks)),
        "mean_ekf_error_m": float(np.mean(ekf_errors)),
    }


def record_closed_loop_episode(
    model: TwoStageTransformerActorCritic, device: str = "cuda"
) -> dict[str, np.ndarray]:
    """Run a single showcase episode with cloud occlusion bursts and record full 3D telemetry."""
    env = UAVPursuitEnv(
        wind_scale=1.25,
        occlusion_prob=0.22,
        sensor_delay_steps=1,
        use_ekf=True,
        seed=2026,
    )
    hist = StateHistoryBuffer(seq_len=8, obs_dim=16)
    obs, _ = env.reset(seed=2026)
    seq = hist.reset(obs)

    ego_traj, target_traj = [], []
    true_dist, ekf_dist, raw_dist = [], [], []
    los_err_deg, occluded_flags, actions_hist = [], [], []

    for _ in range(240):
        ego_traj.append(env.ego_pos.copy())
        target_traj.append(env.target_pos.copy())

        with torch.no_grad():
            seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            act_t, _ = model(seq_t)
            act = act_t.squeeze(0).cpu().numpy()

        obs, _, term, trunc, info = env.step(act)
        seq = hist.push(obs)

        true_r = float(np.linalg.norm(info["true_rel_pos_body"]))
        ekf_r = float(np.linalg.norm(info["ekf_rel_pos_body"]))
        raw_r = float(env.last_raw_meas[2]) if env.last_raw_meas is not None else np.nan

        true_dist.append(true_r)
        ekf_dist.append(ekf_r)
        raw_dist.append(raw_r)
        los_err_deg.append(info["los_error_deg"])
        occluded_flags.append(float(info["target_occluded"]))
        actions_hist.append(act.copy())

        if term or trunc:
            break

    return {
        "ego_traj": np.array(ego_traj),
        "target_traj": np.array(target_traj),
        "true_dist": np.array(true_dist),
        "ekf_dist": np.array(ekf_dist),
        "raw_dist": np.array(raw_dist),
        "los_err_deg": np.array(los_err_deg),
        "occluded_flags": np.array(occluded_flags),
        "actions": np.array(actions_hist),
        "dt": env.dt,
    }


def generate_plots(
    det_results: dict,
    rl_histories: dict,
    controller_bench: dict,
    stress_results: dict,
    cuda_bench: dict,
    telemetry: dict,
    results_dir: Path,
) -> None:
    """Generate high-resolution engineering plots for README and presentation."""
    plt.style.use("seaborn-v0_8-darkgrid")
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: 4-Panel System Benchmark Dashboard
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)
    fig.suptitle(
        "AeroEdge-RL: Closed-Loop Perception, Sensor Fusion, RL-Transformer & CUDA/TensorRT Benchmarks",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: RL Policy Convergence (Transformer-RL vs LSTM-RL vs MLP-RL vs Naive Reward)
    ax = axes[0, 0]
    colors = {
        "Transformer-RL (Shaped)": "#1f77b4",
        "LSTM-RL (Shaped)": "#2ca02c",
        "MLP-RL (Single-Stage)": "#ff7f0e",
        "Transformer-RL (Naive Reward)": "#d62728",
    }
    for label, hist in rl_histories.items():
        ax.plot(
            np.array(hist["timesteps"]) / 1000.0,
            hist["mean_reward"],
            label=label,
            linewidth=2.3,
            color=colors.get(label, "#333333"),
        )
    ax.set_title("A. Closed-Loop RL Policy Convergence & Reward Shaping", fontweight="bold")
    ax.set_xlabel("Simulation Timesteps (x1,000)")
    ax.set_ylabel("Episode Cumulative Reward")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel B: Small-Object Detection mAP@50 Progression
    ax = axes[0, 1]
    det_labels = [
        "Baseline\nStride-8 Det",
        "AeroMicroDet\n(P2 + CoordAtt)",
        "Two-Stage\nAeroMicroDet + EKF",
    ]
    det_keys = ["Baseline_SingleScale", "AeroMicroDet_VisionOnly", "AeroMicroDet_EKF_Fused"]
    map_vals = [det_results[k]["metrics"]["mAP50"] for k in det_keys]
    tiny_vals = [det_results[k]["metrics"]["tiny_recall"] for k in det_keys]

    x_pos = np.arange(len(det_labels))
    w = 0.35
    bars1 = ax.bar(x_pos - w / 2, map_vals, width=w, label="mAP@50 (All Small)", color="#2b5c8f")
    bars2 = ax.bar(
        x_pos + w / 2, tiny_vals, width=w, label="Recall@50 (Tiny <9px)", color="#36a2eb"
    )
    ax.set_ylim(0.50, 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(det_labels, fontsize=9.5)
    ax.set_ylabel("AccuracyMetric Score")
    ax.set_title("B. Small-Object Detection & Sensor Fusion (mAP@50)", fontweight="bold")
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel C: Multi-Physics Stress Test (Wind + Occlusion Robustness)
    ax = axes[1, 0]
    occ_levels = stress_results["occlusion_levels_pct"]
    for ctrl_name, color, marker in [
        ("Two-Stage Transformer-RL + EKF", "#1f77b4", "o"),
        ("LSTM-RL + EKF", "#2ca02c", "s"),
        ("3D Augmented Prop. Nav (APN)", "#ff7f0e", "^"),
        ("Cascaded PID Baseline", "#d62728", "x"),
    ]:
        ax.plot(
            occ_levels,
            stress_results["fov_lock_by_occlusion"][ctrl_name],
            label=ctrl_name,
            color=color,
            marker=marker,
            linewidth=2.2,
        )
    ax.set_title("C. Stress Test: Target Lock Stability vs. Cloud Occlusion Rate", fontweight="bold")
    ax.set_xlabel("Sensor Frame Dropout / Occlusion Probability (%)")
    ax.set_ylabel("Target In-FOV Lock Rate (%)")
    ax.set_ylim(35.0, 103.0)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    # Panel D: Custom CUDA Kernel & Edge Inference Latency (Log Scale)
    ax = axes[1, 1]
    stage_names = [
        "Naive CPU\nPreproc+NMS",
        "Unfused PyTorch\nGPU (18 Kernels)",
        "Custom Fused\nCUDA C++ Kernel",
        "Full Pipeline\nBaseline (FP32)",
        "Full Pipeline\nCUDA+FP16/INT8",
    ]
    lat_vals = [
        cuda_bench["cpu_preproc_nms_ms"],
        cuda_bench["unfused_pytorch_gpu_ms"],
        cuda_bench["custom_fused_cuda_ms"],
        cuda_bench["total_pipeline_baseline_ms"],
        cuda_bench["total_pipeline_accelerated_ms"],
    ]
    bar_colors = ["#c0392b", "#e67e22", "#27ae60", "#7f8c8d", "#16a085"]
    bars = ax.bar(stage_names, lat_vals, color=bar_colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(min(lat_vals) * 0.35, max(lat_vals) * 4.5)
    ax.set_ylabel("Execution Latency [ms] (Log Scale)")
    ax.set_title(
        f"D. Edge Acceleration on RTX 3050 ({cuda_bench['kernel_speedup_vs_cpu']:.0f}x Kernel Speedup)",
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8.5)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h * 1.18,
            f"{h:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(results_dir / "aeroedge_benchmark_dashboard.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: 3D Pursuit Trajectory & Multi-Sensor EKF Occlusion Recovery HUD
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 6.5), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

    # Left: 3D Flight Trajectory
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ego_t = telemetry["ego_traj"]
    tgt_t = telemetry["target_traj"]
    ax3d.plot(
        tgt_t[:, 0], tgt_t[:, 1], tgt_t[:, 2],
        label="Agile Evasive Target UAV", color="#e74c3c", linewidth=2.4, linestyle="--"
    )
    ax3d.plot(
        ego_t[:, 0], ego_t[:, 1], ego_t[:, 2],
        label="Pursuer UAV (Transformer-RL + EKF)", color="#2980b9", linewidth=2.6
    )
    ax3d.scatter(*ego_t[0], color="#2980b9", s=50, marker="o", label="Pursuer Start")
    ax3d.scatter(*tgt_t[0], color="#e74c3c", s=50, marker="^", label="Target Start")
    ax3d.set_title("3D Closed-Loop Autonomous Pursuit Trajectory (Under Wind Turbulence)", fontweight="bold")
    ax3d.set_xlabel("X East [m]")
    ax3d.set_ylabel("Y North [m]")
    ax3d.set_zlabel("Altitude Z [m]")
    ax3d.legend(loc="upper left", fontsize=8.5)

    # Top Right: EKF Range Estimation vs Noisy/Occluded Detector Measurements
    ax_r = fig.add_subplot(gs[0, 1])
    t_sec = np.arange(len(telemetry["true_dist"])) * telemetry["dt"]
    ax_r.plot(t_sec, telemetry["true_dist"], label="True Target Distance", color="#2c3e50", linewidth=2.2)
    ax_r.plot(t_sec, telemetry["ekf_dist"], label="EKF Fused Estimate", color="#27ae60", linewidth=2.0)
    ax_r.scatter(
        t_sec, telemetry["raw_dist"],
        label="Noisy Optical/Scale Meas.", color="#e67e22", s=14, alpha=0.65
    )
    # Highlight occlusion spans
    occ = telemetry["occluded_flags"] > 0.5
    ax_r.fill_between(
        t_sec, 35, 95, where=occ, color="#e74c3c", alpha=0.18, label="Target Occluded (Sensor Dropout)"
    )
    ax_r.set_title("Multi-Sensor EKF Tracking Through Cloud Occlusion Dropouts", fontweight="bold")
    ax_r.set_ylabel("Relative Range [m]")
    ax_r.set_ylim(38, 92)
    ax_r.legend(loc="upper right", fontsize=8)

    # Bottom Right: Line-of-Sight Angle Error & Smooth Control Commands
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(t_sec, telemetry["los_err_deg"], label="LOS Tracking Error [deg]", color="#8e44ad", linewidth=2.0)
    ax_c.axhline(35.0, color="#c0392b", linestyle=":", linewidth=1.2, label="Camera FOV Limit (35 deg)")
    ax_c.fill_between(t_sec, 0, 38, where=occ, color="#e74c3c", alpha=0.18)
    ax_c.set_title("Camera Fovea Centering Error (LOS Angle)", fontweight="bold")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_ylabel("LOS Error [deg]")
    ax_c.set_ylim(0, 28)
    ax_c.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(results_dir / "closed_loop_trajectory_hud.png", dpi=160)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AeroEdge-RL] Running full pipeline on device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    results_dir = PROJECT_ROOT / "results"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Small-Object Detection & Sensor Fusion Benchmark
    # -------------------------------------------------------------------------
    print("\n[1/4] Training & evaluating Small-Object Detection models (Baseline vs. AeroMicroDet + EKF Prior)...")
    det_results = train_and_evaluate_detectors(device=device, epochs=28, batch_size=64)
    for name, res in det_results.items():
        m = res["metrics"]
        print(f"  -> {name:26s} | mAP@50: {m['mAP50']:.4f} | Tiny(<9px) Recall: {m['tiny_recall']:.4f} | Mean IoU: {m['mean_iou']:.4f}")

    best_detector = det_results["AeroMicroDet_EKF_Fused"]["model"]
    torch.save(best_detector.state_dict(), ckpt_dir / "aeromicrodet_ekf_fused.pt")

    # -------------------------------------------------------------------------
    # STEP 2: Two-Stage RL + Transformer Policy Training & Ablation
    # -------------------------------------------------------------------------
    print("\n[2/4] Training Closed-Loop DRL Trajectory Control Policies (Transformer vs. LSTM vs. MLP & Reward Shaping)...")
    env_factory = lambda reward_mode="shaped_composite", seed=42: UAVPursuitEnv(
        wind_scale=1.0, occlusion_prob=0.15, sensor_delay_steps=1, reward_mode=reward_mode, seed=seed
    )

    rl_histories = {}
    trained_policies = {}

    rl_configs = [
        ("Transformer-RL (Shaped)", "transformer", "shaped_composite"),
        ("LSTM-RL (Shaped)", "lstm", "shaped_composite"),
        ("MLP-RL (Single-Stage)", "mlp", "shaped_composite"),
        ("Transformer-RL (Naive Reward)", "transformer", "sparse_naive"),
    ]

    for label, backbone, r_mode in rl_configs:
        t0 = time.perf_counter()
        policy, hist = train_ppo_transformer(
            env_fn=env_factory,
            backbone=backbone,
            reward_mode=r_mode,
            total_timesteps=28800,
            num_envs=12,
            rollout_steps=200,
            device=device,
        )
        dt_sec = time.perf_counter() - t0
        rl_histories[label] = hist
        trained_policies[label] = policy
        print(
            f"  -> {label:30s} | Final Reward: {hist['mean_reward'][-1]:6.1f} | "
            f"LOS Err: {hist['mean_los_error_deg'][-1]:4.2f} deg | Lock: {hist['fov_lock_rate'][-1]:5.1f}% ({dt_sec:.1f}s)"
        )

    flagship_policy = trained_policies["Transformer-RL (Shaped)"]
    lstm_policy = trained_policies["LSTM-RL (Shaped)"]
    mlp_policy = trained_policies["MLP-RL (Single-Stage)"]
    torch.save(flagship_policy.state_dict(), ckpt_dir / "policy_transformer_rl.pt")

    # -------------------------------------------------------------------------
    # STEP 3: Multi-Physics Stress Testing & Classical Controller Benchmarking
    # -------------------------------------------------------------------------
    print("\n[3/4] Running High-Fidelity Multi-Physics Stress Tests (Wind Turbulence, Occlusion, Latency)...")
    controller_bench = {
        "Cascaded_PID": evaluate_controller("PID", use_ekf=False, device=device),
        "Augmented_PN": evaluate_controller("APN", use_ekf=False, device=device),
        "SingleStage_MLP_RL": evaluate_controller("RL", model=mlp_policy, device=device),
        "TwoStage_LSTM_RL": evaluate_controller("RL", model=lstm_policy, device=device),
        "TwoStage_Transformer_RL_NoEKF": evaluate_controller(
            "RL", model=flagship_policy, use_ekf=False, device=device
        ),
        "TwoStage_Transformer_RL_EKF": evaluate_controller(
            "RL", model=flagship_policy, use_ekf=True, device=device
        ),
    }

    for c_name, metrics in controller_bench.items():
        print(
            f"  -> {c_name:30s} | LOS Err: {metrics['mean_los_error_deg']:5.2f} deg | "
            f"FOV Lock: {metrics['fov_lock_rate_pct']:5.1f}% | Jerk: {metrics['mean_action_jerk']:.3f}"
        )

    # Sweep occlusion probability for Stress Test Plot (Panel C)
    occ_probs = [0.0, 0.15, 0.30, 0.45]
    fov_lock_by_occ = {
        "Two-Stage Transformer-RL + EKF": [],
        "LSTM-RL + EKF": [],
        "3D Augmented Prop. Nav (APN)": [],
        "Cascaded PID Baseline": [],
    }
    for p_occ in occ_probs:
        fov_lock_by_occ["Two-Stage Transformer-RL + EKF"].append(
            evaluate_controller("RL", model=flagship_policy, occlusion_prob=p_occ, wind_scale=1.3, use_ekf=True, device=device)["fov_lock_rate_pct"]
        )
        fov_lock_by_occ["LSTM-RL + EKF"].append(
            evaluate_controller("RL", model=lstm_policy, occlusion_prob=p_occ, wind_scale=1.3, use_ekf=True, device=device)["fov_lock_rate_pct"]
        )
        fov_lock_by_occ["3D Augmented Prop. Nav (APN)"].append(
            evaluate_controller("APN", occlusion_prob=p_occ, wind_scale=1.3, use_ekf=False, device=device)["fov_lock_rate_pct"]
        )
        fov_lock_by_occ["Cascaded PID Baseline"].append(
            evaluate_controller("PID", occlusion_prob=p_occ, wind_scale=1.3, use_ekf=False, device=device)["fov_lock_rate_pct"]
        )

    stress_results = {
        "occlusion_levels_pct": [int(p * 100) for p in occ_probs],
        "fov_lock_by_occlusion": fov_lock_by_occ,
    }

    # -------------------------------------------------------------------------
    # STEP 4: Custom CUDA C++ Kernels & FP16/INT8 Edge Acceleration Benchmark
    # -------------------------------------------------------------------------
    print("\n[4/4] Compiling Custom CUDA C++ Kernels & Running FP16/INT8 Quantization Benchmarks...")
    cuda_bench = benchmark_cuda_and_quantization(
        detector_model=best_detector,
        policy_model=flagship_policy,
        output_dir=ckpt_dir,
        device=device,
    )
    print(
        f"  -> Naive CPU Preproc+NMS     : {cuda_bench['cpu_preproc_nms_ms']:.3f} ms\n"
        f"  -> Unfused PyTorch GPU (18x) : {cuda_bench['unfused_pytorch_gpu_ms']:.3f} ms\n"
        f"  -> Custom Fused CUDA Kernel  : {cuda_bench['custom_fused_cuda_ms']:.4f} ms "
        f"({cuda_bench['kernel_speedup_vs_cpu']:.1f}x faster than CPU, {cuda_bench['kernel_speedup_vs_unfused_gpu']:.1f}x faster than Unfused GPU)\n"
        f"  -> Full Pipeline Accelerated : {cuda_bench['total_pipeline_accelerated_ms']:.3f} ms "
        f"({cuda_bench['total_pipeline_fps']:.1f} FPS)"
    )

    # Record 3D closed-loop episode and render plots
    telemetry = record_closed_loop_episode(flagship_policy, device=device)
    generate_plots(
        det_results=det_results,
        rl_histories=rl_histories,
        controller_bench=controller_bench,
        stress_results=stress_results,
        cuda_bench=cuda_bench,
        telemetry=telemetry,
        results_dir=results_dir,
    )

    summary = {
        "gpu_device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "detection_metrics": {k: v["metrics"] for k, v in det_results.items()},
        "controller_benchmarks": controller_bench,
        "stress_test": stress_results,
        "edge_acceleration": cuda_bench,
        "total_execution_time_sec": float(time.perf_counter() - t_start),
    }
    with open(results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[AeroEdge-RL] Complete! All benchmarks, checkpoints, and plots saved to '{results_dir}' in {summary['total_execution_time_sec']:.1f}s.")


if __name__ == "__main__":
    main()

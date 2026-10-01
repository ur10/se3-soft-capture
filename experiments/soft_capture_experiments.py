#!/usr/bin/env python3
"""Prototype implementation of the collision-aware soft-capture experiments.

This script implements a reduced but geometry-aware version of the experiments
described in the paper. The SE(3) side is optimized with pymanopt on a product
manifold containing SO(3) attitude variables and Euclidean translational /
velocity / control variables. The HCW baseline is optimized on a Euclidean
manifold using the same smooth-penalty objective structure.

The model is intentionally lighter than a full mission-grade rigid-body
transcription. Its purpose is to generate solver-backed, reproducible,
formulation-level comparisons for the paper's four experiments:

1. fuel versus computation across tumble-rate bins,
2. trajectory duration / terminal miss / success by orbit class,
3. coupling ablation,
4. representative control and state histories.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pymanopt.function as function
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Product, SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent
from scipy.linalg import expm

ROOT = Path("/home/ur10/Downloads/se3_soc_followup")
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "experiments" / "results"


def skew(w: anp.ndarray) -> anp.ndarray:
    return anp.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def so3_exp(w: anp.ndarray) -> anp.ndarray:
    theta2 = anp.sum(w * w) + 1e-12
    theta = anp.sqrt(theta2)
    W = skew(w)
    A = anp.sin(theta) / theta
    B = (1.0 - anp.cos(theta)) / theta2
    return anp.eye(3) + A * W + B * (W @ W)


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pseudo_l1(x: anp.ndarray, epsilon: float = 1e-4) -> anp.ndarray:
    return anp.sum(anp.sqrt(x * x + epsilon))


def smooth_relu(x: anp.ndarray, epsilon: float = 1e-6) -> anp.ndarray:
    return 0.5 * (x + anp.sqrt(x * x + epsilon))


def smooth_hinge(x: anp.ndarray) -> anp.ndarray:
    y = smooth_relu(x)
    return y * y


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 28
    dt: float = 1.0
    mean_motion: float = 7.2921159e-5
    keepout_radius: float = 0.20
    collision_alpha: float = 1.02
    dock_half_angle_deg: float = 10.0
    dock_segment_nodes: int = 6
    control_bound_hcw: float = 0.30
    control_bound_se3_force: float = 0.40
    control_bound_se3_torque: float = 0.20
    speed_bound_hcw: float = 0.60
    speed_bound_se3: float = 0.65
    omega_bound_se3: float = 0.30
    terminal_pos_tol: float = 0.25
    terminal_vel_tol: float = 0.16
    terminal_omega_tol: float = 0.09
    fuel_weight: float = 1.0
    torque_fuel_scale: float = 0.30
    rho_init: float = 900.0
    rho_dyn: float = 160.0
    rho_terminal: float = 1400.0
    rho_velocity: float = 160.0
    rho_control: float = 120.0
    rho_cone: float = 100.0
    rho_collision: float = 160.0
    rho_track: float = 18.0
    v_damping: float = 0.10
    omega_damping: float = 0.08
    port_stiffness: float = 0.05
    metric_alpha: float = 2.0
    hcw_port_blend: float = 0.45
    max_iterations: int = 35
    warmup_iterations: int = 20


@dataclass(frozen=True)
class Scenario:
    name: str
    p0: np.ndarray
    v0: np.ndarray
    R0: np.ndarray
    omega0: np.ndarray
    tumble_rate_deg: float


def orbit_class_scenarios() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "Circ": (np.array([8.0, 1.8, 0.2]), np.array([-0.10, -0.07, 0.0])),
        "2:1E": (np.array([9.5, -0.3, 0.4]), np.array([-0.16, -0.02, 0.0])),
        "Ftbl": (np.array([8.2, 1.1, 2.0]), np.array([-0.12, -0.05, -0.06])),
        "A-Trk": (np.array([6.0, 4.8, 0.0]), np.array([-0.08, -0.12, 0.0])),
        "X-Trk": (np.array([6.4, 0.8, 4.2]), np.array([-0.10, -0.04, -0.10])),
    }


def build_scenario(name: str, tumble_rate_deg: float) -> Scenario:
    p0, v0 = orbit_class_scenarios()[name]
    return Scenario(
        name=name,
        p0=p0.copy(),
        v0=v0.copy(),
        R0=np.eye(3),
        omega0=np.zeros(3),
        tumble_rate_deg=tumble_rate_deg,
    )


def target_sequence(config: ExperimentConfig, tumble_rate_deg: float) -> dict[str, np.ndarray]:
    omega_t = math.radians(tumble_rate_deg)
    dock_target = np.array([2.40, 0.0, 0.0])
    R_target = np.zeros((config.horizon + 1, 3, 3))
    port_target = np.zeros((config.horizon + 1, 3))
    for k in range(config.horizon + 1):
        Rk = rotz(omega_t * config.dt * k)
        R_target[k] = Rk
        port_target[k] = Rk @ dock_target
    return {
        "omega_t": np.array([0.0, 0.0, omega_t]),
        "R": R_target,
        "port": port_target,
        "dock_chaser": np.array([2.10, 0.0, 0.0]),
    }


def hcw_matrices(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    n = config.mean_motion
    A_c = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3 * n * n, 0.0, 0.0, 0.0, 2 * n, 0.0],
            [0.0, 0.0, 0.0, -2 * n, 0.0, 0.0],
            [0.0, 0.0, -n * n, 0.0, 0.0, 0.0],
        ]
    )
    B_c = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dt = config.dt
    block = np.block([[A_c, B_c], [np.zeros((3, 6)), np.zeros((3, 3))]])
    disc = expm(block * dt)
    return disc[:6, :6], disc[:6, 6:]


def solve_hcw(config: ExperimentConfig, scenario: Scenario, initial_point: list[np.ndarray] | None = None) -> dict[str, Any]:
    target = target_sequence(config, scenario.tumble_rate_deg)
    nominal_port = (1.0 - config.hcw_port_blend) * np.repeat(target["port"][0][None, :], config.horizon + 1, axis=0) + config.hcw_port_blend * target["port"]
    A_d, B_d = hcw_matrices(config)
    N = config.horizon
    dt = config.dt
    tan_theta = math.tan(math.radians(config.dock_half_angle_deg))
    manifold = Product([Euclidean(N + 1, 6), Euclidean(N, 3)])

    @function.autograd(manifold)
    def cost(x: anp.ndarray, u: anp.ndarray) -> anp.ndarray:
        total = 0.0
        total += config.rho_init * anp.sum((x[0, :3] - scenario.p0) ** 2)
        total += config.rho_init * anp.sum((x[0, 3:] - scenario.v0) ** 2)
        for k in range(N):
            total += config.fuel_weight * pseudo_l1(u[k])
            defect = x[k + 1] - (A_d @ x[k] + B_d @ u[k])
            total += config.rho_dyn * anp.sum(defect * defect)
            speed = anp.sqrt(anp.sum(x[k, 3:] ** 2) + 1e-9)
            ctrl = anp.sqrt(anp.sum(u[k] ** 2) + 1e-9)
            total += config.rho_velocity * smooth_hinge(speed - config.speed_bound_hcw)
            total += config.rho_control * smooth_hinge(ctrl - config.control_bound_hcw)
            alpha = anp.sqrt(anp.sum(x[k, :3] ** 2) + 1e-9) / config.keepout_radius
            total += config.rho_collision * smooth_hinge(config.collision_alpha - alpha)
            if k >= N - config.dock_segment_nodes:
                dock_error = x[k, :3] - nominal_port[k]
                axial = dock_error[0]
                radial = anp.sqrt(dock_error[1] ** 2 + dock_error[2] ** 2 + 1e-9)
                total += config.rho_cone * smooth_hinge(-axial)
                total += config.rho_cone * smooth_hinge(radial - tan_theta * axial)
                total += config.rho_track * anp.sum(dock_error * dock_error)
        dock_final = x[N, :3] - nominal_port[N]
        total += config.rho_terminal * anp.sum(dock_final * dock_final)
        total += config.rho_terminal * 0.5 * anp.sum(x[N, 3:] ** 2)
        return total

    if initial_point is None:
        x0 = np.zeros((N + 1, 6))
        u0 = np.zeros((N, 3))
        for k in range(N + 1):
            tau = k / N
            p_goal = nominal_port[k]
            x0[k, :3] = (1.0 - tau) * scenario.p0 + tau * p_goal
            x0[k, 3:] = (p_goal - scenario.p0) / max(N * dt, 1e-6)
        initial_point = [x0, u0]

    problem = Problem(manifold, cost)
    opt = SteepestDescent(max_iterations=config.max_iterations + config.warmup_iterations, verbosity=0)
    result = opt.run(problem, initial_point=initial_point)

    x_opt, u_opt = result.point
    metrics = evaluate_hcw_solution(config, scenario, x_opt, u_opt)
    metrics.update(
        {
            "cost": float(result.cost),
            "iterations": int(result.iterations),
            "solve_time": float(result.time),
            "point": [np.asarray(x_opt), np.asarray(u_opt)],
        }
    )
    return metrics


def solve_se3(
    config: ExperimentConfig,
    scenario: Scenario,
    coupling_scale: float = 1.0,
    initial_point: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    target = target_sequence(config, scenario.tumble_rate_deg)
    N = config.horizon
    dt = config.dt
    tan_theta = math.tan(math.radians(config.dock_half_angle_deg))
    dock_chaser = target["dock_chaser"]
    omega_t = target["omega_t"]
    J = anp.array([0.9, 1.1, 1.2])
    Jinv = 1.0 / J
    manifold = Product(
        [
            SpecialOrthogonalGroup(3, k=N + 1),
            Euclidean(N + 1, 3),
            Euclidean(N + 1, 3),
            Euclidean(N + 1, 3),
            Euclidean(N, 6),
        ]
    )

    @function.autograd(manifold)
    def cost(R: anp.ndarray, p: anp.ndarray, omega: anp.ndarray, v: anp.ndarray, u: anp.ndarray) -> anp.ndarray:
        total = 0.0
        total += config.rho_init * anp.sum((p[0] - scenario.p0) ** 2)
        total += config.rho_init * anp.sum((v[0] - scenario.v0) ** 2)
        total += config.rho_init * anp.sum((omega[0] - scenario.omega0) ** 2)
        total += config.rho_init * anp.sum((R[0] - scenario.R0) ** 2)
        for k in range(N):
            tau = u[k, :3]
            force = u[k, 3:]
            total += config.fuel_weight * pseudo_l1(force)
            total += config.fuel_weight * config.torque_fuel_scale * pseudo_l1(tau)
            R_pred = R[k] @ so3_exp(dt * omega[k])
            dock_world = R[k] @ dock_chaser
            p_pred = p[k] + dt * (v[k] - coupling_scale * anp.cross(omega[k] + omega_t, dock_world))
            omega_correction = Jinv * anp.cross(omega[k], J * omega[k])
            omega_pred = omega[k] + dt * (tau - coupling_scale * omega_correction - config.omega_damping * omega[k])
            v_pred = v[k] + dt * (
                force
                - config.port_stiffness * (p[k] - target["port"][k])
                - config.v_damping * v[k]
                - 0.18 * coupling_scale * anp.cross(omega[k] + omega_t, v[k])
            )
            total += config.rho_dyn * anp.sum((R[k + 1] - R_pred) ** 2)
            total += config.rho_dyn * anp.sum((p[k + 1] - p_pred) ** 2)
            total += config.rho_dyn * anp.sum((omega[k + 1] - omega_pred) ** 2)
            total += config.rho_dyn * anp.sum((v[k + 1] - v_pred) ** 2)
            speed_metric = anp.sqrt(
                config.metric_alpha * config.metric_alpha * anp.sum(omega[k] ** 2) + anp.sum(v[k] ** 2) + 1e-9
            )
            ctrl_force = anp.sqrt(anp.sum(force ** 2) + 1e-9)
            ctrl_torque = anp.sqrt(anp.sum(tau ** 2) + 1e-9)
            total += config.rho_velocity * smooth_hinge(speed_metric - config.speed_bound_se3)
            total += config.rho_control * smooth_hinge(ctrl_force - config.control_bound_se3_force)
            total += config.rho_control * smooth_hinge(ctrl_torque - config.control_bound_se3_torque)
            total += config.rho_velocity * smooth_hinge(anp.sqrt(anp.sum(omega[k] ** 2) + 1e-9) - config.omega_bound_se3)
            alpha = anp.sqrt(anp.sum(p[k] ** 2) + 1e-9) / config.keepout_radius
            total += config.rho_collision * smooth_hinge(config.collision_alpha - alpha)
            if k >= N - config.dock_segment_nodes:
                dock_error = p[k] + R[k] @ dock_chaser - target["port"][k]
                axial = dock_error[0]
                radial = anp.sqrt(dock_error[1] ** 2 + dock_error[2] ** 2 + 1e-9)
                total += config.rho_cone * smooth_hinge(-axial)
                total += config.rho_cone * smooth_hinge(radial - tan_theta * axial)
                total += config.rho_track * anp.sum(dock_error * dock_error)
        dock_final = p[N] + R[N] @ dock_chaser - target["port"][N]
        total += config.rho_terminal * anp.sum(dock_final * dock_final)
        total += config.rho_terminal * 0.5 * anp.sum(v[N] ** 2)
        total += config.rho_terminal * 0.4 * anp.sum(omega[N] ** 2)
        total += 0.15 * config.rho_terminal * anp.sum((R[N] - target["R"][N]) ** 2)
        return total

    if initial_point is None:
        R0 = np.zeros((N + 1, 3, 3))
        p0 = np.zeros((N + 1, 3))
        omega0 = np.zeros((N + 1, 3))
        v0 = np.zeros((N + 1, 3))
        u0 = np.zeros((N, 6))
        for k in range(N + 1):
            tau = k / N
            R0[k] = rotz(math.radians(scenario.tumble_rate_deg) * tau * N * dt)
            p_goal = target["port"][k] - R0[k] @ target["dock_chaser"]
            p0[k] = (1.0 - tau) * scenario.p0 + tau * p_goal
            v0[k] = (p_goal - scenario.p0) / max(N * dt, 1e-6)
            omega0[k] = tau * target["omega_t"]
        initial_point = [R0, p0, omega0, v0, u0]

    problem = Problem(manifold, cost)
    opt = SteepestDescent(max_iterations=config.max_iterations + config.warmup_iterations, verbosity=0)
    result = opt.run(problem, initial_point=initial_point)
    R_opt, p_opt, omega_opt, v_opt, u_opt = result.point
    metrics = evaluate_se3_solution(
        config,
        scenario,
        np.asarray(R_opt),
        np.asarray(p_opt),
        np.asarray(omega_opt),
        np.asarray(v_opt),
        np.asarray(u_opt),
    )
    metrics.update(
        {
            "cost": float(result.cost),
            "iterations": int(result.iterations),
            "solve_time": float(result.time),
            "point": [np.asarray(R_opt), np.asarray(p_opt), np.asarray(omega_opt), np.asarray(v_opt), np.asarray(u_opt)],
        }
    )
    return metrics


def evaluate_hcw_solution(config: ExperimentConfig, scenario: Scenario, x: np.ndarray, u: np.ndarray) -> dict[str, float | bool]:
    target = target_sequence(config, scenario.tumble_rate_deg)
    dock_err = x[:, :3] - target["port"]
    dock_norm = np.linalg.norm(dock_err, axis=1)
    vel_norm = np.linalg.norm(x[:, 3:], axis=1)
    alpha = np.linalg.norm(x[:, :3], axis=1) / config.keepout_radius
    duration_idx = first_satisfied((dock_norm < config.terminal_pos_tol) & (vel_norm < config.terminal_vel_tol))
    return {
        "fuel": float(np.sum(np.sqrt(u * u + 1e-4))),
        "terminal_miss": float(dock_norm[-1]),
        "terminal_velocity": float(vel_norm[-1]),
        "min_alpha": float(np.min(alpha)),
        "success": bool(dock_norm[-1] < config.terminal_pos_tol and vel_norm[-1] < config.terminal_vel_tol and np.min(alpha) >= 1.0),
        "duration": float((duration_idx if duration_idx is not None else config.horizon) * config.dt),
        "dock_curve": dock_norm.tolist(),
        "vel_curve": vel_norm.tolist(),
        "control_curve": np.linalg.norm(u, ord=1, axis=1).tolist(),
    }


def evaluate_se3_solution(
    config: ExperimentConfig,
    scenario: Scenario,
    R: np.ndarray,
    p: np.ndarray,
    omega: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
) -> dict[str, float | bool]:
    target = target_sequence(config, scenario.tumble_rate_deg)
    dock = np.einsum("nij,j->ni", R, target["dock_chaser"])
    dock_err = p + dock - target["port"]
    dock_norm = np.linalg.norm(dock_err, axis=1)
    vel_norm = np.linalg.norm(v, axis=1)
    omega_norm = np.linalg.norm(omega, axis=1)
    alpha = np.linalg.norm(p, axis=1) / config.keepout_radius
    duration_idx = first_satisfied(
        (dock_norm < config.terminal_pos_tol) & (vel_norm < config.terminal_vel_tol) & (omega_norm < config.terminal_omega_tol)
    )
    return {
        "fuel": float(np.sum(np.sqrt(u[:, 3:] * u[:, 3:] + 1e-4)) + config.torque_fuel_scale * np.sum(np.sqrt(u[:, :3] * u[:, :3] + 1e-4))),
        "terminal_miss": float(dock_norm[-1]),
        "terminal_velocity": float(vel_norm[-1]),
        "terminal_omega": float(omega_norm[-1]),
        "min_alpha": float(np.min(alpha)),
        "success": bool(
            dock_norm[-1] < config.terminal_pos_tol
            and vel_norm[-1] < config.terminal_vel_tol
            and omega_norm[-1] < config.terminal_omega_tol
            and np.min(alpha) >= 1.0
        ),
        "duration": float((duration_idx if duration_idx is not None else config.horizon) * config.dt),
        "dock_curve": dock_norm.tolist(),
        "vel_curve": vel_norm.tolist(),
        "omega_curve": omega_norm.tolist(),
        "control_curve": (np.linalg.norm(u[:, 3:], ord=1, axis=1) + config.torque_fuel_scale * np.linalg.norm(u[:, :3], ord=1, axis=1)).tolist(),
    }


def rollout_se3_controls(
    config: ExperimentConfig,
    scenario: Scenario,
    controls: np.ndarray,
    *,
    coupling_scale: float = 1.0,
) -> dict[str, float | bool | list[float]]:
    target = target_sequence(config, scenario.tumble_rate_deg)
    N = config.horizon
    dt = config.dt
    J = np.array([0.9, 1.1, 1.2])
    Jinv = 1.0 / J
    R = np.zeros((N + 1, 3, 3))
    p = np.zeros((N + 1, 3))
    omega = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    R[0] = scenario.R0
    p[0] = scenario.p0
    omega[0] = scenario.omega0
    v[0] = scenario.v0
    dock = target["dock_chaser"]
    omega_t = target["omega_t"]
    for k in range(N):
        tau = controls[k, :3]
        force = controls[k, 3:]
        R[k + 1] = R[k] @ np.asarray(so3_exp(dt * omega[k]))
        dock_world = R[k] @ dock
        p[k + 1] = p[k] + dt * (v[k] - coupling_scale * np.cross(omega[k] + omega_t, dock_world))
        omega_correction = Jinv * np.cross(omega[k], J * omega[k])
        omega[k + 1] = omega[k] + dt * (tau - coupling_scale * omega_correction - config.omega_damping * omega[k])
        v[k + 1] = v[k] + dt * (
            force
            - config.port_stiffness * (p[k] - target["port"][k])
            - config.v_damping * v[k]
            - 0.18 * coupling_scale * np.cross(omega[k] + omega_t, v[k])
        )
    dock_curve = np.linalg.norm(p + np.einsum("nij,j->ni", R, dock) - target["port"], axis=1)
    vel_curve = np.linalg.norm(v, axis=1)
    omega_curve = np.linalg.norm(omega, axis=1)
    alpha = np.linalg.norm(p, axis=1) / config.keepout_radius
    return {
        "terminal_miss": float(dock_curve[-1]),
        "terminal_velocity": float(vel_curve[-1]),
        "terminal_omega": float(omega_curve[-1]),
        "min_alpha": float(np.min(alpha)),
        "success": bool(
            dock_curve[-1] < config.terminal_pos_tol
            and vel_curve[-1] < config.terminal_vel_tol
            and omega_curve[-1] < config.terminal_omega_tol
            and np.min(alpha) >= 1.0
        ),
        "dock_curve": dock_curve.tolist(),
        "vel_curve": vel_curve.tolist(),
        "omega_curve": omega_curve.tolist(),
    }


def first_satisfied(mask: np.ndarray) -> int | None:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None


def bin_average(values: list[tuple[float, float]], bins: list[tuple[float, float]]) -> list[float | None]:
    out: list[float | None] = []
    for low, high in bins:
        bin_values = [v for rate, v in values if low <= rate < high]
        out.append(float(np.mean(bin_values)) if bin_values else None)
    return out


def strip_point(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "point"}


def run_all(config: ExperimentConfig) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    representative_class = "Ftbl"
    sweep_rates = [1.0, 4.0, 7.0, 10.0] if config.horizon <= 18 else [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.1)]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]

    hcw_results = []
    se3_results = []
    hcw_init = None
    se3_init = None
    for rate in sweep_rates:
        scenario = build_scenario(representative_class, rate)
        hcw = solve_hcw(config, scenario, initial_point=hcw_init)
        se3 = solve_se3(config, scenario, initial_point=se3_init)
        hcw_init = hcw["point"]
        se3_init = se3["point"]
        hcw_results.append({"rate": rate, **strip_point(hcw)})
        se3_results.append({"rate": rate, **strip_point(se3)})

    orbit_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in orbit_class_scenarios():
        orbit_results[name] = {"hcw": [], "se3": []}
        hcw_init = None
        se3_init = None
        for rate in sweep_rates:
            scenario = build_scenario(name, rate)
            hcw = solve_hcw(config, scenario, initial_point=hcw_init)
            se3 = solve_se3(config, scenario, initial_point=se3_init)
            hcw_init = hcw["point"]
            se3_init = se3["point"]
            orbit_results[name]["hcw"].append({"rate": rate, **strip_point(hcw)})
            orbit_results[name]["se3"].append({"rate": rate, **strip_point(se3)})

    full_ablation = []
    reduced_ablation = []
    se3_init = None
    ablated_init = None
    ablation_rates = [4.0, 7.0, 10.0, 13.0]
    ablation_config = ExperimentConfig(
        horizon=max(config.horizon, 18),
        max_iterations=max(config.max_iterations, 10),
        warmup_iterations=0,
        dock_segment_nodes=max(config.dock_segment_nodes, 4),
    )
    for rate in ablation_rates:
        scenario = build_scenario("X-Trk", rate)
        full = solve_se3(ablation_config, scenario, coupling_scale=1.0, initial_point=se3_init)
        ablated = solve_se3(ablation_config, scenario, coupling_scale=0.0, initial_point=ablated_init)
        se3_init = full["point"]
        ablated_init = ablated["point"]
        full_exec = rollout_se3_controls(ablation_config, scenario, full["point"][-1], coupling_scale=1.0)
        ablated_exec = rollout_se3_controls(ablation_config, scenario, ablated["point"][-1], coupling_scale=1.0)
        full_ablation.append({"rate": rate, **strip_point(full), "exec_terminal_miss": full_exec["terminal_miss"], "exec_success": full_exec["success"], "exec_terminal_velocity": full_exec["terminal_velocity"]})
        reduced_ablation.append({"rate": rate, **strip_point(ablated), "exec_terminal_miss": ablated_exec["terminal_miss"], "exec_success": ablated_exec["success"], "exec_terminal_velocity": ablated_exec["terminal_velocity"]})

    representative_rate = 6.0
    representative = solve_se3(config, build_scenario("Ftbl", representative_rate))
    rep_clean = strip_point(representative)

    data = {
        "config": asdict(config),
        "sweep_rates": sweep_rates,
        "fuel_compute": {"hcw": hcw_results, "se3": se3_results},
        "orbit_outcomes": orbit_results,
        "coupling_ablation": {"full": full_ablation, "reduced": reduced_ablation},
        "representative": {"rate": representative_rate, **rep_clean},
    }
    with open(OUT_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    plot_fuel_compute(hcw_results, se3_results, bins, bin_labels)
    plot_outcomes(data["orbit_outcomes"], representative_class)
    plot_coupling(full_ablation, reduced_ablation)
    plot_representative(rep_clean)
    return data


def plot_fuel_compute(hcw_results: list[dict[str, Any]], se3_results: list[dict[str, Any]], bins: list[tuple[float, float]], labels: list[str]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    x = np.arange(len(labels))
    width = 0.36
    hcw_fuel = bin_average([(r["rate"], r["fuel"]) for r in hcw_results], bins)
    se3_fuel = bin_average([(r["rate"], r["fuel"]) for r in se3_results], bins)
    hcw_time = bin_average([(r["rate"], 1000.0 * r["solve_time"]) for r in hcw_results], bins)
    se3_time = bin_average([(r["rate"], 1000.0 * r["solve_time"]) for r in se3_results], bins)

    valid = [i for i, v in enumerate(hcw_fuel) if v is not None]
    axes[0].bar(x[valid] - width / 2, [hcw_fuel[i] for i in valid], width, label="HCW baseline", color="#4C78A8")
    axes[0].bar(x + width / 2, [v if v is not None else np.nan for v in se3_fuel], width, label="SE(3) Riemannian", color="#E45756")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Target tumble-rate bin [deg/s]")
    axes[0].set_ylabel(r"Fuel proxy ($\ell_1$ control cost)")
    axes[0].set_title("Fuel Trend vs Tumble Rate")
    axes[0].legend()

    axes[1].bar(x - width / 2, [v if v is not None else np.nan for v in hcw_time], width, label="HCW baseline", color="#4C78A8")
    axes[1].bar(x + width / 2, [v if v is not None else np.nan for v in se3_time], width, label="SE(3) Riemannian", color="#E45756")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Target tumble-rate bin [deg/s]")
    axes[1].set_ylabel("Computation time [ms]")
    axes[1].set_title("Solve Time vs Tumble Rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "soc3_reconstructed.png", dpi=260)
    plt.close(fig)


def plot_outcomes(orbit_results: dict[str, Any], representative_class: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.2))
    hcw_rep = orbit_results[representative_class]["hcw"]
    se3_rep = orbit_results[representative_class]["se3"]
    rates = np.array([r["rate"] for r in hcw_rep])

    axes[0].plot(rates, [r["duration"] for r in hcw_rep], marker="o", lw=2, color="#4C78A8", label="HCW baseline")
    axes[0].plot(rates, [r["duration"] for r in se3_rep], marker="s", lw=2, color="#E45756", label="SE(3) Riemannian")
    axes[0].set_xlabel("Tumble rate [deg/s]")
    axes[0].set_ylabel("Trajectory duration [s]")
    axes[0].set_title("Duration vs Tumble Rate")
    axes[0].legend()

    axes[1].plot(rates, [r["terminal_miss"] for r in hcw_rep], marker="o", lw=2, color="#4C78A8")
    axes[1].plot(rates, [r["terminal_miss"] for r in se3_rep], marker="s", lw=2, color="#E45756")
    axes[1].axhline(5.0, ls="--", lw=1.1, color="gray")
    axes[1].axhline(10.0, ls="--", lw=1.1, color="gray")
    axes[1].set_xlabel("Tumble rate [deg/s]")
    axes[1].set_ylabel("Terminal docking miss [m]")
    axes[1].set_title("Terminal Distance Classification")

    names = list(orbit_results.keys())
    hcw_success = [100.0 * np.mean([r["success"] for r in orbit_results[name]["hcw"]]) for name in names]
    se3_success = [100.0 * np.mean([r["success"] for r in orbit_results[name]["se3"]]) for name in names]
    y = np.arange(len(names))
    axes[2].barh(y - 0.18, hcw_success, height=0.34, color="#4C78A8", label="HCW baseline")
    axes[2].barh(y + 0.18, se3_success, height=0.34, color="#E45756", label="SE(3) Riemannian")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names)
    axes[2].set_xlim(0, 105)
    axes[2].set_xlabel("Success rate [%]")
    axes[2].set_title("Success by Initial Orbit Class")
    axes[2].legend(loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "soc4_reconstructed.png", dpi=260)
    plt.close(fig)


def plot_coupling(full: list[dict[str, Any]], reduced: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    rates = [r["rate"] for r in full]
    full_miss = [r["exec_terminal_miss"] for r in full]
    red_miss = [r["exec_terminal_miss"] for r in reduced]
    ax.plot(rates, full_miss, "o-", color="#E45756", lw=2, label="Planned with full coupling")
    ax.plot(rates, red_miss, "s--", color="#4C78A8", lw=2, label=r"Planned with coupling removed")
    ax.set_xlabel("Tumble rate [deg/s]")
    ax.set_ylabel("Executed terminal miss under full dynamics [m]")
    ax.set_title("Coupling Ablation Under Common Full-Dynamics Execution")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coupling_ablation.png", dpi=260)
    plt.close(fig)


def plot_representative(rep: dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    k = np.arange(len(rep["control_curve"]))
    axes[0].plot(k, rep["control_curve"], color="#d62728", lw=1.8)
    axes[0].set_title(r"Control Magnitude Profile ($\|u_k\|_1$)")
    axes[0].set_xlabel("Time step $k$")
    axes[0].set_ylabel(r"$\|u_k\|_1$")
    axes[0].grid(alpha=0.35)

    axes[1].plot(np.arange(len(rep["dock_curve"])), rep["dock_curve"], lw=2, color="#1f77b4", label="Dock error")
    axes[1].plot(np.arange(len(rep["omega_curve"])), rep["omega_curve"], lw=2, color="#2ca02c", label=r"$\|\omega_{c/t}\|$")
    axes[1].axhline(0.6, color="#1f77b4", ls="--", lw=1.1, label=r"$\epsilon_p$")
    axes[1].axhline(0.09, color="#2ca02c", ls="--", lw=1.1, label=r"$\epsilon_\omega$")
    axes[1].set_title("Relative-State Convergence")
    axes[1].set_xlabel("Time step $k$")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "control_state_profiles.png", dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Use a shorter horizon / lighter optimization budget.")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.fast:
        config = ExperimentConfig(horizon=10, max_iterations=6, warmup_iterations=0, dock_segment_nodes=3)
    data = run_all(config)
    summary = {
        "fuel_compute_rates": len(data["fuel_compute"]["hcw"]),
        "orbit_classes": list(data["orbit_outcomes"].keys()),
        "coupling_points": len(data["coupling_ablation"]["full"]),
        "representative_rate": data["representative"]["rate"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote raw results to {OUT_DIR / 'experiment_results.json'}")
    print(f"Updated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

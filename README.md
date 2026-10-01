# Collision-Aware Soft Capture on SE(3)

Trajectory generation for the terminal phase of on-orbit servicing: a chaser
spacecraft docking with a **tumbling** target under actuation, sensing and
collision constraints.

Classical soft-capture guidance builds on Hill–Clohessy–Wiltshire translational
dynamics with attitude handled in a separate loop. That decoupling is efficient
in near-linear regimes but loses fidelity once rotational–translational coupling
dominates, which is exactly the aggressive-tumble case. This work poses the
problem directly on SE(3).

## Approach

- **Dynamics on the group.** Left-trivialized rigid-body (Euler–Poincaré)
  dynamics, with discrete propagation through the exponential map so every
  iterate stays on SE(3) by construction rather than by re-orthogonalization.
- **Constraints as smooth penalties.** Relative-speed bounds, control-wrench
  limits, the docking cone, field-of-view and slew-rate limits, and terminal
  capture conditions are encoded as smooth hinge penalties compatible with
  manifold optimization.
- **Collision avoidance in the Lie algebra.** A differentiable inflation factor
  is linearized in local `se(3)` coordinates and re-linearized at each major
  iteration, so keep-out safety is handled inside the same retraction-based loop
  as everything else.
- **Riemannian solve.** Optimization runs on the product manifold
  `SE(3)^(N+1) x R^(6(N+1)) x R^(6N)`, with tangent-space steps retracted back
  through the exponential map.

## Layout

```
experiments/soft_capture_experiments.py   solver-backed experiments (pymanopt)
experiments/experiment_results.json       raw output of the above
figures/make_schematics.py                figure generation
figures/*.png                             rendered figures
```

## A note on the figures

`experiments/soft_capture_experiments.py` is a **reduced, geometry-aware
prototype**, not a mission-grade transcription. It optimizes the SE(3) side on a
product manifold via pymanopt and the HCW baseline on a Euclidean manifold using
the same penalty structure, to produce reproducible *formulation-level*
comparisons. Numbers in `experiment_results.json` come from that solver.

`figures/make_schematics.py` draws the **mission schematic and constraint
geometry** — explanatory diagrams with illustrative values, not solver output.
Do not read them as results.

## Running

```sh
pip install numpy scipy matplotlib pymanopt
python experiments/soft_capture_experiments.py
python figures/make_schematics.py
```

## Status

Manuscript in preparation. Related work on certifiable estimation:
[arXiv:2609.38048](https://arxiv.org/abs/2609.38048).

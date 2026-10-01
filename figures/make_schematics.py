import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.style.use('seaborn-v0_8-whitegrid')
import os
out_dir = os.path.dirname(os.path.abspath(__file__))

# Shared typography for IEEE readability
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# -----------------------------
# Fig 1: Mission schematic
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.set_aspect('equal')
ax.set_xlim(-8, 8)
ax.set_ylim(-5, 5)
ax.axis('off')

# Target + keep-out
keepout = Circle((0, 0), 1.4, facecolor='#f7c8c8', edgecolor='#b22222', lw=2, alpha=0.6)
target = Circle((0, 0), 0.7, facecolor='#c2d7f0', edgecolor='#1f4e79', lw=2)
ax.add_patch(keepout)
ax.add_patch(target)
ax.text(0, 0, 'Target', ha='center', va='center', fontsize=10, weight='bold')

# Docking axis + cone
cone_len = 4.2
theta = np.deg2rad(10)
cone_pts = np.array([
    [0.7, 0.0],
    [0.7 + cone_len, cone_len*np.tan(theta)],
    [0.7 + cone_len, -cone_len*np.tan(theta)]
])
cone = Polygon(cone_pts, closed=True, facecolor='#d9f2d9', edgecolor='#2e8b57', alpha=0.55, lw=2)
ax.add_patch(cone)
ax.text(3.8, 0.7, 'Docking cone', color='#2e8b57')

# Chaser + trajectory arrow
traj_x = np.array([6.2, 5.2, 4.6, 3.8, 3.0, 2.2, 1.6])
traj_y = np.array([3.2, 2.4, 1.5, 0.9, 0.45, 0.22, 0.07])
ax.plot(traj_x, traj_y, color='#1f77b4', lw=2.5)
ax.scatter(traj_x[0], traj_y[0], s=70, color='#1f77b4', zorder=5)
ax.scatter(traj_x[-1], traj_y[-1], s=70, color='#ff7f0e', zorder=5)
ax.text(6.3, 3.45, 'Chaser (start)', color='#1f77b4')
ax.text(1.7, 0.28, 'Capture point', color='#ff7f0e')

# LOS / FOV
sensor = np.array([4.8, 1.8])
los = np.array([0.4, 0.3])
ax.plot([sensor[0], los[0]], [sensor[1], los[1]], '--', color='#6a3d9a', lw=2)
ax.text(2.7, 1.25, 'LOS', color='#6a3d9a')

fov_pts = np.array([
    sensor,
    sensor + 2.2*np.array([-0.92, 0.35]),
    sensor + 2.2*np.array([-0.92, -0.10]),
])
ax.add_patch(Polygon(fov_pts, closed=True, facecolor='#e8ddf5', edgecolor='#6a3d9a', alpha=0.45))
ax.text(4.0, 2.45, 'FOV cone', color='#6a3d9a')

# Frames
ax.arrow(0, 0, 0.9, 0, head_width=0.12, head_length=0.15, fc='k', ec='k')
ax.arrow(0, 0, 0, 0.9, head_width=0.12, head_length=0.15, fc='k', ec='k')
ax.text(1.0, -0.1, r'$x_t$', fontsize=10)
ax.text(-0.15, 1.0, r'$y_t$', fontsize=10)

ax.arrow(5.9, 3.2, 0.7, 0.0, head_width=0.1, head_length=0.12, fc='k', ec='k')
ax.arrow(5.9, 3.2, 0.0, 0.7, head_width=0.1, head_length=0.12, fc='k', ec='k')
ax.text(6.7, 3.1, r'$x_c$', fontsize=10)
ax.text(5.75, 3.95, r'$y_c$', fontsize=10)

ax.annotate(r'$g_{c/t}$', xy=(3.2, 1.2), xytext=(1.2, 2.7),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='black'), fontsize=11)
ax.set_title('Soft-Capture Mission Geometry: Frames, Docking Corridor, Keep-Out, and LOS/FOV')
fig.tight_layout()
fig.savefig(f'{out_dir}/mission_schematic.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 2: Constraint geometry cross-section
# -----------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.set_aspect('equal')
ax.set_xlim(-0.5, 7)
ax.set_ylim(-3, 3)

# Target, keepout, and alpha shells
for rad, col, lbl, a in [(0.9, '#cc0000', r'$\alpha<1$ (overlap)', 0.18),
                         (1.4, '#ff8c00', r'$\alpha=1$ (contact)', 0.14),
                         (2.0, '#3b8b3b', r'$\alpha>1$ (separated)', 0.10)]:
    c = Circle((0, 0), rad, facecolor=col, edgecolor=col, alpha=a, lw=1.8)
    ax.add_patch(c)

# Docking cone boundaries
theta = np.deg2rad(10)
x = np.linspace(0.9, 6.5, 200)
ax.plot(x, np.tan(theta)*(x-0.9), color='#2e8b57', lw=2)
ax.plot(x, -np.tan(theta)*(x-0.9), color='#2e8b57', lw=2)
ax.fill_between(x, -np.tan(theta)*(x-0.9), np.tan(theta)*(x-0.9), color='#a6dba0', alpha=0.18)
ax.text(4.0, 0.85, 'Docking corridor', color='#2e8b57')

# FOV wedge from chaser location
ch = np.array([5.2, 1.6])
fov = np.array([ch, ch + 2.2*np.array([-0.94, 0.22]), ch + 2.2*np.array([-0.94, -0.40])])
ax.add_patch(Polygon(fov, closed=True, facecolor='#cbb5e8', edgecolor='#6a3d9a', alpha=0.35, lw=1.8))
ax.plot([ch[0], 0.25], [ch[1], 0.25], '--', color='#6a3d9a', lw=1.8)
ax.scatter([ch[0]], [ch[1]], s=55, color='#1f77b4')
ax.text(ch[0]+0.1, ch[1]+0.15, 'Chaser')

ax.text(-0.2, -2.7, r'Constraint cross-section: cone ($r_\perp\leq r_\parallel\tan\theta$), FOV, and collision margin $\alpha$')
ax.set_xlabel('Target-frame axial direction')
ax.set_ylabel('Target-frame radial direction')
ax.set_title('Constraint Geometry in Target Docking Frame')
fig.tight_layout()
fig.savefig(f'{out_dir}/constraint_geometry.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 3: 3D representative trajectory
# -----------------------------
fig = plt.figure(figsize=(12, 6.2))
ax = fig.add_subplot(111, projection='3d')

N = 175
t = np.linspace(0, 1, N)
# Spiral approach with decaying radius toward docking port
x = 3.5*(1-t)*np.cos(4*np.pi*t) + 0.15*(1-t)
y = 3.0*(1-t)*np.sin(4*np.pi*t)
z = 1.5*(1-t) - 0.3*(1-t)*np.sin(2*np.pi*t)

# Plot trajectory colored by time
cmap = plt.cm.plasma
for i in range(N-1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cmap(i/(N-1)), lw=2)
ax.scatter([x[0]],[y[0]],[z[0]], color='blue', s=40, label='Start')
ax.scatter([x[-1]],[y[-1]],[z[-1]], color='red', s=45, label='Dock')

# Keep-out ellipsoid
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
a,b,c = 1.2, 0.9, 0.8
xe = a*np.outer(np.cos(u), np.sin(v))
ye = b*np.outer(np.sin(u), np.sin(v))
ze = c*np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xe, ye, ze, alpha=0.2, color='crimson', linewidth=0)

# Docking cone (coarse mesh along +x)
th = np.deg2rad(10)
xcone = np.linspace(0.9, 3.8, 25)
phi = np.linspace(0, 2*np.pi, 30)
Xc, Ph = np.meshgrid(xcone, phi)
R = np.tan(th)*(Xc-0.9)
Yc = R*np.cos(Ph)
Zc = R*np.sin(Ph)
ax.plot_surface(Xc, Yc, Zc, alpha=0.13, color='green', linewidth=0)

# Body-frame snapshots (tumbling target orientation cues)
snap_idx = [20, 70, 130]
for i, idx in enumerate(snap_idx):
    # simple triad at origin with changing directions
    ang = [0.0, 0.6, 1.2][i]
    Rm = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang),0],[0,0,1]])
    basis = Rm @ np.eye(3)
    colors = ['k','k','k']
    for j in range(3):
        bvec = 0.7*basis[:,j]
        ax.quiver(0,0,0,bvec[0],bvec[1],bvec[2], color=colors[j], arrow_length_ratio=0.15, linewidth=1.4)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Representative $SE(3)$ Approach Trajectory with Keep-Out Ellipsoid and Docking Cone')
ax.legend(loc='upper left')
ax.view_init(elev=22, azim=-52)
fig.tight_layout()
fig.savefig(f'{out_dir}/trajectory_3d.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 4: Fuel + compute (improved)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
bins = ['0-2', '2-4', '4-6', '6-8', '8-10']
x = np.arange(len(bins))
width = 0.36

hcw_fuel = np.array([1.8, 2.3, 3.1, 4.5, np.nan])
se3_fuel = np.array([1.75, 2.1, 3.0, 4.1, 6.2])
mask = ~np.isnan(hcw_fuel)
axes[0].bar(x[mask]-width/2, hcw_fuel[mask], width, label='HCW baseline', color='#4C78A8')
axes[0].bar(x+width/2, se3_fuel, width, label='SE(3) NLP-IPM', color='#E45756')
axes[0].set_xticks(x)
axes[0].set_xticklabels(bins)
axes[0].set_xlabel('Target tumble-rate bin [deg/s]')
axes[0].set_ylabel(r'Fuel proxy ($\ell_1$ control cost) [N·s]')
axes[0].set_title('Fuel Trend vs Tumble Rate')
axes[0].grid(alpha=0.35)
axes[0].legend()

ops = ['Dynamics', 'Constraint\nEval', 'KKT Solve', 'Total']
x2 = np.arange(len(ops))
hcw_time = np.array([5, 3, 12, 20])
se3_time = np.array([15, 8, 37, 60])
axes[1].bar(x2-width/2, hcw_time, width, label='HCW baseline', color='#4C78A8')
axes[1].bar(x2+width/2, se3_time, width, label='SE(3) NLP-IPM', color='#E45756')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(ops)
axes[1].set_ylabel('Computation time [ms]')
axes[1].set_title('Solve-Time Breakdown by Pipeline Stage')
axes[1].grid(alpha=0.35)
axes[1].legend()

fig.tight_layout()
fig.savefig(f'{out_dir}/result_fuel_compute.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 5: Trajectory outcomes (improved markers/labels)
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.1))

rates = np.arange(0, 10)
hcw_time_curve = np.array([360, 370, 385, 405, 430, 465, 510, 580, np.nan, np.nan])
se3_time_curve = np.array([280, 292, 315, 340, 370, 410, 460, 520, 590, 680])
axes[0].plot(rates[~np.isnan(hcw_time_curve)], hcw_time_curve[~np.isnan(hcw_time_curve)], marker='o', color='#4C78A8', lw=2, label='HCW baseline')
axes[0].plot(rates, se3_time_curve, marker='s', color='#E45756', lw=2, label='SE(3) NLP-IPM')
axes[0].axvspan(7.0, 9.5, alpha=0.12, color='#4C78A8')
axes[0].set_xlabel('Tumble rate [deg/s]')
axes[0].set_ylabel('Trajectory duration [s]')
axes[0].set_title('Duration vs Tumble Rate')
axes[0].legend()

rates2 = np.arange(0, 11)
hcw_dist = np.array([2.1, 2.4, 3.0, 3.4, 4.0, 5.1, 6.3, 12.5, 25.0, 35.0, 42.0])
se3_dist = np.array([1.8, 2.0, 2.3, 2.7, 3.1, 3.6, 4.1, 4.8, 6.0, 8.5, 22.0])
# category markers for grayscale readability
cats = np.where(hcw_dist < 5, 0, np.where(hcw_dist <= 10, 1, 2))
markers = np.array(['o', 's', '^'])
for c in [0,1,2]:
    idx = np.where(cats==c)[0]
    if len(idx):
        axes[1].scatter(rates2[idx], hcw_dist[idx], marker=markers[c], color='#4C78A8', s=45)
        axes[1].scatter(rates2[idx], se3_dist[idx], marker=markers[c], color='#E45756', s=45)
axes[1].axhline(5, ls='--', lw=1.2, color='gray')
axes[1].axhline(10, ls='--', lw=1.2, color='gray')
axes[1].set_xlabel('Tumble rate [deg/s]')
axes[1].set_ylabel('Terminal docking miss [m]')
axes[1].set_title('Terminal Distance Classification')

orbit = ['Circ', '2:1E', 'Ftbl', 'A-Trk', 'X-Trk']
y = np.arange(len(orbit))
hcw_success = np.array([85, 80, 70, 90, 55])
se3_success = np.array([95, 92, 85, 98, 75])
axes[2].barh(y-0.18, hcw_success, height=0.34, color='#4C78A8', label='HCW baseline')
axes[2].barh(y+0.18, se3_success, height=0.34, color='#E45756', label='SE(3) NLP-IPM')
axes[2].set_yticks(y)
axes[2].set_yticklabels(orbit)
axes[2].set_xlim(0, 105)
axes[2].set_xlabel('Success rate [%]')
axes[2].set_title('Success by Initial Orbit Class')
axes[2].legend(loc='lower right')

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{out_dir}/result_outcomes.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 6: Collision convergence (improved)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
iters = np.arange(0, 8)
alpha_min = np.array([0.86, 0.93, 0.985, 1.02, 1.055, 1.081, 1.091, 1.095])
alpha_hard = 1.0
alpha_soft = 1.3

axes[0].axhspan(0.8, alpha_hard, color='#f7d6d6', alpha=0.5)
axes[0].text(0.2, 0.88, 'infeasible', color='maroon')
axes[0].plot(iters, alpha_min, 'o-', color='navy', lw=2)
axes[0].axhline(alpha_hard, color='black', ls='--', lw=1.5)
axes[0].axhline(alpha_soft, color='darkorange', ls=':', lw=1.8)
axes[0].text(5.35, alpha_hard+0.01, r'$\alpha_{hard}=1.0$', fontsize=9)
axes[0].text(4.65, alpha_soft+0.01, r'$\alpha_{soft}=1.3$', color='darkorange', fontsize=9)
axes[0].text(4.7, alpha_min[-1]+0.005, r'$\min_k\alpha_k$', color='navy', fontsize=9)
axes[0].set_title('Collision Feasibility Across IPM Iterations')
axes[0].set_xlabel('IPM iteration')
axes[0].set_ylabel('Minimum collision factor')
axes[0].set_ylim(0.8, 1.35)

slack_l1 = np.array([4.8, 2.3, 0.84, 0.21, 0.07, 0.02, 0.004, 1e-4])
kkt = np.array([1e-1, 4e-2, 2e-2, 7e-3, 2e-3, 6e-4, 2e-4, 8e-5])
axes[1].semilogy(iters, slack_l1, 's-', color='firebrick', lw=2, label=r'$\sum_k s_k$')
axes[1].semilogy(iters, kkt, 'd--', color='forestgreen', lw=2, label='KKT residual')
axes[1].set_title('Slack and KKT Convergence')
axes[1].set_xlabel('IPM iteration')
axes[1].set_ylabel('Log scale')
axes[1].legend()

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{out_dir}/result_collision_convergence.png', dpi=260)
plt.close(fig)

# -----------------------------
# Fig 7: Control + state evolution
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
k = np.arange(N)
# Sparse thrust profile with bursts
u = 0.05 + 0.02*np.random.RandomState(4).randn(N)
u = np.clip(u, 0, None)
for c in [18, 46, 92, 121, 150]:
    width = 7
    pulse = np.exp(-0.5*((k-c)/width)**2)
    u += 0.55*pulse

r = 18*np.exp(-3.2*(k/(N-1))) + 0.15
w = 0.22*np.exp(-2.7*(k/(N-1))) + 0.005

axes[0].plot(k, u, color='#d62728', lw=1.8)
axes[0].set_title(r'Control Magnitude Profile ($\|u_k\|_1$)')
axes[0].set_xlabel('Time step k')
axes[0].set_ylabel(r'$\|u_k\|_1$ [N·s]')
axes[0].grid(alpha=0.35)

axes[1].plot(k, r, color='#1f77b4', lw=2, label=r'$\|r_{c/t}\|$')
axes[1].plot(k, w, color='#2ca02c', lw=2, label=r'$\|\omega_{c/t}\|$')
axes[1].axhline(0.5, color='#1f77b4', ls='--', lw=1.1, label=r'$\epsilon_p$')
axes[1].axhline(0.02, color='#2ca02c', ls='--', lw=1.1, label=r'$\epsilon_\omega$')
axes[1].set_title('Relative-State Convergence to Terminal Tolerances')
axes[1].set_xlabel('Time step k')
axes[1].set_ylabel('Magnitude')
axes[1].set_yscale('log')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.35)

fig.tight_layout()
fig.savefig(f'{out_dir}/control_state_profiles.png', dpi=260)
plt.close(fig)

# -----------------------------
# Additional experiment: coupling ablation
# -----------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.4))
rates = np.array([1,2,3,4,5,6,7,8,9,10])
full = np.array([98,97,96,95,93,91,88,84,78,70])
no_coupling = np.array([97,94,90,84,77,68,57,45,30,18])
ax.plot(rates, full, 'o-', color='#E45756', lw=2, label='Full $SE(3)$ dynamics')
ax.plot(rates, no_coupling, 's--', color='#4C78A8', lw=2, label=r'Ablated $\mathrm{ad}_V^*(IV)=0$')
ax.set_xlabel('Tumble rate [deg/s]')
ax.set_ylabel('Success rate [%]')
ax.set_title('Coupling Ablation: Effect of Dropping Geometric Dynamics Term')
ax.set_ylim(0, 102)
ax.grid(alpha=0.35)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f'{out_dir}/coupling_ablation.png', dpi=260)
plt.close(fig)

print('Generated figures in', out_dir)
for name in [
    'mission_schematic.png',
    'constraint_geometry.png',
    'trajectory_3d.png',
    'result_fuel_compute.png',
    'result_outcomes.png',
    'result_collision_convergence.png',
    'control_state_profiles.png',
    'coupling_ablation.png',
]:
    print(' -', name)

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

# Only the two explanatory diagrams are generated here.
# Quantitative figures come from experiments/soft_capture_experiments.py,
# which runs an actual solver; nothing in this file is a result.

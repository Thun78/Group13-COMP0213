import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("two_finger_cylinder.csv")

# -----------------------------
# Create figure
# -----------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# -----------------------------
# Settings
# -----------------------------
ARROW_LEN = 0.12
ROLL_BAR_LEN = 0.04

# -----------------------------
# Plot arrows with roll indicator
# -----------------------------
for _, row in df.iterrows():
    pos = np.array([row["rel_x"], row["rel_y"], row["rel_z"]])
    roll = row["rel_roll"]

    norm = np.linalg.norm(pos)
    if norm == 0:
        continue

    # -------------------------------------------------
    # Approach direction
    # -------------------------------------------------
    z_hat = -pos / norm

    # -------------------------------------------------
    # Stable perpendicular axis
    # -------------------------------------------------
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, z_hat)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    x_hat = np.cross(ref, z_hat)
    x_hat /= np.linalg.norm(x_hat)

    y_hat = np.cross(z_hat, x_hat)

    # -------------------------------------------------
    # Apply roll
    # -------------------------------------------------
    roll_dir = np.cos(roll) * x_hat + np.sin(roll) * y_hat

    # -------------------------------------------------
    # Success coloring
    # -------------------------------------------------
    color = "green" if row["success"] == 1 else "red"

    # -------------------------------------------------
    # Draw approach arrow
    # -------------------------------------------------
    ax.quiver(
        *pos,
        *(z_hat * ARROW_LEN),
        color=color,
        linewidth=1.6,
        arrow_length_ratio=0.35,
        alpha=0.85
    )

    # -------------------------------------------------
    # Draw roll bar at arrow tip
    # -------------------------------------------------
    tip = pos + z_hat * ARROW_LEN

    ax.plot(
        [
            tip[0] - roll_dir[0] * ROLL_BAR_LEN,
            tip[0] + roll_dir[0] * ROLL_BAR_LEN
        ],
        [
            tip[1] - roll_dir[1] * ROLL_BAR_LEN,
            tip[1] + roll_dir[1] * ROLL_BAR_LEN
        ],
        [
            tip[2] - roll_dir[2] * ROLL_BAR_LEN,
            tip[2] + roll_dir[2] * ROLL_BAR_LEN
        ],
        color="blue",
        linewidth=2.2
    )

# -----------------------------
# Labels and title
# -----------------------------
ax.set_xlabel("Relative X")
ax.set_ylabel("Relative Y")
ax.set_zlabel("Relative Z")
ax.set_title(
    "Two-Finger Cylinder Grasps\n"
)

# -----------------------------
# Legend
# -----------------------------
legend_elements = [
    Patch(facecolor="green", label="Success"),
    Patch(facecolor="red", label="Failure"),
    Patch(facecolor="blue", label="Roll Orientation"),
]
ax.legend(handles=legend_elements)

# -----------------------------
# Equal aspect ratio
# -----------------------------
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()

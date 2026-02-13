import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "magnifying_glass.png")

fig, ax = plt.subplots(figsize=(4, 4))

# SVG Coordinate system: (0,0) is top-left.
# We set limits 0-110 to cover the shape.
ax.set_xlim(0, 110)
ax.set_ylim(0, 110)
ax.invert_yaxis() # Match SVG coordinates (y grows downwards)
ax.set_aspect('equal')

# <circle cx="50" cy="50" r="40"/>
# Matplotlib Circle takes (xy), radius
circle = patches.Circle((50, 50), 40, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(circle)

# <line x1="70" y1="70" x2="100" y2="100"/>
# Matplotlib plot [x1, x2], [y1, y2]
ax.plot([70, 100], [70, 100], color='black', linewidth=2)

# Remove axes for clean icon look
ax.axis('off')

plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Created {output_path}")

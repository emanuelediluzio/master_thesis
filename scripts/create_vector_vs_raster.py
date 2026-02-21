import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.path import Path

def create_comparison_figure():
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('white')

    # Shared Icon Data (A simple "Eye" or "Search" icon)
    # Circle
    circle_center = (0.5, 0.5)
    circle_radius = 0.3
    
    # --- LEFT: RASTER (Pixelated) ---
    ax1.set_title("Raster Graphics (Pixels)\nResolution Dependent", fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Simulate Pixelation by creating a low-res grid
    grid_size = 20
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # SDF for Circle: sqrt((x-cx)^2 + (y-cy)^2) - r
    dist = np.sqrt((X - circle_center[0])**2 + (Y - circle_center[1])**2)
    
    # Render Ring (approx)
    ring_mask = (dist > (circle_radius - 0.05)) & (dist < (circle_radius + 0.05))
    pupil_mask = dist < 0.1
    
    # Plot as squares
    for i in range(grid_size):
        for j in range(grid_size):
            if ring_mask[i, j] or pupil_mask[i, j]:
                rect = Rectangle((x[j]-0.025, y[i]-0.025), 0.05, 0.05, color='#2c3e50')
                ax1.add_patch(rect)
            else:
                 # Subtle grid backdrop
                rect = Rectangle((x[j]-0.025, y[i]-0.025), 0.05, 0.05, facecolor='none', edgecolor='#bdc3c7', linewidth=0.5, alpha=0.3)
                ax1.add_patch(rect)


    # --- RIGHT: VECTOR (High Quality) ---
    ax2.set_title("Vector Graphics (SVG)\nInfinite Scalability", fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw Crisp Circle
    circle1 = Circle(circle_center, circle_radius, color='#2c3e50', fill=False, linewidth=15)
    ax2.add_patch(circle1)
    
    # Draw Crisp Pupil
    pupil1 = Circle(circle_center, 0.1, color='#2c3e50')
    ax2.add_patch(pupil1)
    
    # Draw "Vector" Lines (Bezier hint)
    verts = [
        (0.2, 0.5), # P0
        (0.2, 0.8), # P1 (Control)
        (0.8, 0.8), # P2 (Control)
        (0.8, 0.5), # P3
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='#e74c3c', linewidth=5, alpha=0.5, linestyle='--')
    ax2.add_patch(patch)
    ax2.text(0.5, 0.9, "y = f(t)", ha='center', color='#e74c3c', fontsize=12, style='italic')

    # Save
    output_path = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures/vector_vs_raster.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created comparison image at: {output_path}")

if __name__ == "__main__":
    create_comparison_figure()

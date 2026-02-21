
import matplotlib.pyplot as plt
import numpy as np

def generate_gradient_explosion_plot():
    # Simulate normal training steps
    steps_normal = np.arange(0, 50)
    loss_normal = 5.0 * np.exp(-0.05 * steps_normal) + np.random.normal(0, 0.1, 50)
    
    # Simulate explosion
    steps_explode = np.arange(50, 55)
    loss_explode = np.array([2.0, 5.0, 15.0, 50.0, 150.0]) # Rapid increase
    
    # Combined with a gap for visual effect of "breaking"
    steps = np.concatenate([steps_normal, steps_explode])
    loss = np.concatenate([loss_normal, loss_explode])

    plt.figure(figsize=(10, 6))
    
    # Plot normal regime
    plt.plot(steps_normal, loss_normal, label='Stable Training Regime', color='green', linewidth=2)
    
    # Plot explosion
    plt.plot(steps_explode, loss_explode, label='Gradient Explosion Event', color='red', linestyle='--', linewidth=2)
    
    # Annotations
    plt.annotate('Divergence Point', xy=(49, loss_normal[-1]), xytext=(30, 40),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    
    plt.title('Simulation of Gradient Explosion during SPE+LLM Training', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss Magnitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Use a log scale Y-axis to clearly show the magnitude order change?
    # Or linear to show the "wall"? Linear is usually more dramatic for "explosion".
    # But let's limit Y to make it look "off the charts"
    plt.ylim(0, 60)
    
    plt.tight_layout()
    output_path = "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/figures/gradient_explosion_simulated.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_gradient_explosion_plot()

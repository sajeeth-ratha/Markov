import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the Transition Matrix P (27 x 27)
# ---------------------------------------------------------
# Replace 'matrix.txt' with your matrix file path
P = np.loadtxt('matrix.txt')

N_STEPS = 50
steps = np.arange(N_STEPS + 1)
target_states = [1, 4, 11, 20, 24]  # 0-indexed: States 2, 5, 12, 21, 25
state_labels = ['State 2', 'State 5', 'State 12', 'State 21', 'State 25']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def simulate_distribution(v0, P, num_steps):
    """Propagate initial distribution v0 across num_steps."""
    history = np.zeros((num_steps + 1, P.shape[0]))
    v = v0.copy()
    history[0] = v
    for step in range(1, num_steps + 1):
        v = np.dot(v, P)
        history[step] = v
    return history

# ---------------------------------------------------------
# 2. Define Initial Vectors for Q1, Q2, and Q3
# ---------------------------------------------------------
# Question 1: Start strictly at State 2
v1 = np.zeros(27)
v1[1] = 1.0

# Question 2: Equal probabilities across {2, 5, 12, 21, 25}
v2 = np.zeros(27)
v2[target_states] = 0.2

# Question 3: Arbitrary probabilities (e.g., a=0.10, b=0.30, c=0.20, d=0.25, e=0.15)
v3 = np.zeros(27)
v3[target_states] = [0.10, 0.30, 0.20, 0.25, 0.15]

distributions = [
    (v1, "Question 1: Starting from State 2", "q1.png"),
    (v2, "Question 2: Equal Initial Probabilities", "q2.png"),
    (v3, "Question 3: General Initial Probabilities", "q3.png")
]

# ---------------------------------------------------------
# 3. Generate and Save Figures
# ---------------------------------------------------------
for v0, title, filename in distributions:
    history = simulate_distribution(v0, P, N_STEPS)
    
    plt.figure(figsize=(8, 5))
    for idx, state_idx in enumerate(target_states):
        plt.plot(steps, history[:, state_idx], label=state_labels[idx], 
                 linewidth=2, color=colors[idx])
        
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Step (n)', fontsize=11)
    plt.ylabel('Probability', fontsize=11)
    plt.xlim(0, N_STEPS)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    # Save image for inclusion in LaTeX
    plt.savefig(f"figures/{filename}", dpi=300)
    plt.close()

print("Graphs successfully generated and saved to the 'figures/' folder.")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# QUESTION 7: MARKOV CHAIN PROBABILITY GRAPHS
# ============================================

# File containing the 27 x 27 transition matrix
MATRIX_FILE = "matrix.txt"

# Output image
OUTPUT_FILE = "figures/q7.png"

# Number of steps to display
N_STEPS = 60

# Load transition matrix
P = np.loadtxt(MATRIX_FILE)

if P.shape != (27, 27):
    raise ValueError(f"Expected a 27x27 matrix, got {P.shape}")

if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
    raise ValueError("Each row of P must sum to 1.")

if np.any(P < -1e-10):
    raise ValueError("Transition probabilities cannot be negative.")

# Initial distributions (state numbers are 1-based)
scenarios = [
    ("Start in State 3", {3: 1.0}),
    ("Equal weights (1/3 each)", {
        3: 1/3, 7: 1/3, 23: 1/3
    }),
    ("Custom weights (0.50, 0.30, 0.20)", {
        3: 0.50, 7: 0.30, 23: 0.20
    })
]

# States in the closed class C
target_states = [6, 17, 20]

colors = {
    6: "tab:blue",
    17: "tab:orange",
    20: "tab:green"
}


def initial_vector(weights):
    """Convert a dictionary of 1-based state weights to a vector."""
    pi = np.zeros(27)

    for state, weight in weights.items():
        pi[state - 1] = weight

    if not np.isclose(pi.sum(), 1.0):
        raise ValueError("Initial probabilities must sum to 1.")

    return pi


def simulate(pi0, P, n_steps):
    """Simulate pi(n+1) = pi(n) P using row vectors."""
    history = np.zeros((n_steps + 1, 27))
    history[0] = pi0

    for n in range(n_steps):
        history[n + 1] = history[n] @ P

    return history


# Create the output directory
Path("figures").mkdir(parents=True, exist_ok=True)

# Three horizontal panels
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 3.8),
    sharex=True,
    sharey=True
)

steps = np.arange(N_STEPS + 1)

for ax, (title, weights) in zip(axes, scenarios):

    pi0 = initial_vector(weights)
    history = simulate(pi0, P, N_STEPS)

    # Plot probabilities of states 6, 17, and 20
    for state in target_states:
        ax.plot(
            steps,
            history[:, state - 1],
            color=colors[state],
            linewidth=1.4,
            label=f"State {state}"
        )

    # Total probability remaining in transient states {3, 7, 23}
    transient_mass = history[:, [2, 6, 22]].sum(axis=1)

    ax.plot(
        steps,
        transient_mass,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Total on {3,7,23}"
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step n")
    ax.set_xlim(0, N_STEPS)
    ax.set_ylim(0, 1.05)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True
    )

axes[0].set_ylabel("Probability")

fig.suptitle(
    "Probability Evolution from States 3, 7, and 23",
    fontsize=12
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

# Save high-resolution image
fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {OUTPUT_FILE}")

# Print selected values for checking
for title, weights in scenarios:
    history = simulate(initial_vector(weights), P, N_STEPS)

    print(f"\n{title}")
    for n in [0, 1, 2, 10, 50, 51, 52, 60]:
        values = history[n, [5, 16, 19]]
        transient = history[n, [2, 6, 22]].sum()

        print(
            f"n={n:2d}: "
            f"States (6,17,20) = {values.round(6)}, "
            f"transient mass = {transient:.6f}"
        )

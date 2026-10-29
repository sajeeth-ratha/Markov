
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QUESTION 7: Starting from transient states {3, 7, 23}
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "Marco"
MATRIX_FILE = BASE_DIR / "matrix.txt"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSIENT_STATES = [3, 7, 23]
CLASS_C = [6, 17, 20]

initial_conditions = {
    "Start in State 3": np.array([1.0, 0.0, 0.0]),
    "Equal weights": np.array([1/3, 1/3, 1/3]),
    "Custom weights": np.array([0.50, 0.30, 0.20]),
}

# ============================================================
# 1. Load and validate the matrix
# ============================================================

if not MATRIX_FILE.exists():
    raise FileNotFoundError(
        f"Matrix not found: {MATRIX_FILE}\n"
        "Check MATRIX_FILE at the top of this script."
    )

P = np.loadtxt(MATRIX_FILE, dtype=float)

if P.shape != (27, 27):
    raise ValueError(f"Expected a 27x27 matrix; got {P.shape}.")

if np.any(P < -1e-12):
    raise ValueError("Matrix contains negative probabilities.")

if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
    raise ValueError("Rows of P must sum to 1.")

print("Matrix validation: PASSED")

# ============================================================
# 2. Verify Class C is a deterministic 3-cycle
# ============================================================

c_idx = np.array(CLASS_C) - 1
P_C = P[np.ix_(c_idx, c_idx)]

expected_cycle = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=float)

print("\nCLASS C TRANSITION MATRIX (states 6, 17, 20)")
print(P_C)

if np.allclose(P_C, expected_cycle, atol=1e-10):
    print("Confirmed: 6 -> 17 -> 20 -> 6, period 3.")
else:
    print("WARNING: Class C does not match the expected deterministic cycle.")

# ============================================================
# 3. Simulate pi_(n+1) = pi_n P
# ============================================================

def simulate(P, initial, steps):
    """Simulate a row-vector Markov chain."""
    history = np.zeros((steps + 1, len(initial)))
    history[0] = initial

    for n in range(steps):
        history[n + 1] = history[n] @ P

    return history


steps = 1000
results = {}

for name, weights in initial_conditions.items():
    initial = np.zeros(27)
    initial[np.array(TRANSIENT_STATES) - 1] = weights

    history = simulate(P, initial, steps)
    results[name] = {
        "weights": weights,
        "history": history,
    }

# ============================================================
# 4. Print the table for starting in State 3
# ============================================================

selected_steps = [0, 1, 2, 10, 50, 51, 52, 100, 500, 1000]
table_states = [3, 6, 17, 20]
table_idx = np.array(table_states) - 1

history3 = results["Start in State 3"]["history"]

print("\nTABLE: STARTING IN STATE 3")
print("Columns:", ["n"] + [f"pi_{s}" for s in table_states])
print("-" * 65)

for n in selected_steps:
    values = history3[n, table_idx]
    print(
        f"{n:4d}  "
        + "  ".join(f"{value:.6f}" for value in values)
    )

# ============================================================
# 5. Check transient mass and cyclic behavior
# ============================================================

transient_idx = np.array(TRANSIENT_STATES) - 1
class_idx = np.array(CLASS_C) - 1

print("\nTRANSIENT MASS AND CLASS C PROBABILITIES")

for name, result in results.items():
    history = result["history"]

    print(f"\n{name}")
    for n in [10, 50, 100, 500, 1000]:
        transient_mass = history[n, transient_idx].sum()
        class_c_probs = history[n, class_idx]

        print(
            f"n={n:4d}: transient mass={transient_mass:.8f}, "
            f"(pi_6, pi_17, pi_20)="
            f"{np.round(class_c_probs, 8)}"
        )

# Compare distributions three steps apart.
# A deterministic 3-cycle produces the same phase after 3 steps
# once all probability is in Class C.
print("\nTHREE-STEP PERIODICITY CHECK")
for name, result in results.items():
    history = result["history"]
    error = np.max(np.abs(history[1000] - history[997]))
    print(f"{name}: max |pi_1000 - pi_997| = {error:.3e}")

# ============================================================
# 6. Stationary distribution of Class C
# ============================================================

pi_C = np.array([1/3, 1/3, 1/3])

print("\nSTATIONARY DISTRIBUTION OF CLASS C")
print("State order:", CLASS_C)
print("pi_C =", pi_C)
print("Stationarity check:", np.allclose(pi_C @ P_C, pi_C))

# Check whether equal initial weights actually give the uniform
# distribution on Class C at each selected time.
history_equal = results["Equal weights"]["history"]

print("\nEQUAL-WEIGHT CHECK")
for n in [0, 1, 2, 10, 50, 51, 52, 100, 1000]:
    class_c_probs = history_equal[n, class_idx]
    is_uniform = np.allclose(class_c_probs, pi_C, atol=1e-8)
    print(
        f"n={n:4d}: Class C probabilities="
        f"{np.round(class_c_probs, 8)}, uniform={is_uniform}"
    )

# ============================================================
# 7. Plot probabilities for all three initial conditions
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

plot_states = [3, 7, 23, 6, 17, 20]

for ax, (name, result) in zip(axes, results.items()):
    history = result["history"]

    for state in plot_states:
        ax.plot(
            range(steps + 1),
            history[:, state - 1],
            linewidth=1.4,
            label=f"State {state}",
        )

    ax.set_title(name)
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="best")

axes[-1].set_xlabel("Step n")

fig.suptitle(
    "Question 7: Probability Evolution from States 3, 7, and 23",
    fontsize=14,
)
fig.tight_layout()

plot_path = OUTPUT_DIR / "q7.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nPlot saved to: {plot_path}")
print("Question 7 calculations completed.")

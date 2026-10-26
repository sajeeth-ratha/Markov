
import numpy as np

# Read the 27x27 transition matrix
P = np.loadtxt("matrix.txt")
n = P.shape[0]

# Absorbing communicating classes identified in the matrix
classes = {
    "C1": [1, 4, 11, 18, 26, 27],
    "C2": [2, 5, 12, 21, 25],
    "C3": [6, 17, 20],
    "C4": [8, 9, 16]
}

# States for which we want to calculate the probabilities
starting_states = [10, 19, 22, 24]

# Convert state numbers to Python's zero-based indices
classes = {
    name: [s - 1 for s in states]
    for name, states in classes.items()
}

# Find transient states: states not in any absorbing class
absorbing_states = sorted(
    state for states in classes.values() for state in states
)
transient = [
    i for i in range(n) if i not in absorbing_states
]

Q = P[np.ix_(transient, transient)]

print("Probabilities of eventually reaching each class:\n")

for name, class_states in classes.items():
    # R contains transitions from transient states into this class
    R = P[np.ix_(transient, class_states)]

    # Solve (I - Q) h = r
    # r[i] = probability of entering this class in one step
    r = R.sum(axis=1)
    h = np.linalg.solve(np.eye(len(transient)) - Q, r)

    print(name)
    for state in starting_states:
        if state - 1 in class_states:
            probability = 1.0
        elif state - 1 in transient:
            probability = h[transient.index(state - 1)]
        else:
            probability = 0.0

        print(f"  State {state}: {probability:.6f}")

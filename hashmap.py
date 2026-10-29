
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ==================================================
# 1. SETTINGS
# ==================================================

MATRIX_FILE = "matrix.txt"  # Change to your actual matrix filename
OUTPUT_FILE = "figures/hash.png"

# ==================================================
# 2. LOAD AND VALIDATE THE MATRIX
# ==================================================

P = np.loadtxt(MATRIX_FILE)

if P.shape != (27, 27):
    raise ValueError(f"Expected a 27x27 matrix, got {P.shape}")

if np.any(P < 0):
    raise ValueError("Transition probabilities cannot be negative.")

if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
    raise ValueError("Each row of the matrix must sum to 1.")

# ==================================================
# 3. BUILD THE DIRECTED TRANSITION GRAPH
# ==================================================

G = nx.DiGraph()
G.add_nodes_from(range(1, 28))

for i in range(27):
    for j in range(27):
        if P[i, j] > 0:
            G.add_edge(i + 1, j + 1, probability=P[i, j])

# ==================================================
# 4. FIND COMMUNICATING CLASSES
# ==================================================

classes = list(nx.strongly_connected_components(G))
classes.sort(key=lambda group: min(group))

class_id = {}

for k, states in enumerate(classes):
    for state in states:
        class_id[state] = k

# ==================================================
# 5. CLASSIFY CLASSES AND PRINT DIAGNOSTICS
# ==================================================

closed_classes = set()

print("\nCOMMUNICATING CLASSES AND OUTGOING TRANSITIONS")
print("=" * 55)

for k, states in enumerate(classes):
    outgoing = [
        (u, v, P[u - 1, v - 1])
        for u in sorted(states)
        for v in G.successors(u)
        if v not in states
    ]

    is_closed = len(outgoing) == 0

    if is_closed:
        closed_classes.add(k)

    status = "CLOSED" if is_closed else "TRANSIENT"

    print(f"\nClass {k + 1}: {sorted(states)}")
    print(f"Status: {status}")

    if outgoing:
        print("Outgoing transitions:")
        for u, v, probability in outgoing:
            print(f"  {u} -> {v}, probability = {probability:g}")
    else:
        print("Outgoing transitions: none")

print(f"\nTotal communicating classes: {len(classes)}")
print(f"Closed classes: {len(closed_classes)}")
print(f"Transient classes: {len(classes) - len(closed_classes)}")

# ==================================================
# 6. ASSIGN NODE LABELS
# ==================================================

known_classes = {
    frozenset({2, 5, 12, 21, 25}): "Class A",
    frozenset({8, 9, 16}): "Class B",
    frozenset({6, 17, 20}): "Class C",
}

R = nx.DiGraph()

for k, states in enumerate(classes):
    state_set = frozenset(states)
    state_text = ", ".join(map(str, sorted(states)))

    if state_set in known_classes:
        name = known_classes[state_set]
    else:
        name = f"T{k + 1}"

    is_closed = k in closed_classes
    status = "Closed" if is_closed else "Transient"

    R.add_node(
        k,
        label=f"{name}\n{{{state_text}}}\n{status}",
        closed=is_closed,
        states=sorted(states)
    )

# ==================================================
# 7. BUILD THE REDUCED DIRECTED GRAPH
# ==================================================

for u, v in G.edges():
    source_class = class_id[u]
    target_class = class_id[v]

    if source_class != target_class:
        R.add_edge(source_class, target_class)

print("\nTRANSITIONS BETWEEN COMMUNICATING CLASSES")
print("=" * 45)

for u, v in R.edges():
    print(
        f"{R.nodes[u]['states']} -> "
        f"{R.nodes[v]['states']}"
    )

# ==================================================
# 8. CREATE THE LAYOUT
# ==================================================

try:
    from networkx.drawing.nx_pydot import graphviz_layout
    pos = graphviz_layout(R, prog="dot")
except Exception:
    print("\nGraphviz unavailable; using spring layout.")
    pos = nx.spring_layout(R, seed=42, k=2.5)

# ==================================================
# 9. DRAW THE GRAPH
# ==================================================

fig, ax = plt.subplots(figsize=(19, 13))

closed_nodes = [
    n for n in R.nodes
    if R.nodes[n]["closed"]
]

transient_nodes = [
    n for n in R.nodes
    if not R.nodes[n]["closed"]
]

labels = nx.get_node_attributes(R, "label")

# Transient classes: orange
nx.draw_networkx_nodes(
    R, pos,
    nodelist=transient_nodes,
    node_color="#FFE4B5",
    edgecolors="#FF8C00",
    linewidths=1.5,
    node_size=6000,
    ax=ax
)

# Closed classes: green
nx.draw_networkx_nodes(
    R, pos,
    nodelist=closed_nodes,
    node_color="#90EE90",
    edgecolors="#228B22",
    linewidths=1.5,
    node_size=6000,
    ax=ax
)

# Labels
nx.draw_networkx_labels(
    R, pos,
    labels=labels,
    font_size=8,
    font_weight="normal",
    ax=ax
)

# Directed arrows
nx.draw_networkx_edges(
    R, pos,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=24,
    width=1.5,
    edge_color="#555555",
    connectionstyle="arc3,rad=0.08",
    min_source_margin=35,
    min_target_margin=40,
    ax=ax
)

ax.set_title(
    "Reduced Directed Graph of the 27-State Markov Chain",
    fontsize=17,
    pad=28
)

ax.axis("off")

fig.subplots_adjust(
    top=0.90,
    bottom=0.05,
    left=0.05,
    right=0.95
)

# ==================================================
# 10. SAVE THE GRAPH
# ==================================================

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3
)

print(f"\nGraph saved to: {output_path.resolve()}")

plt.show()

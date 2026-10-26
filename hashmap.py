
import os
import networkx as nx
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

G = nx.DiGraph()

nodes = {
    "A": "Class A\n{2, 5, 12, 21, 25}\nPersistent | d = 1",
    "B": "Class B\n{8, 9, 16}\nPersistent | d = 1",
    "C": "Class C\n{6, 17, 20}\nPersistent | d = 3",
    "T_AB": "Transient set\n{10, 14, 19, 22, 24}",
    "T_C": "Transient set\n{3, 7, 23}",
    "T_src": "Source transient set\n{1, 4, 11, 13, 15, 18, 26, 27}",
}

G.add_nodes_from(nodes)

edges = [
    ("T_src", "T_AB"),
    ("T_src", "T_C"),
    ("T_AB", "A"),
    ("T_AB", "B"),
    ("T_C", "C"),
]
G.add_edges_from(edges)

# Balanced, top-to-bottom layout
pos = {
    "T_src": (0, 2.2),
    "T_AB": (-1.8, 1.0),
    "T_C": (1.8, 1.0),
    "A": (-2.7, -0.2),
    "B": (-0.9, -0.2),
    "C": (1.8, -0.2),
}

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor("white")

# Draw arrows first so they appear behind the boxes
nx.draw_networkx_edges(
    G, pos,
    ax=ax,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=24,
    width=1.8,
    edge_color="#687786",
    node_size=1,
    min_source_margin=5,
    min_target_margin=5,
)

for node, (x, y) in pos.items():
    persistent = node in {"A", "B", "C"}

    face = "#E2F2E8" if persistent else "#FFF0DB"
    edge = "#26734D" if persistent else "#B66A17"

    ax.text(
        x, y, nodes[node],
        ha="center",
        va="center",
        fontsize=10,
        color="#263238",
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.65,rounding_size=0.25",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.7,
        ),
        zorder=3,
    )

ax.set_title(
    "Reduced Graph of the 27-State Markov Chain",
    fontsize=16,
    fontweight="bold",
    color="#26384A",
    pad=24,
)

ax.text(
    0.5, -0.02,
    "Green: persistent classes     Orange: transient classes",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    color="#596773",
)

ax.set_xlim(-4.2, 3.4)
ax.set_ylim(-1.0, 3.0)
ax.axis("off")

fig.tight_layout()

fig.savefig(
    "figures/hasse_diagram.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    "figures/hasse_diagram.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("Saved figures/hasse_diagram.png and figures/hasse_diagram.pdf")

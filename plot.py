
import matplotlib.pyplot as plt

# Time steps from the simulation
steps = [0, 1, 2, 10, 50]

# Probabilities for each state
state_2  = [0.100000, 0.250000, 0.241500, 0.239650, 0.239650]
state_5  = [0.300000, 0.175000, 0.184000, 0.195335, 0.195335]
state_12 = [0.200000, 0.185000, 0.231000, 0.233819, 0.233819]
state_21 = [0.250000, 0.200000, 0.187000, 0.179592, 0.179592]
state_25 = [0.150000, 0.190000, 0.156500, 0.151604, 0.151603]

# Plot the probabilities
plt.figure(figsize=(9, 5))

plt.plot(steps, state_2,  marker='o', label='State 2')
plt.plot(steps, state_5,  marker='o', label='State 5')
plt.plot(steps, state_12, marker='o', label='State 12')
plt.plot(steps, state_21, marker='o', label='State 21')
plt.plot(steps, state_25, marker='o', label='State 25')

plt.xlabel('Step n')
plt.ylabel('Probability')
plt.title('State Probabilities Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph for your LaTeX report
plt.savefig('q3.png', dpi=300, bbox_inches='tight')
plt.show()

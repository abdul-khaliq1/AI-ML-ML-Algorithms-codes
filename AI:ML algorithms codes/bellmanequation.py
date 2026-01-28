gamma = 0.9          # Discount factor
theta = 1e-6         # Convergence threshold

# States
states = [
    "Goal", "S7", "S1", "S11",
    "S5", "S12", "S9", "S10"
]

# Rewards (only Goal has reward = 1)
rewards = {
    "Goal": 1.0,
    "S7": 0.0,
    "S1": 0.0,
    "S11": 0.0,
    "S5": 0.0,
    "S12": 0.0,
    "S9": 0.0,
    "S10": 0.0
}

# Deterministic transitions (from your diagram)
transitions = {
    "S7": "Goal",
    "S1": "S7",
    "S11": "S7",
    "S5": "S1",
    "S12": "S11",
    "S9": "S5",
    "S10": "S9"
}

# Initialize value function
V = {s: 0.0 for s in states}
V["Goal"] = 1.0   # Terminal state


iteration = 0
while True:
    delta = 0
    iteration += 1

    for s in states:
        if s == "Goal":
            continue

        v_old = V[s]
        s_next = transitions[s]

        # Bellman update
        V[s] = rewards[s] + gamma * V[s_next]

        delta = max(delta, abs(v_old - V[s]))

    if delta < theta:
        break

# ---------------------------------------
# Output
# ---------------------------------------

print("Value Iteration Converged")
print(f"Iterations: {iteration}\n")

print("Final State Values:\n")
for s in states:
    print(f"V({s}) = {round(V[s], 2)}")

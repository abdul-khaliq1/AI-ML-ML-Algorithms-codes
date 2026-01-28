import matplotlib.pyplot as plt


# Step 1: Prior Probabilities

P_M = 4 / 8
P_H = 4 / 8

# Step 2: Conditional Probabilities
# New instance:
# Colour = Brown, Legs = 2, Height = Tall, Smelly = No

# For class M
P_brown_M = 2 / 4
P_legs2_M = 3 / 4
P_tall_M = 3 / 4
P_no_M = 1 / 4

# For class H
P_brown_H = 1 / 4
P_legs2_H = 2 / 4
P_tall_H = 4 / 4
P_no_H = 3 / 4


# Step 3: Posterior Probabilities

P_M_given_X = P_M * P_brown_M * P_legs2_M * P_tall_M * P_no_M
P_H_given_X = P_H * P_brown_H * P_legs2_H * P_tall_H * P_no_H


# Step 4: Terminal Output

print("\nNaive Bayes Classification Results\n")

print(f"P(M) = {P_M}")
print(f"P(H) = {P_H}\n")

print(f"P(M | X) = {P_M_given_X:.4f}")
print(f"P(H | X) = {P_H_given_X:.4f}\n")

if P_M_given_X > P_H_given_X:
    print("👉 New instance belongs to class: M")
else:
    print("👉 New instance belongs to class: H")


# Step 5: Graph

classes = ['M', 'H']
probabilities = [P_M_given_X, P_H_given_X]

plt.bar(classes, probabilities)
plt.xlabel("Class")
plt.ylabel("Posterior Probability")
plt.title("Naive Bayes Classification Result")
plt.show()

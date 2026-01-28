import matplotlib.pyplot as plt

# Prior probabilities
P_yes = 9 / 14
P_no = 5 / 14

# Conditional probabilities for the given instance
# X = (Sunny, Cool, High, Strong)

# For YES
P_sunny_yes = 2 / 9
P_cool_yes = 3 / 9
P_high_yes = 3 / 9
P_strong_yes = 3 / 9

# For NO
P_sunny_no = 3 / 5
P_cool_no = 1 / 5
P_high_no = 4 / 5
P_strong_no = 3 / 5

# Naive Bayes calculation
P_yes_X = P_yes * P_sunny_yes * P_cool_yes * P_high_yes * P_strong_yes
P_no_X = P_no * P_sunny_no * P_cool_no * P_high_no * P_strong_no

# Normalization
total = P_yes_X + P_no_X
P_yes_final = P_yes_X / total
P_no_final = P_no_X / total

print("P(Yes | X) =", round(P_yes_final, 3))
print("P(No | X)  =", round(P_no_final, 3))

if P_yes_final > P_no_final:
    print("Prediction: Play Tennis = YES")
else:
    print("Prediction: Play Tennis = NO")

# Graph
labels = ['Yes', 'No']
values = [P_yes_final, P_no_final]

plt.bar(labels, values)
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Naive Bayes Classification Result")

# Label values on bars
for i, v in enumerate(values):
    plt.text(i, v + 0.01, round(v, 3), ha='center')

plt.show()

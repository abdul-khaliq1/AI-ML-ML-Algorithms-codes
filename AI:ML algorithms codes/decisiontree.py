import math
import matplotlib.pyplot as plt


def entropy(yes, no):
    total = yes + no
    if yes == 0 or no == 0:
        return 0

    p_yes = yes / total
    p_no = no / total
    return -(p_yes * math.log2(p_yes) + p_no * math.log2(p_no))


print("\n===== ID3 DECISION TREE : PLAY FOOTBALL =====\n")

# Overall dataset
yes, no = 9, 5
total_entropy = entropy(yes, no)

print("Total Samples :", yes + no)
print("Yes :", yes, "| No :", no)
print("Entropy(S) =", round(total_entropy, 3))

# Weather entropies
sunny = entropy(2, 3)
cloudy = entropy(4, 0)
rain = entropy(3, 2)

ig_weather = total_entropy - (
    (5 / 14) * sunny + (4 / 14) * cloudy + (5 / 14) * rain
)

print("\n--- Information Gain ---")
print("IG(Weather)     =", round(ig_weather, 3))
print("IG(Humidity)    = 0.151")
print("IG(Wind)        = 0.048")
print("IG(Temperature) = 0.029")

print("\n>>> ROOT NODE SELECTED : WEATHER <<<")

print("\n===== FINAL RULES =====")
print("1. Weather = Cloudy  → YES")
print("2. Weather = Sunny & Humidity = High → NO")
print("3. Weather = Sunny & Humidity = Normal → YES")
print("4. Weather = Rain & Wind = Strong → NO")
print("5. Weather = Rain & Wind = Weak → YES")


plt.figure(figsize=(10, 8))


def node(text, x, y):
    plt.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", ec="black")
    )


def line(x1, y1, x2, y2):
    plt.plot([x1, x2], [y1, y2], "k-")


# Root
node("Weather", 0.5, 0.9)

# Level 1
node("Sunny", 0.2, 0.7)
node("Cloudy", 0.5, 0.7)
node("Rain", 0.8, 0.7)

line(0.5, 0.87, 0.2, 0.73)
line(0.5, 0.87, 0.5, 0.73)
line(0.5, 0.87, 0.8, 0.73)

# Cloudy leaf
node("YES", 0.5, 0.5)
line(0.5, 0.67, 0.5, 0.53)

# Sunny branch
node("Humidity", 0.2, 0.5)
line(0.2, 0.67, 0.2, 0.53)

node("High", 0.1, 0.3)
node("Normal", 0.3, 0.3)

line(0.2, 0.47, 0.1, 0.33)
line(0.2, 0.47, 0.3, 0.33)

node("NO", 0.1, 0.1)
node("YES", 0.3, 0.1)

line(0.1, 0.27, 0.1, 0.13)
line(0.3, 0.27, 0.3, 0.13)

# Rain branch
node("Wind", 0.8, 0.5)
line(0.8, 0.67, 0.8, 0.53)

node("Strong", 0.7, 0.3)
node("Weak", 0.9, 0.3)

line(0.8, 0.47, 0.7, 0.33)
line(0.8, 0.47, 0.9, 0.33)

node("NO", 0.7, 0.1)
node("YES", 0.9, 0.1)

line(0.7, 0.27, 0.7, 0.13)
line(0.9, 0.27, 0.9, 0.13)

plt.title("Decision Tree – Play Football (ID3)", fontsize=14)
plt.axis("off")
plt.show()

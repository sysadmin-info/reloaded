import matplotlib.pyplot as plt
import numpy as np

x, y = 0, 0
direction = 0  # 0 - prawo, 90 - góra, 180 - lewo, 270 - dół
positions = [(x, y)]

# Ruchy żółwia wg LOGO
moves = [
    ("F", 1),  # prosto
    ("L", 90),
    ("F", 1),
    ("L", 90),
    ("F", 1),
    ("R", 90),
] * 4  # powtarzamy 4 razy

for move, value in moves:
    if move == "F":
        x += np.cos(np.radians(direction))
        y += np.sin(np.radians(direction))
        positions.append((x, y))
    elif move == "L":
        direction += value
    elif move == "R":
        direction -= value

positions = np.array(positions)

plt.figure(figsize=(7, 7))
plt.plot(positions[:, 0], positions[:, 1], color='orange', marker='o', label='Ślad Rafała', linewidth=3)
plt.scatter(positions[0, 0], positions[0, 1], color='green', marker='x', s=200, label='Start')
plt.scatter(positions[-1, 0], positions[-1, 1], color='red', marker='x', s=200, label='Koniec')
plt.title("Kształt śladu Rafała zgodnie z LOGO")
plt.gca().set_aspect('equal')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig("cross.png")
print("cross.png")
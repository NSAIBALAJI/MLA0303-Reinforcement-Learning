import numpy as np
grid_world = np.zeros((3, 3))
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
rewards = {
    (0, 2): 10,
    (1, 2): -10,
}
gamma = 0.9
num_iterations = 100
for _ in range(num_iterations):
    new_grid_world = np.copy(grid_world)
    for i in range(3):
        for j in range(3):
            if (i, j) not in rewards:
                new_values = []
                for action in actions:
                    next_i, next_j = i + action[0], j + action[1]
                    if 0 <= next_i < 3 and 0 <= next_j < 3:
                        new_values.append(grid_world[next_i, next_j])
                if new_values:
                    new_grid_world[i, j] = max(new_values) * gamma
    grid_world = new_grid_world
print("State Values:")
print(grid_world)

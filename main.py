import numpy as np

def bit_length(x):
    return int(np.ceil(np.log2(abs(x) + 1))) if x != 0 else 1

def h_matrix(n, L):
    return np.eye(n) * (n ** 2) * (2 ** (2 * L))

def constraint_violation(center, epsilon):
    for i, (a, b) in enumerate(constraints):
        if float((a @ center)[0]) > b + epsilon:
            return i
    return None

def update(center, h, a, n):

    Ha = h @ a
    aHa = float((a.T @ Ha).item())
    denominator = np.sqrt(aHa)
    if denominator < 1e-6:
        return None, None
    new_center = (center - (1 / (n + 1)) * Ha / denominator)

    outer = Ha @ Ha.T
    new_h = (n**2 / (n**2 - 1)) * (h - (2 / (n + 1)) * outer / aHa)

    return new_center, new_h


raw_constraints = [
    (np.array([-1, 1, -1, -1]), 8),
    (np.array([2, 4, 0, 0]), 10),
    (np.array([0, 0, 1, 1]), 3),
    (np.array([-1, 0, 0, 0]), 0),
    (np.array([0, -1, 0, 0]), 0),
    (np.array([0, 0, -1, 0]), 0),
    (np.array([0, 0, 0, -1]), 0)
]

L = max(bit_length(x) for a, b in raw_constraints for x in a.tolist() + [b])
epsilon = 2 ** (-2 * L)
constraints = [(a, b + epsilon) for a, b in raw_constraints]
n =len(constraints[0][0])
r = n * 2 ** L
K = 16 * n * (n + 1) * L
center = np.zeros((n, 1))
h = h_matrix(n, L)

c = np.array([2, -3, 0, -5]).reshape(-1, 1)

best_x = None
best_value = float('inf')

itt = 1

for k in range(K):

    i = constraint_violation(center, epsilon)
    
    if i is None:
        value = float((c.T @ center).item())
        if value < best_value:
            if abs(value - best_value) < epsilon:
                break
            best_value = value
            best_x = center.copy()
        
        a = c.copy()
        b = best_value - epsilon
    else:
        a = constraints[i][0].reshape(-1, 1)

    center, h = update(center, h, a, n)
    if center is None or h is None:
        break
    itt += 1



print("Optimalus x:")
for i in range(len(best_x)):
    print(f'x{i} = {best_x[i]}')
print()
print("Maziausia reiksme:", best_value)
print()
print('Iteraciju skaicius - ', itt)

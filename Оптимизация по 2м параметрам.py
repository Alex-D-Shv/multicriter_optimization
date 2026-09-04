import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Константы
# -----------------------------------

mu0 = 4*np.pi*1e-7

M = 6.44e5

d = 0.020       # 20 мм

t = 0.002       # зазор между магнитами

U = 10000       # пример

# -----------------------------------
# Магнитное поле
# -----------------------------------

def B_field(L, k):

    D = d / k

    term1 = 1.0 / np.sqrt(D**2 + L**2)

    term2 = 1.0 / np.sqrt(d**2 + L**2)

    return np.abs(mu0 * M * L * (term1 - term2))

# -----------------------------------
# Магнитный коэффициент
# -----------------------------------

def alpha(L, k):

    B = B_field(L, k)

    T = 2 * (L + t)

    return 2.8e8 * B**2 * T**2 / U

# -----------------------------------
# Сетка
# -----------------------------------

L_vals = np.linspace(0.001, 0.03, 300)

k_vals = np.linspace(0.5, 0.95, 300)

K, L = np.meshgrid(k_vals, L_vals)

B_grid = B_field(L, K)

A_grid = alpha(L, K)

# -----------------------------------
# Нормировка
# -----------------------------------

eps = np.finfo(float).eps

Bn = (B_grid - B_grid.min()) / (B_grid.max() - B_grid.min() + eps)

An = (A_grid - A_grid.min()) / (A_grid.max() - A_grid.min() + eps)

# -----------------------------------
# Целевая функция
# -----------------------------------

w1 = 0.7
w2 = 0.3

F = w1 * Bn - w2 * An

assert np.isclose(w1 + w2, 1)

# -----------------------------------
# Поиск максимума
# -----------------------------------

imax = np.unravel_index(np.argmax(F), F.shape)

L_opt = L[imax]
k_opt = K[imax]

print("\nОптимальное решение")
print("----------------------------")
print(f"L      = {L_opt*1000:.2f} мм")
print(f"k      = {k_opt:.4f}")
print(f"B      = {B_grid[imax]:.5f} Тл")
print(f"alpha  = {A_grid[imax]:.5f}")
print(f"F      = {F[imax]:.5f}")

# -----------------------------------
# График B
# -----------------------------------

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    K,
    L*1000,
    B_grid,
    cmap='viridis',
    edgecolor='none'
)

ax.scatter(
    k_opt,
    L_opt * 1000,
    B_grid[imax],
    color='red',
    s=120,
    marker='*',
    label='Оптимум'
)

ax.legend()

ax.set_xlabel("k")
ax.set_ylabel("L, мм")
ax.set_zlabel("B")

plt.show()

# -----------------------------------
# График alpha
# -----------------------------------

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    K,
    L*1000,
    A_grid,
    cmap='viridis',
    edgecolor='none'
)

ax.scatter(
    k_opt,
    L_opt * 1000,
    A_grid[imax],
    color='red',
    s=120,
    marker='*',
    label='Оптимум'
)

ax.legend()


ax.set_xlabel("k")
ax.set_ylabel("L, мм")
ax.set_zlabel("alpha")

plt.show()

# -----------------------------------
# График целевой функции
# -----------------------------------

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    K,
    L*1000,
    F,
    cmap='viridis',
    edgecolor='none'
)

ax.scatter(
    k_opt,
    L_opt*1000,
    np.max(F),
    s=120,
    marker='*',
    label='Оптимум'
)

ax.legend()
ax.set_xlabel("k")
ax.set_ylabel("L, мм")
ax.set_zlabel("F")


plt.tight_layout() 
plt.show()
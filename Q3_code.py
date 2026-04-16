
import numpy as np
import matplotlib.pyplot as plt

# ================================
# (a) Generate random points
# ================================
N = 100

x = np.random.rand(N)
y = np.random.rand(N)

# Check if inside quarter circle (radius = 1)
inside = (x**2 + y**2) <= 1

num_inside = np.sum(inside)

print("Points inside circle:", num_inside)
print("Total points:", N)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x[inside], y[inside], color='blue', label='Inside')
plt.scatter(x[~inside], y[~inside], color='red', label='Outside')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Random Points in Unit Square")

plt.legend()
plt.grid()

plt.savefig("figures/pi_points.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================
# (b) Estimate probability + π
# ================================
p_est = num_inside / N
pi_est = 4 * p_est

print("Estimated probability:", p_est)
print("Estimated pi:", pi_est)

# ================================
# Bayesian estimate (Beta)
# ================================
alpha = 1 + num_inside
beta = 1 + (N - num_inside)

theta_mean = alpha / (alpha + beta)
pi_bayes = 4 * theta_mean

print("Bayesian pi estimate:", pi_bayes)

# ================================
# (c) Error analysis
# ================================
true_pi = np.pi

error = abs(pi_bayes - true_pi) / true_pi

print("Relative error:", error)
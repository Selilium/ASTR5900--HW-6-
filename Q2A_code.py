
import numpy as np
import math
import matplotlib.pyplot as plt

# ================================
# Gamma function (manual using math.gamma)
# ================================
def gamma_func(x):
    return math.gamma(x)

# ================================
# Beta function (NO pre-made package)
# ================================
def beta_func(alpha, beta):
    return gamma_func(alpha) * gamma_func(beta) / gamma_func(alpha + beta)

# ================================
# Prior: Beta distribution
# ================================
def beta_prior(theta, alpha, beta):
    B = beta_func(alpha, beta)
    return (theta**(alpha - 1)) * ((1 - theta)**(beta - 1)) / B

# ================================
# Likelihood: Binomial
# ================================
def binomial_likelihood(theta, h, n):
    coeff = math.comb(n, h)
    return coeff * (theta**h) * ((1 - theta)**(n - h))

# ================================
# Bayesian Inference Function
# ================================
def bayesian_inference(theta_vals, alpha, beta, h, n):
    posterior = []
    
    for theta in theta_vals:
        prior = beta_prior(theta, alpha, beta)
        likelihood = binomial_likelihood(theta, h, n)
        posterior.append(prior * likelihood)
    
    posterior = np.array(posterior)
    
    # Normalize posterior
    posterior /= np.trapezoid(posterior, theta_vals)

    
    return posterior

# ================================
# Example usage (TEST)
# ================================
theta_vals = np.linspace(0, 1, 1000)

# Prior parameters
alpha = 2
beta = 5

# Data (example)
h = 7   # heads
n = 10  # flips

posterior = bayesian_inference(theta_vals, alpha, beta, h, n)

# ================================
# Plot
# ================================
plt.figure(figsize=(8,6))

plt.plot(theta_vals, posterior, label="Posterior Distribution", linewidth=2)

plt.xlabel("Theta (Probability of Heads)")
plt.ylabel("Probability Density")
plt.title("Posterior Distribution for Coin Toss (Bayesian Inference)")

plt.legend()
plt.grid()

# Save figure to folder
plt.savefig("figures/posterior_distribution.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()
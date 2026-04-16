
import numpy as np
import math
import matplotlib.pyplot as plt

# ================================
# Gamma + Beta (manual)
# ================================
def gamma_func(x):
    return math.gamma(x)

def beta_func(alpha, beta):
    return gamma_func(alpha) * gamma_func(beta) / gamma_func(alpha + beta)

# ================================
# Prior (Beta)
# ================================
def beta_prior(theta, alpha, beta):
    B = beta_func(alpha, beta)
    return (theta**(alpha - 1)) * ((1 - theta)**(beta - 1)) / B

# ================================
# Likelihood (Binomial)
# ================================
def binomial_likelihood(theta, h, n):
    coeff = math.comb(n, h)
    return coeff * (theta**h) * ((1 - theta)**(n - h))

# ================================
# Bayesian Inference
# ================================
def bayesian_inference(theta_vals, alpha, beta, h, n):
    posterior = []
    
    for theta in theta_vals:
        prior = beta_prior(theta, alpha, beta)
        likelihood = binomial_likelihood(theta, h, n)
        posterior.append(prior * likelihood)
    
    posterior = np.array(posterior)
    
    # Normalize (NEW NUMPY FIX)
    posterior /= np.trapezoid(posterior, theta_vals)
    
    return posterior

# ================================
# Load data (2b)
# ================================
data = np.loadtxt("HW06_data.txt")

h = int(np.sum(data))
n = len(data)

print("Heads:", h)
print("Total flips:", n)
print("Estimated theta:", h/n)

theta_vals = np.linspace(0, 1, 1000)

# Prior
alpha = 2
beta = 5

# ================================
# (b) Posterior using full data
# ================================
posterior = bayesian_inference(theta_vals, alpha, beta, h, n)

plt.figure(figsize=(8,6))
plt.plot(theta_vals, posterior, linewidth=2)
plt.xlabel("Theta")
plt.ylabel("Probability Density")
plt.title("Posterior Distribution (Full Data)")
plt.grid()

plt.savefig("figures/posterior_data.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================
# (c) Different data sizes
# ================================
sizes = [5, 50, 500]

plt.figure(figsize=(8,6))

for size in sizes:
    subset = data[:size]
    h_sub = int(np.sum(subset))
    n_sub = len(subset)
    
    posterior_sub = bayesian_inference(theta_vals, alpha, beta, h_sub, n_sub)
    
    plt.plot(theta_vals, posterior_sub, label=f"{size} flips")

plt.xlabel("Theta")
plt.ylabel("Probability Density")
plt.title("Posterior for Different Data Sizes")
plt.legend()
plt.grid()

plt.savefig("figures/posterior_sizes.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================
# (d) Different priors
# ================================
priors = [(1,1), (2,5), (10,10)]

plt.figure(figsize=(8,6))

for (a, b) in priors:
    posterior_prior = bayesian_inference(theta_vals, a, b, h, n)
    plt.plot(theta_vals, posterior_prior, label=f"Beta({a},{b})")

plt.xlabel("Theta")
plt.ylabel("Probability Density")
plt.title("Effect of Different Priors")
plt.legend()
plt.grid()

plt.savefig("figures/posterior_priors.png", dpi=300, bbox_inches='tight')
plt.show()
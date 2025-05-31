import jax
import jax.numpy as jnp  # JAX NumPy
import optax  # Optimizers
import equinox as eqx  # Equinox

from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline

from sklearn.datasets import make_moons
# For evaluation
from scipy.special import kl_div
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import gaussian_kde

# Gaussian mixture
def generate_gmm(n_samples, n_dimensions, n_components=4):
    # Create and initialize GMM with random parameters
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=0,
        means_init=np.random.randn(n_components, n_dimensions)  # Random initial means
    )
    # Generate dummy data for fitting (doesn't affect final sampling)
    dummy_data = np.random.randn(10, n_dimensions)
    gmm.fit(dummy_data)  # Required to initialize covariance matrices
    samples, _ = gmm.sample(n_samples)
    return jnp.array(samples)


n_feature = 2  # Set to desired dimension
# data = generate_gmm(n_samples=10000, n_dimensions=n_feature, n_components=5)
data = jnp.array(make_moons(n_samples=10000, noise=0.05)[0])

key, subkey = jax.random.split(jax.random.PRNGKey(1))


# Organize models in a dictionary
models = {
    "RealNVP": {
        "model": RealNVP(
            n_features=n_feature,
            n_layers=10,
            n_hidden=100,
            key=subkey,
            data_mean=jnp.mean(data, axis=0),
            data_cov=jnp.cov(data.T),
        ),
        "color": "#ff7f0e"
    },
    "Masked RQSpline": {
        "model": MaskedCouplingRQSpline(
            n_features=n_feature,
            n_layers=8,
            hidden_size=[64, 64],
            num_bins=8,
            key=subkey,
            # data_cov=jnp.cov(data.T),
            # data_mean=jnp.mean(data, axis=0),
        ),
        "color": "#2ca02c"
    }
}

# Training configuration
train_config = {
    "num_epochs": 100,
    "batch_size": 1000,
    "learning_rate": 0.001
}

def compute_hist_kl(data1, data2, bins=50):
    hist1, edges = jnp.histogramdd(data1, bins=bins, density=True)
    hist2, _ = jnp.histogramdd(data2, bins=edges, density=True)
    hist1 += 1e-8  # Avoid log(0)
    hist2 += 1e-8
    return jnp.sum(kl_div(hist1, hist2))

# def compute_model_kl(true_data, model_samples, model):
#     """
#     Compute KL(P||Q) = E_P[logP - logQ] using:
#     - KDE for P (true_data distribution)
#     - Model's log_prob for Q (learned distribution)
#     """
#     # Convert to numpy for KDE
#     true_data_np = np.array(true_data)
#     model_samples_np = np.array(model_samples)
    
#     # Fit KDE to true data (P)
#     kde_p = KernelDensity(bandwidth=0.1).fit(true_data_np)
    
#     # Calculate terms
#     log_p = kde_p.score_samples(true_data_np)  # logP(x~P)
#     log_q = jax.vmap(model.log_prob)(true_data)  # logQ(x~P)
    
#     # Compute KL divergence
#     kl = np.mean(log_p - log_q)
#     return kl



# Train models and store results
results = {}
for model_name, config in models.items():
    print(f"\n=== Training {model_name} model ===")
    
    # Initialize optimizer
    optim = optax.adam(train_config["learning_rate"])
    state = optim.init(eqx.filter(config["model"], eqx.is_array))
    
    # Train model
    key, trained_model, state, losses = config["model"].train(
        key,
        data,
        optim,
        state,
        train_config["num_epochs"],
        train_config["batch_size"],
        verbose=True
    )
    
    # Generate samples
    _, sample_key = jax.random.split(key)
    nf_samples = trained_model.sample(sample_key, 1000)
    
    # Calculate KL divergence
    kl_value = compute_model_kl(data, nf_samples,trained_model)
    # kl_value = compute_hist_kl(data, nf_samples)
    # Store results
    results[model_name] = {
        "samples": nf_samples,
        "kl": kl_value,
        "color": config["color"],
        "losses": losses
    }

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Common axis limits
all_samples = jnp.concatenate([r["samples"] for r in results.values()])
x_min = min(jnp.min(data[:, 0]), jnp.min(all_samples[:, 0])) - 0.5
x_max = max(jnp.max(data[:, 0]), jnp.max(all_samples[:, 0])) + 0.5
y_min = min(jnp.min(data[:, 1]), jnp.min(all_samples[:, 1])) - 0.5
y_max = max(jnp.max(data[:, 1]), jnp.max(all_samples[:, 1])) + 0.5

# Plot each model's results
for ax, (model_name, result) in zip(axes, results.items()):
    ax.scatter(data[:, 0], data[:, 1], 
               s=10, alpha=0.3, c='#1f77b4', label='True Data')
    ax.scatter(result["samples"][:, 0], result["samples"][:, 1],
               s=15, alpha=0.5, marker='x', 
               c=result["color"], label='Generated Samples')
    ax.set_title(f'{model_name}\nKL: {result["kl"]:.2f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("flowMC_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Print KL divergence table
print("\nModel Performance Comparison:")
print("{:<20} {:<10}".format('Model', 'KL Divergence'))
print("-" * 30)
for model_name, result in results.items():
    print("{:<20} {:<10.4f}".format(model_name, result["kl"]))


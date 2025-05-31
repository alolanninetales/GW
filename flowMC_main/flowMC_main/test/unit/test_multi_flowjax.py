import jax
import jax.numpy as jnp  # JAX NumPy
import optax  # Optimizers
import equinox as eqx  # Equinox
# flowjax
from flowMC.resource.nf_model.flowjax import NFModelFlowJAX
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import jax.random as jr
import numpy as np
from scipy.special import kl_div
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

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
    
    # Now generate actual samples from the initialized distribution
    samples, _ = gmm.sample(n_samples)
    return jnp.array(samples)

# data = generate_gmm(n_samples=10000, n_dimensions=4, n_components=5)
# Load data
data = jnp.array(make_moons(n_samples=10000, noise=0.05)[0])

# Model parameters
n_feature = 2  # Set to desired dimension
# n_layers = 10
# n_hidden = 100

key, subkey = jax.random.split(jax.random.key(0), 2)

models = {
    "Coupling": {
        "model": NFModelFlowJAX(
            dim=n_feature,
            key=subkey,
            flow_type="coupling",
            # n_layers=10
        ),
        "color": "#ff7f0e"
    },
    "triangular": {
        "model": NFModelFlowJAX(
            dim=n_feature,
            key=subkey,
            flow_type="triangular",
            # n_layers=10
        ),
        "color": "#1f77b4"
    },
    "Masked Autoregressive": {
        "model": NFModelFlowJAX(
            dim=n_feature,
            key=subkey,
            flow_type="masked_autoregressive",
            # n_layers=10
        ),
        "color": "#d62728"
    }
}


# def compute_hist_kl(data1, data2, bins=50):
#     hist1, edges = jnp.histogramdd(data1, bins=bins, density=True)
#     hist2, _ = jnp.histogramdd(data2, bins=edges, density=True)
#     hist1 += 1e-8  # Avoid log(0)
#     hist2 += 1e-8
#     return jnp.sum(kl_div(hist1, hist2))
def compute_model_kl(true_data, model_samples, model):
    """
    Compute KL(P||Q) = E_P[logP - logQ] using:
    - KDE for P (true_data distribution)
    - Model's log_prob for Q (learned distribution)
    """
    # Convert to numpy for KDE
    true_data_np = np.array(true_data)
    model_samples_np = np.array(model_samples)
    
    # Fit KDE to true data (P)
    kde_p = KernelDensity(bandwidth=0.1).fit(true_data_np)
    
    # Calculate terms
    log_p = kde_p.score_samples(true_data_np)  # logP(x~P)
    log_q = jax.vmap(model.log_prob)(true_data)  # logQ(x~P)
    
    # Compute KL divergence
    kl = np.mean(log_p - log_q)
    return kl


# Train models and store results
results = {}
for model_name, config in models.items():
    print(f"\n=== Training {model_name} model ===")
    
    # Initialize fresh optimizer
    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(config["model"], eqx.is_array))
    
    # Train model
    final_key, trained_model, _, losses = config["model"].train(
        key,
        data,
        optimizer,
        state,
        num_epochs=100,
        batch_size=100
    )
    
    # Generate samples
    sample_key = jax.random.fold_in(key, 0)
    samples = trained_model.sample(sample_key, 1000)
    
    # Calculate KL divergence
    kl_value = compute_model_kl(data, samples, trained_model)
    
    # Store results
    results[model_name] = {
        "samples": samples,
        "kl": kl_value,
        "color": config["color"],
        "losses": losses
    }
    
    # Save model
    eqx.tree_serialise_leaves(f"trained_{model_name.lower().replace(' ', '_')}_model.eqx", trained_model)

# Create individual comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Common axis limits
all_samples = jnp.concatenate([r["samples"] for r in results.values()])
x_min = min(jnp.min(data[:, 0]), jnp.min(all_samples[:, 0])) - 0.5
x_max = max(jnp.max(data[:, 0]), jnp.max(all_samples[:, 0])) + 0.5
y_min = min(jnp.min(data[:, 1]), jnp.min(all_samples[:, 1])) - 0.5
y_max = max(jnp.max(data[:, 1]), jnp.max(all_samples[:, 1])) + 0.5

# Plot each model's results
for ax, (model_name, result) in zip(axes, results.items()):
    # Plot true data
    ax.scatter(data[:, 0], data[:, 1], 
               s=10, alpha=0.3, c='#1f77b4', label='True Data')
    
    # Plot generated samples
    ax.scatter(result["samples"][:, 0], result["samples"][:, 1],
               s=15, alpha=0.5, marker='x', 
               c=result["color"], label='Generated Samples')
    
    ax.set_title(f'{model_name}\nKL Divergence: {result["kl"]:.2f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("individual_model_comparisons.png", dpi=300, bbox_inches='tight')
plt.show()

# Print KL divergence table
print("\nModel Performance Comparison:")
print("{:<25} {:<10}".format('Model', 'KL Divergence'))
print("-" * 35)
for model_name, result in results.items():
    print("{:<25} {:<10.4f}".format(model_name, result["kl"]))


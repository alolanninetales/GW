import jax
import jax.numpy as jnp  # JAX NumPy
import optax  # Optimizers
import equinox as eqx  # Equinox

# flowMC
from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
# flowjax
from flowMC.resource.nf_model.flowjax import NFModelFlowJAX
# Data and plotting
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import jax.random as jr

# For evaluation
from scipy.special import kl_div

# Load data
data = jnp.array(make_moons(n_samples=10000, noise=0.05)[0])

# Model parameters
n_feature = 2
n_layers = 10
n_hidden = 100

key, subkey = jax.random.split(jax.random.key(0), 2)

#flowjax model
coupling_model = NFModelFlowJAX(
    dim=2,
    key=subkey,
    flow_type="coupling",  # Explicitly specify flow type
    n_layers=10,
    # knots=64,  # Example custom parameter
    # nn_width=256
)
# bnaf_model = NFModelFlowJAX(
#     dim=2,
#     key=subkey,
#     flow_type="bnaf",  # Explicitly specify flow type
#     n_layers=10,
#     # knots=64,  # Example custom parameter
#     # nn_width=256
# )
triangular_model = NFModelFlowJAX(
    dim=2,
    key=subkey,
    flow_type="triangular",  # Explicitly specify flow type
    n_layers=10,
    # knots=64,  # Example custom parameter
    # nn_width=256
)

masked_autoregressive_model = NFModelFlowJAX(
    dim=2,
    key=subkey,
    flow_type="masked_autoregressive", 
    n_layers=10,
)


#flowMC model



optimizer = optax.adam(1e-3)
state = optimizer.init(eqx.filter(triangular_model, eqx.is_array))

# Train model
final_key, trained_model, _, losses = triangular_model.train(
    key,
    data,
    optimizer,
    state,
    num_epochs=200,
    batch_size=100
)

# Sample from trained model
sample_key = jax.random.fold_in(key, 0)
# samples = trained_model.flow.sample(sample_key, (5000,))
samples = trained_model.sample(sample_key, 1000)  # Generate more samples for analysis


# Compute KL divergence (approximate)
# Note: Since we don't have the true PDF of 'make_moons', this is a simplified comparison.
# Here we compare histograms as discrete distributions.
def compute_hist_kl(data1, data2, bins=50):
    hist1, edges = jnp.histogramdd(data1, bins=bins, density=True)
    hist2, _ = jnp.histogramdd(data2, bins=edges, density=True)
    hist1 += 1e-8  # Avoid log(0)
    hist2 += 1e-8
    return jnp.sum(kl_div(hist1, hist2))

kl_value = compute_hist_kl(data, samples)
print(f"Approximate KL Divergence between true data and generated samples: {kl_value:.4f}")

# Optional: Save model or samples
jnp.save("true_data.npy", data)
jnp.save("generated_samples.npy", samples)
eqx.tree_serialise_leaves("trained_nf_model.eqx", trained_model)


# Plotting results
plt.figure(figsize=(8, 6))

# Plot both datasets on the same axes
plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6,  c='#1f77b4', label='True Data (Make Moons)')

plt.scatter(samples[:, 0], samples[:, 1],s=10,alpha=0.6,c='#ff7f0e',  marker='x',label='Generated Samples')

plt.title('Comparison of True Data and Generated Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.gca().set_facecolor('#f0f0f0')
plt.grid(True, alpha=0.3)

x_min = min(jnp.min(data[:, 0]), jnp.min(samples[:, 0])) - 0.5
x_max = max(jnp.max(data[:, 0]), jnp.max(samples[:, 0])) + 0.5
y_min = min(jnp.min(data[:, 1]), jnp.min(samples[:, 1])) - 0.5
y_max = max(jnp.max(data[:, 1]), jnp.max(samples[:, 1])) + 0.5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.savefig("nf_moons_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

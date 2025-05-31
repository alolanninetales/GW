from typing import Optional, Callable
from flowMC.resource.nf_model.base import NFModel
from flowjax.flows import (
    block_neural_autoregressive_flow,
    coupling_flow,
    masked_autoregressive_flow,
    planar_flow,
    triangular_spline_flow
)
from flowjax.distributions import Transformed, Normal
from flowjax.bijections import RationalQuadraticSpline, Affine
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from flowjax.train import fit_to_data
import optax
import jax
from functools import partial
from typing_extensions import Self
import equinox as eqx  # Equinox
# from sklearn.datasets import make_moons

class NFModelFlowJAX(NFModel):
    flow: Transformed
    _data_mean: jax.Array
    data_std: jax.Array

    def __init__(
        self,
        dim: int,
        key: PRNGKeyArray,
        flow_type: str = "coupling",
        n_layers: int = 8,
        invert: bool = True,
        **flow_kwargs
    ):
        """
        FlowJAX wrapper supporting multiple flow architectures.

        Args:
            dim (int): Data dimension
            key (PRNGKeyArray): JAX random key
            flow_type (str): Type of flow (coupling|bnaf|masked_autoregressive|planar|triangular)
            n_layers (int): Number of flow layers
            invert (bool): Prioritize log_prob (True) or sampling (False) efficiency
            flow_kwargs: Architecture-specific parameters
        """
        self._n_features = dim
        self._data_mean = jnp.zeros(dim)
        self.data_std = jnp.ones(dim)
        self._data_cov = jnp.eye(dim)
        # Initialize base distribution
        # data = jnp.array(make_moons(n_samples=5000, noise=0.05)[0])
        # self.flow : Transformed = eqx.field(mutable=True)  # 声明为可变字段
        base_dist = Normal(jnp.zeros(dim))
        
        # Create flow architecture
        flow_constructors = {
            "coupling": coupling_flow,
            "bnaf": block_neural_autoregressive_flow,
            "masked_autoregressive": masked_autoregressive_flow,
            "planar": planar_flow,
            "triangular": triangular_spline_flow
        }
        
        constructor = flow_constructors.get(flow_type.lower())
        if constructor is None:
            raise ValueError(f"Unsupported flow type: {flow_type}")
            
        self.flow = constructor(
            key=key,
            base_dist=base_dist,
            flow_layers=n_layers,
            invert=invert,
            **self._get_flow_params(flow_type, flow_kwargs)
        )

    def _get_flow_params(self, flow_type: str, kwargs: dict) -> dict:
        """Get architecture-specific parameters"""
        params_map = {
            "coupling": {
                # "transformer": RationalQuadraticSpline(knots=128, interval=10),
                # "nn_width": 50,
                # "nn_depth": 1
            },
            "bnaf": {
                "nn_depth": 2,
                "nn_block_dim": 8,
                "activation": jax.nn.tanh
            },
            "masked_autoregressive": {
                # "transformer": Affine(),
                # "nn_width": 128,
                # "nn_depth": 2
            },
            "planar": {
                "negative_slope": 0.01,
                "mlp_kwargs": {"width_size": 64, "depth": 2}
            },
            "triangular": {
                "knots": 16,
                "tanh_max_val": 3.0
            }
        }
        default_params = params_map.get(flow_type, {})
        return {**default_params, **kwargs}

    def _normalize(self, x: Array) -> Array:
        """Z-score normalization"""
        return (x - self.data_mean) / self.data_std

    def _denormalize(self, x: Array) -> Array:
        """Reverse normalization"""
        return x * self.data_std + self.data_mean

    def log_prob(self, x: Float[Array, " n_dim"]) -> Float:
        return self.flow.log_prob(self._normalize(x))

    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> Array:
        # return self._denormalize(self.flow.sample(rng_key, (n_samples,)))
        # samples = self.flow.sample(rng_key, (n_samples,))
        # L = jnp.linalg.cholesky(self.data_cov)
        # return samples @ L.T + self.data_mean
        return self.flow.sample(rng_key, (n_samples,))
    def train(
        self,
        rng: PRNGKeyArray,
        data: Array,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int = 200,
        batch_size: int = 100,
        verbose: bool = True,
        
    ) -> tuple[PRNGKeyArray, Self, optax.OptState, Array]:
        """Train using flowjax's fit_to_data with proper state handling"""
        # Update normalization parameters
        new_model = eqx.tree_at(
            lambda m: (m._data_mean, m._data_cov),
            self,
            (jnp.mean(data, axis=0), jnp.cov(data.T))
        )
        # self._data_mean = jnp.mean(data, axis=0)
        # self._data_cov = jnp.cov(data.T)
        L = jnp.linalg.cholesky(self._data_cov)
        normalized_data = (data - self._data_mean) @ jnp.linalg.inv(L)

        # Train with flowjax's built-in method
        trained_flow, training_info = fit_to_data(
            key=rng,
            dist=self.flow,
            x=normalized_data,
            optimizer=optim,
            max_epochs=num_epochs,
            max_patience = 20,
            batch_size=batch_size,
            show_progress=verbose
        )
        final_model = eqx.tree_at(
            lambda m: m.flow,
            new_model,
            trained_flow
        )
        # Update model parameters while maintaining other state
        # self.flow = trained_flow
        loss_values = jnp.array(getattr(training_info, "loss_history", jnp.zeros(num_epochs)))

        return (
            rng,
            final_model,
            state,
            loss_values
        )
    
    def print_parameters(self):
        print("NFModelFlowJAX parameters:")
        print(f"Data mean: {self._data_mean}")
        print(f"Data covariance: {self._data_cov}")

    def forward(
        self, 
        x: Float[Array, " n_dim"], 
        key: Optional[PRNGKeyArray] = None
    ) -> tuple[Float[Array, " n_dim"], Float]:
        """Forward transformation: data space -> latent space"""
        # Normalize input
        x_norm = self._normalize(x)
        
        # Apply flow transformation
        z, log_det = self.flow.bijection.transform(x_norm)
        
        # Add normalization log determinant
        log_det -= jnp.sum(jnp.log(self.data_std))  # Jacobian from normalization
        
        return z, log_det

    def inverse(
        self, 
        x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], Float]:
        """Inverse transformation: latent space -> data space"""
        # Apply inverse flow transformation
        x_norm, log_det = self.flow.bijection.inverse(x)
        
        # Denormalize output
        z = self._denormalize(x_norm)
        
        # Add denormalization log determinant
        log_det += jnp.sum(jnp.log(self.data_std))  # Jacobian from denormalization
        
        return z, log_det
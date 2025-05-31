from typing import Optional
from flowMC.resource.nf_model.base import Distribution, NFModel
from flowjax.distributions import Transformed, Normal
from flowjax.bijections import Chain, Coupling, RationalQuadraticSpline, Affine
from flowjax.train import fit_to_data
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import optax
import jax
from functools import partial
from typing_extensions import Self

class NFModelFlowJAX(NFModel):
    chain: Chain 
    base_dist:  Normal
    flow: Transformed

    def __init__(
        self,
        dim: int,
        key: PRNGKeyArray,
        n_layers: int = 10,
        flow_type: str = "RationalQuadraticCoupling",
        **kwargs
    ):
        # super().__init__()
        self._n_features = dim  
        self._data_mean = jnp.zeros(dim)
        self._data_cov = jnp.eye(dim)

        # self.bijections = []
        _layers = []
        for _ in range(n_layers):
            key, subkey = jr.split(key)
            _layers.append(
                self._create_bijection(subkey, flow_type, dim=dim, **kwargs)
                )
        
        self.chain = Chain(_layers)
        self.base_dist = Normal(jnp.zeros(dim))
        self.flow = Transformed(base_dist=self.base_dist, bijection=self.chain)

    def _create_bijection(self, key, flow_type, dim, **kwargs):
        if flow_type == "RationalQuadraticCoupling":
            return Coupling(
                key=key,
                transformer=RationalQuadraticSpline(
                    knots=kwargs.get("spline_knots", 128),
                    interval=kwargs.get("interval", 10)
                ),
                untransformed_dim=dim//2,
                dim=dim,
                nn_width=kwargs.get("nn_width", 128),
                nn_depth=kwargs.get("nn_depth", 100)
            )
        elif flow_type == "AffineCoupling":
            return Coupling(
                key=key,
                transformer=Affine(),
                untransformed_dim=dim//2,
                dim=dim,
                nn_width=kwargs.get("nn_width", 128),
                nn_depth=kwargs.get("nn_depth", 2)
            )
        else:
            raise ValueError(f"Unsupported flow type: {flow_type}")

    def log_prob(self, x: Float[Array, " n_dim"]) -> Float:

        L = jnp.linalg.cholesky(self.data_cov)
        normalized_x = (x - self.data_mean) @ jnp.linalg.inv(L)
        return self.flow.log_prob(normalized_x).squeeze()

    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> Array:
        samples = self.flow.sample(rng_key, (n_samples,))
        L = jnp.linalg.cholesky(self.data_cov)
        return samples @ L.T + self.data_mean 

    def forward(
        self, 
        x: Float[Array, " n_dim"], 
        key: Optional[PRNGKeyArray] = None
    ) -> tuple[Float[Array, " n_dim"], Float]:
        L = jnp.linalg.cholesky(self.data_cov)
        normalized_x = (x - self.data_mean) @ jnp.linalg.inv(L)
        z, log_det = self.chain.transform(normalized_x)
        return z, log_det

    def inverse(
        self, 
        x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], Float]:
        x, log_det = self.chain.inverse(x)
        L = jnp.linalg.cholesky(self.data_cov)
        return x @ L.T + self.data_mean, log_det

    def train(
        self,
        rng: PRNGKeyArray,
        data: Array,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> tuple[PRNGKeyArray, Self, optax.OptState, Array]:

        data_mean = jnp.mean(data, axis=0)
        data_cov = jnp.cov(data.T)
        L = jnp.linalg.cholesky(data_cov)
        normalized_data = (data - data_mean) @ jnp.linalg.inv(L)

        flow, training_info = fit_to_data(
            dist = self.flow,
            x = normalized_data,
            optimizer=optim,
            max_epochs=num_epochs,
            batch_size=batch_size,
            # verbose=verbose,
            key=rng,
            max_patience=100
        )

        loss_values = jnp.array(getattr(training_info, "loss_history", jnp.zeros(num_epochs)))

        return rng, self, state, loss_values
    
    def print_parameters(self):
        print("NFModelFlowJAX parameters:")
        print(f"Data mean: {self.data_mean}")
        print(f"Data covariance: {self.data_cov}")




class NFModelFlowJAX111(NFModel):

    def __init__(
        self,
        dim: int,
        key: PRNGKeyArray,
        n_layers: int = 3,
        flow_type: str = "RationalQuadraticCoupling",
        **kwargs
    ):
        self.dim = dim
        self._n_features = dim
        self._data_mean = jnp.zeros(dim)
        self._data_cov = jnp.eye(dim)
        
        self.bijections = []
        for _ in range(n_layers):
            key, subkey = jr.split(key)
            self.bijections.append(
                self._create_bijection(subkey, flow_type, **kwargs)
            )
        
        self.chain = Chain(self.bijections)
        self.base_dist = Normal(jnp.zeros(dim))
        self.flow = Transformed(self.base_dist, self.chain)

    def _create_bijection(self, key, flow_type, **kwargs):
        if flow_type == "RationalQuadraticCoupling":
            return Coupling(
                key=key,
                transformer=RationalQuadraticSpline(
                    knots=kwargs.get("spline_knots", 128),
                    interval=kwargs.get("interval", 10)
                ),
                untransformed_dim=self.dim//2,
                dim=self.dim,
                nn_width=kwargs.get("nn_width", 128),
                nn_depth=kwargs.get("nn_depth", 2)
            )
        elif flow_type == "AffineCoupling":
            return Coupling(
                key=key,
                transformer=Affine(),
                untransformed_dim=self.dim//2,
                dim=self.dim,
                nn_width=kwargs.get("nn_width", 128),
                nn_depth=kwargs.get("nn_depth", 2)
            )
        else:
            raise ValueError(f"not supported: {flow_type}")

    def log_prob(self, x: Float[Array, " n_dim"]) -> Float:
        x = (x - self.data_mean) / jnp.sqrt(jnp.diag(self.data_cov))
        y, log_det = self.__call__(x)
        log_det = log_det + jax.scipy.stats.multivariate_normal.logpdf(
            y, jnp.zeros(self.n_features), jnp.eye(self.n_features)
        )
        return log_det

    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> jnp.ndarray:
        return self.flow.sample(rng_key, (n_samples,)).reshape(n_samples, -1)

    def forward1(self, x: jnp.ndarray, key: PRNGKeyArray) -> tuple[jnp.ndarray, float]:
        z, log_det = self.chain.transform(x)

        return z, log_det
    def forward(self, x: Float[Array, " n_dim"], key: Optional[PRNGKeyArray] = None):
        z, log_det = self.chain.transform(x)
        return z,log_det
    
    def inverse(
        self, x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], Float]:
        x, log_det = self.chain.inverse(z)
        return x, log_det

    # def inverse(self, z: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    #     x, log_det = self.chain.inverse(z)
    #     return x, log_det

    def train(self, rng: PRNGKeyArray, data: jnp.ndarray, optim: optax.GradientTransformation, **kwargs):
        self._data_mean = jnp.mean(data, axis=0)
        self._data_cov = jnp.cov(data.T)
        normalized_data = (data - self._data_mean) @ jnp.linalg.inv(jnp.linalg.cholesky(self._data_cov))
        # optimizer = optax.adam(1e-4)
        self.flow, _ = fit_to_data(
            dist=self.flow,
            x = normalized_data,
            optimizer=optim,
            **kwargs
        )

import jax
import jax.numpy as jnp
import numpyro
import tqdm
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.handlers import seed
from numpyro.infer import MCMC, NUTS, HMC

import numpy as np
from tensorflow.keras.datasets import mnist
from flax import linen as nn
from flax.training import train_state
import optax

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], 28*28)
        x = nn.Dense(features=10)(x)
        return x

def train_nn(learning_rate, train_images, train_labels, num_epochs=10):
    key = jax.random.PRNGKey(0)
    
    model = CNN()
    params = model.init(key, jnp.ones((1, 28, 28)))
    
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(params, batch['image'])
            return jnp.mean(optax.softmax_cross_entropy(logits, batch['label']))
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    for _ in tqdm.tqdm(range(num_epochs)):
        for i in range(0, len(train_images), 32):
            batch = {
                'image': train_images[i:i+32],
                'label': jax.nn.one_hot(train_labels[i:i+32], 10)  # One-hot encode the labels
            }
            state, loss = train_step(state, batch)
    
    return state

# Define the Bayesian model
def model(images, labels):
    lr = numpyro.sample("lr", dist.Gamma(1.0, 1.0 / 0.01))
    
    state = train_nn(lr, images, labels)
    
    logits = state.apply_fn(state.params, images)
    numpyro.sample("obs", dist.Categorical(logits=logits), obs=labels)

print("Start Loading Mnist")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

def run_svi(model, images, labels):
    guide = autoguide.AutoDelta(model)
    optimizer = numpyro.optim.Adam(0.001)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(jax.random.PRNGKey(0), 2, images=images, labels=labels)
    params = svi_results.params

    return params, guide

if __name__ == "__main__":
    # Subset of data for faster computation
    print("Start inference")
    n_subset = 1000
    train_images_subset = train_images[:n_subset]
    train_labels_subset = train_labels[:n_subset]

    print(train_labels_subset)

    train_results = train_nn(0.001, train_images_subset, train_labels_subset)
    predictions = train_results.apply_fn(train_results.params, train_images_subset)
    
    log_probs = []
    for i in tqdm.tqdm(range(n_subset)):
        cat_dist = dist.Categorical(logits=predictions[i])
        log_prob = cat_dist.log_prob(train_labels_subset[i])
        log_probs.append(log_prob)
    
    log_probs = jnp.array(log_probs)
    
    print(f"Mean log probability: {jnp.mean(log_probs):.4f}")
    print(f"Min log probability: {jnp.min(log_probs):.4f}")
    print(f"Max log probability: {jnp.max(log_probs):.4f}")

    # Vectorized computation of log probabilities
    cat_dist = dist.Categorical(logits=predictions)
    log_probs = cat_dist.log_prob(train_labels_subset)
    
    print(f"Mean log probability: {jnp.mean(log_probs):.4f}")
    print(f"Min log probability: {jnp.min(log_probs):.4f}")
    print(f"Max log probability: {jnp.max(log_probs):.4f}")

    run_svi(model, train_images_subset, train_labels_subset)
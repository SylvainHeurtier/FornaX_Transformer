import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
import matplotlib.pyplot as plt
import tensorflow as tf  # Only for data loading
import json
import pickle

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# Import constants from config file
from Constantes import BATCH_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, name_dir

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                            TRANSFORMER MODEL
#///////////////////////////////////////////////////////////////////////////////////////////////////////

#////////// Load Configuration /////////

# Load model configuration
with open(f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/constantes_du_modele.json", 'r') as f:
    config = json.load(f)

# Extract values
MAX_SOURCES  = config["MAX_SOURCES"]
MAX_CLUSTERS = config["MAX_CLUSTERS"]
VOCAB_SIZE   = config["VOCAB_SIZE"]
PAD_TOKEN    = config["PAD_TOKEN"]
SEP_TOKEN    = config["SEP_TOKEN"]
CLS_TOKEN    = config["CLS_TOKEN"]

# Print configuration
print("┌───────────────────────────────┐")
print("│  MODEL CONFIGURATION          │")
print("├───────────────────────────────┤")
for key, value in config.items():
    print(f"│ {key.ljust(15)}: {str(value).rjust(10)}   │")
print("└───────────────────────────────┘")

# Load training and test data (using tf for data loading but convert to jnp immediately)
X_train = jnp.array(tf.io.read_file(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.txt'))
X_test = jnp.array(tf.io.read_file(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.txt'))

print(f"\nlen(X_train) = {len(X_train)}")
print(f"len(X_test)  = {len(X_test)}")

print("\nTrain - Min:", jnp.min(X_train), "Max:", jnp.max(X_train))
print("Test - Min:", jnp.min(X_test), "Max:", jnp.max(X_test))

#////////// Transformer Architecture /////////

class MLP(nn.Module):
    """Multi-Layer Perceptron with ReLU activation"""
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x

class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and MLP"""
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask):
        # Multi-head self-attention
        z = nn.LayerNorm()(x)
        z = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(z, mask=mask)
        x = x + z  # Residual connection
        z = nn.LayerNorm()(x)
        x += MLP(self.d_model)(z)  # Feed-forward with residual
        return x

class AutoregressiveTransformerModel(nn.Module):
    """Autoregressive Transformer model for sequence generation"""
    d_model: int
    num_heads: int
    num_layers: int
    seq_length: int
    vocab_size: int

    @nn.compact
    def __call__(self, x):
        # x shape: [batch_size, seq_length]
        batch_size, seq_len = x.shape

        # Causal mask (1 to keep, 0 to block)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]  # [1, 1, seq_len, seq_len]
        mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, seq_len, seq_len))

        # Token embeddings
        x = nn.Embed(self.vocab_size, self.d_model)(x)  # [batch, seq_len, d_model]
        
        # Positional embeddings
        positions = jnp.arange(seq_len)
        pos_embed = nn.Embed(self.seq_length, self.d_model)(positions)  # [seq_len, d_model]
        x += pos_embed[None, :, :]  # Add to token embeddings

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads)(x, mask=mask)

        # Final layer norm
        x = nn.LayerNorm()(x)

        # Prediction head
        logits = nn.Dense(self.vocab_size)(x)
        return logits

#////////// Training Setup /////////

def initialize_sequences(batch_size, seq_length, rng_key):
    """Initialize sequences with CLS token followed by random tokens"""
    # First token = CLS_TOKEN for all sequences
    cls_tokens = jnp.full((batch_size, 1), CLS_TOKEN, dtype=jnp.int32)

    # Random tokens for the rest (excluding special tokens)
    rng, _ = random.split(rng_key)
    random_tokens = random.randint(
        rng,
        (batch_size, seq_length - 1),  # -1 because CLS takes first position
        0,
        VOCAB_SIZE - 3,  # Range [0, 1023]
        dtype=jnp.int32
    )

    return jnp.concatenate([cls_tokens, random_tokens], axis=1)

# Initialize model
seq_length = X_train.shape[1]
model = AutoregressiveTransformerModel(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_length=seq_length,
    vocab_size=VOCAB_SIZE
)

def loss_fn(params, x):
    """Cross-entropy loss for autoregressive prediction"""
    # Predict next token given previous ones
    logits = model.apply(params, x[:, :-1])  # [batch, seq_len-1, vocab_size]
    targets = x[:, 1:].astype(jnp.int32)    # [batch, seq_len-1]
    return -jnp.mean(tfd.Categorical(logits=logits).log_prob(targets))

@jax.jit
def update_model(params, opt_state, x):
    """Single training step"""
    loss, grads = jax.value_and_grad(loss_fn)(params, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize model parameters
init_tokens = initialize_sequences(
    batch_size=BATCH_SIZE,
    seq_length=seq_length,
    rng_key=random.PRNGKey(0)
)
params = model.init(random.PRNGKey(0), init_tokens)

# Learning rate scheduler
total_steps = 30_000
lr_scheduler = optax.warmup_cosine_decay_schedule(
    3e-5,  # Initial LR
    1e-4,   # Peak LR after warmup
    3000,   # Warmup steps
    total_steps  # Total steps
)

# Optimizer with gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(lr_scheduler)
)
opt_state = optimizer.init(params)

# Create data iterators (using tf.data for efficiency but convert to jnp)
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_iterator = iter(train_dataset.shuffle(1000).batch(BATCH_SIZE).repeat())

test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_iterator = iter(test_dataset.batch(BATCH_SIZE).repeat())

#///////////////////////////////////////////////////////////////////////////////////////
#                                            TRAINING LOOP
#///////////////////////////////////////////////////////////////////////////////////////

# Early stopping parameters
patience = 50
best_val_loss = float('inf')
wait = 0
best_params = None

loss_history = []
val_loss_history = []

# Ensure we're using GPU
print("Using device:", jax.devices())

for i in range(total_steps):
    # Early stopping check
    if wait >= patience:
        print(f"\nEarly stopping at step {i}")
        print(f"Best val_loss: {best_val_loss:.4f}")
        break

    # Training step
    batch_x = jnp.array(next(train_iterator))
    params, opt_state, loss = update_model(params, opt_state, batch_x)
    loss_history.append(loss)

    # Validation every 100 steps
    if i % 100 == 0:
        batch_val = jnp.array(next(test_iterator))
        val_loss = loss_fn(params, batch_val)
        val_loss_history.append(val_loss)

        # Update best params if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            wait = 0
        else:
            wait += 1

        # Print metrics
        print(
            f"Step {i}: "
            f"Train Loss = {loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Best Val Loss = {best_val_loss:.4f}, "
            f"Wait = {wait}/{patience}"
        )

# Restore best parameters
if best_params is not None:
    params = best_params

#////////// Visualization and Saving /////////

# Prepare data for plotting
steps_train = jnp.arange(len(loss_history))
steps_val = jnp.arange(0, len(loss_history), 100)  # Every 100 steps
val_loss_values = val_loss_history[:len(steps_val)]  # Ensure same length

# Plot training curves
plt.figure(figsize=(8, 5))
plt.plot(steps_train, loss_history,
         marker='o', linestyle='-', color='blue',
         markersize=1, linewidth=1, label='Train Loss')

plt.plot(steps_val, val_loss_values,
         marker='s', linestyle='-', color='orange',
         markersize=3, linewidth=1.2, label='Validation Loss')

# Formatting
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss\n'
          f'BATCH_SIZE = {BATCH_SIZE}, D_MODEL = {D_MODEL}, '
          f'NUM_HEADS = {NUM_HEADS}, NUM_LAYERS = {NUM_LAYERS}\n')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xlim(0, len(loss_history) - 1)
plt.tight_layout()

# Save plot
plt.savefig(
    f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/training_curves_D_MODEL{D_MODEL}_NUM_LAYERS{NUM_LAYERS}_NUM_HEADS{NUM_HEADS}.png',
    dpi=300,
    bbox_inches='tight'
)

# Save model parameters
with open(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/params_D_MODEL{D_MODEL}_NUM_HEADS{NUM_HEADS}_NUM_LAYERS{NUM_LAYERS}.pkl', 'wb') as f:
    pickle.dump(params, f)

# Save model dimensions
dim_transformer = {
    "BATCH_SIZE": BATCH_SIZE,
    "D_MODEL": D_MODEL,
    "NUM_HEADS": NUM_HEADS,
    "NUM_LAYERS": NUM_LAYERS,
}

with open(f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/dim_transformer_D_MODEL{D_MODEL}_NUM_HEADS{NUM_HEADS}_NUM_LAYERS{NUM_LAYERS}.json", 'w') as f:
    json.dump(dim_transformer, f, indent=4)

print("Training complete and all artifacts saved")
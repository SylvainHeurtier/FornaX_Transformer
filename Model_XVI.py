import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json
import pickle
import numpy as np

# Import constants from config file
from Constantes import BATCH_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, name_dir, CLS_TOKEN, VOCAB_SIZE, NOMBRE_TOKENS_SPECIAUX

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                            TRANSFORMER MODEL
#///////////////////////////////////////////////////////////////////////////////////////////////////////


titre = "TRANSFORMER MODEL"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)

#////////// Load Configuration /////////
with open(f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/constantes_du_modele.json", 'r') as f:
    config = json.load(f)

MAX_SOURCES  = config["MAX_SOURCES"]
MAX_CLUSTERS = config["MAX_CLUSTERS"]
MAX_AGN = config["MAX_AGN"]

print("┌───────────────────────────────┐")
print("│  MODEL CONFIGURATION          │")
print("├───────────────────────────────┤")
for key, value in config.items():
    print(f"│ {key.ljust(15)}: {str(value).rjust(10)}   │")
print("└───────────────────────────────┘")

# Load data

X_train = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.txt', dtype=np.int32)
X_test = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.txt', dtype=np.int32)

#X_train = np.load(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.npy')
#X_test = np.load(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.npy')

print(f"\nDim de X_train: {X_train.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)
print(f"Dim de X_test: {X_test.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)


#////////// Transformer Architecture /////////
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mlp = keras.Sequential([
            layers.Dense(d_model, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, x, mask):
        attn_output = self.attn(x, x, attention_mask=mask)
        x = x + attn_output  # Residual
        x = self.layernorm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual
        return self.layernorm2(x)

class AutoregressiveTransformerModel(keras.Model):
    def __init__(self, d_model, num_heads, num_layers, seq_length, vocab_size):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.token_embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = layers.Embedding(seq_length, d_model)
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads) 
            for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(vocab_size)

    def call(self, x):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        
        # Causal mask
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        
        # Embeddings
        token_embeddings = self.token_embed(x)
        positions = tf.range(seq_len, dtype=tf.int32)
        pos_embeddings = self.pos_embed(positions)
        x = token_embeddings + pos_embeddings

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return self.output_layer(x)

#////////// Training Setup /////////
def initialize_sequences(batch_size, seq_length):
    cls_tokens = tf.fill((batch_size, 1), CLS_TOKEN)
    random_tokens = tf.random.uniform(
        (batch_size, seq_length - 1),
        0, VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX,
        dtype=tf.int32
    )
    return tf.concat([cls_tokens, random_tokens], axis=1)

# Initialize model
seq_length = X_train.shape[1]
model = AutoregressiveTransformerModel(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_length=seq_length,
    vocab_size=VOCAB_SIZE
)

# Loss and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)

# Metrics
train_loss = keras.metrics.Mean(name="train_loss")
val_loss = keras.metrics.Mean(name="val_loss")

# Data pipelines
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_dataset = test_dataset.batch(BATCH_SIZE).repeat()

#///////////////////////////////////////////////////////////////////////////////////////
#                                            TRAINING LOOP
#///////////////////////////////////////////////////////////////////////////////////////
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        logits = model(x[:, :-1], training=True)
        loss = loss_fn(x[:, 1:], logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

@tf.function
def val_step(x):
    logits = model(x[:, :-1], training=False)
    loss = loss_fn(x[:, 1:], logits)
    val_loss(loss)

# Early stopping
patience = 50
best_val_loss = float('inf')
wait = 0
best_weights = None

loss_history = []
val_loss_history = []

print("Using device:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

for i, batch_x in enumerate(train_dataset):
    if i >= 30000 or wait >= patience:
        break

    train_step(batch_x)
    
    batch_val = next(iter(test_dataset))
    val_step(batch_val)
    
    if val_loss.result() < best_val_loss:
        best_val_loss = val_loss.result()
        best_weights = model.get_weights()
        wait = 0
    else:
        wait += 1

    if i % 100 == 0:
        print(
            f"Step {i}: "
            f"Train Loss = {train_loss.result():.4f}, "
            f"Val Loss = {val_loss.result():.4f}, "
            f"Best Val Loss = {best_val_loss:.4f}, "
            f"Wait = {wait}/{patience}"
        )
        
    loss_history.append(train_loss.result())
    val_loss_history.append(val_loss.result())
    train_loss.reset_states()
    val_loss.reset_states()

if best_weights is not None:
    model.set_weights(best_weights)

#////////// Save Results /////////
# Plotting (same as before)
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Steps (x100)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/training_curves_tf.png')

# Save model
model.save_weights(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/model_weights.h5')

print("Training complete!")
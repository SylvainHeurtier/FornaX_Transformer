import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle



# Chargement des constantes
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import BATCH_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, name_dir
from Constantes import name_dir



#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                       Generation des sequences
#///////////////////////////////////////////////////////////////////////////////////////////////////////


titre = "Generation des sequences"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)

# Chargement des données
X_train = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.txt', dtype=np.int32)
X_test = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.txt', dtype=np.int32)

print(f"\nDim de X_train: {X_train.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)
print(f"Dim de X_test: {X_test.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)


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

#///////////////////////////////////////////////////////////////////////////////////////////////////////


# Architecture du Transformer
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
        x = x + attn_output
        x = self.layernorm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return self.layernorm2(x)

class AutoregressiveTransformerModel(keras.Model):
    def __init__(self, d_model, num_heads, num_layers, seq_length, vocab_size):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.token_embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = layers.Embedding(seq_length, d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(vocab_size)

    def call(self, x):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        
        # Masque causal
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        
        # Embeddings
        token_embeddings = self.token_embed(x)
        positions = tf.range(seq_len, dtype=tf.int32)
        pos_embeddings = self.pos_embed(positions)
        x = token_embeddings + pos_embeddings

        # Blocs Transformer
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return self.output_layer(x)


#///////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialisation et chargement du modèle
seq_length = X_train.shape[1]
model = AutoregressiveTransformerModel(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_length=seq_length,
    vocab_size=VOCAB_SIZE
)

# Chargement des poids sauvegardés
# Création des variables (NÉCESSAIRE avant load_weights)
dummy_input = tf.ones((1, X_train.shape[1]), dtype=tf.int32)
_ = model(dummy_input)  # Cette ligne crée les variables
# Chargement SIMPLE des poids .h5
model.load_weights(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/model_weights.h5')

# Paramètres de génération
#max_length = len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4
max_length = len(X_test[0])

#///////////////////////////////////////////////////////////////////////////////////////////////////////

# Fonction de génération (version TensorFlow pure)
def generate_sequences_batch(model, initial_tokens_batch, max_length):
    batch_size = initial_tokens_batch.shape[0]
    current_seq_len = initial_tokens_batch.shape[1]
    
    output = np.full((batch_size, max_length), PAD_TOKEN, dtype=np.int32)
    output[:, :current_seq_len] = initial_tokens_batch
    
    for i in range(current_seq_len, max_length):
        logits = model(output[:, :i])
        next_tokens = tf.argmax(logits[:, -1, :], axis=-1).numpy()
        output[:, i] = next_tokens
        
        # Arrêt si tous les séquences ont généré SEP_TOKEN
        if np.all(next_tokens == SEP_TOKEN):
            break
    
    return output

# Fonction principale adaptée
def Pythie_optimized(X, min_idx_gen, max_idx_gen, save_list_of_generated_sequences, name, batch_size=128):
    if not save_list_of_generated_sequences:
        return

    index_end_of_Xamin_part = MAX_SOURCES * len(SELECTED_COLUMNS_Xamin) + 1
    max_idx_gen = min(max_idx_gen, len(X) - 1)
    
    num_batches = int(np.ceil((max_idx_gen - min_idx_gen + 1) / batch_size))
    list_of_generated_sequences = []

    for batch_num in range(num_batches):
        start_idx = min_idx_gen + batch_num * batch_size
        end_idx = min(start_idx + batch_size, max_idx_gen + 1)
        
        batch = X[start_idx:end_idx, :index_end_of_Xamin_part]
        generated = generate_sequences_batch(model, batch, max_length)
        list_of_generated_sequences.extend(generated)

    # Sauvegarde des résultats
    save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/"
    suffix = "full" if max_idx_gen == len(X) - 1 else f"{min_idx_gen}-{max_idx_gen}"
    with open(f"{save_path}generated_seq_by_imperator_{name}_{suffix}.pkl", 'wb') as f:
        pickle.dump(list_of_generated_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

# Exécution
Pythie_optimized(X_test, 0, 500000, True, "test", batch_size=128)
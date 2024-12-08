# model.py
import tensorflow as tf
from spektral.layers import GATConv, GlobalAvgPool

def create_model(input_dim, num_heads=4, hidden_dim=128):
    X_in = tf.keras.Input(shape=(None, input_dim))   # Node features
    A_in = tf.keras.Input(shape=(None, None))        # Adjacency matrix

    # GAT 層
    x = GATConv(channels=hidden_dim, attn_heads=num_heads, concat=True)([X_in, A_in])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.elu(x)

    # 第二層 GAT
    x = GATConv(channels=64, attn_heads=num_heads, concat=False)([x, A_in])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.elu(x)

    x = GlobalAvgPool()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='elu')(x)
    outputs = tf.keras.layers.Dense(1)(x)  # output logits

    model = tf.keras.Model(inputs=[X_in, A_in], outputs=outputs)
    return model

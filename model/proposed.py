import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, Dropout, Add, LayerNormalization, Input, Flatten, Lambda
from tensorflow.keras.models import Model

# ==========================================================
# 1. Main Model Construction
# ==========================================================

def build_srars_model(config):
    """
    SRARS (Summarized Review-Aware Recommender System)
    Assembles the main architecture using parameters from config.yaml.
    """
    # Load parameters from config
    num_heads = config['model']['num_heads']
    key_dim = config['model']['key_dim']
    dff = config['model']['dff']
    dropout_rate = config['model']['dropout_rate']
    epsilon = float(config['model'].get('epsilon', 1e-6))

    # [Input Layer] 768-dim BART embeddings for User and Item
    user_review_input = Input(shape=(768,), dtype=tf.float32, name='user_review_input')
    item_review_input = Input(shape=(768,), dtype=tf.float32, name='item_review_input')

    # [Feature Extraction] Dimensionality Reduction using MLP
    user_reduced = LayerNormalization(epsilon=epsilon)(user_review_input)
    user_reduced = Dense(512, activation='relu')(user_reduced)
    user_reduced = Dense(key_dim, activation='relu')(user_reduced) # Adjusted to key_dim

    item_reduced = LayerNormalization(epsilon=epsilon)(item_review_input)
    item_reduced = Dense(512, activation='relu')(item_reduced)
    item_reduced = Dense(key_dim, activation='relu')(item_reduced) # Adjusted to key_dim

    # [Interaction Learning] Outer Product (User ⊗ Item)
    user_expanded = ExpandDimsLayer()(user_reduced)
    item_expanded = ExpandDimsLayer()(item_reduced)
    user_transposed = TransposeLayer()(user_expanded)
    
    # Interaction Map Generation (Resulting shape: [batch, key_dim, key_dim])
    interaction_map = MatMulLayer()([item_expanded, user_transposed])

    # [Attention Module] Self-Attention for Global Dependency Modeling
    attn_block = SelfAttentionBlock(num_heads, key_dim, dff, dropout_rate, epsilon)
    interaction_refined = attn_block(interaction_map)

    # [Rating Prediction] Final MLP Layers
    flattened = Flatten()(interaction_refined)
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    
    # Final Rating Output (Regression)
    output = Dense(1, activation='linear', name='rating_output')(dense3)

    return Model(inputs=[user_review_input, item_review_input], outputs=output)

# ==========================================================
# 2. Custom Layer Definitions (Sub-modules)
# ==========================================================

class SelfAttentionBlock(Layer):
    """Refines the Interaction Map using Multi-Head Self-Attention."""
    def __init__(self, num_heads, key_dim, dff, dropout_rate=0.1, epsilon=1e-6, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.layer_norm = LayerNormalization(epsilon=epsilon)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(key_dim)
        ])
        self.reshape_layer = Lambda(lambda x: tf.reshape(x, [-1, tf.shape(x)[1], key_dim]))

    def call(self, inputs, training=False):
        # Attention + Residual Connection
        inputs_reshaped = self.reshape_layer(inputs)
        attn_out = self.multi_head_attention(inputs_reshaped, inputs_reshaped, inputs_reshaped)
        attn_out = self.dropout(attn_out, training=training)
        res_1 = self.add([attn_out, inputs_reshaped])
        norm_1 = self.layer_norm(res_1)

        # FFN + Residual Connection
        ffn_out = self.ffn(norm_1)
        ffn_out = self.dropout(ffn_out, training=training)
        res_2 = self.add([ffn_out, norm_1])
        return self.layer_norm(res_2)

class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, 2)

class TransposeLayer(Layer):
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 1])

class MatMulLayer(Layer):
    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])

# ==========================================================
# 3. Local Test (Main)
# ==========================================================

if __name__ == "__main__":
    # Mock config for testing
    test_config = {
        'model': {
            'num_heads': 8,
            'key_dim': 128,
            'dff': 2048,
            'dropout_rate': 0.1
        }
    }
    srars_model = build_srars_model(test_config)
    srars_model.summary()
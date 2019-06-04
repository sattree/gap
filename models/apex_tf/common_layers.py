import tensorflow as tf

def attn_pool_layer(X, seq_lens, n_hidden_x, seed):
    dense_init = tf.keras.initializers.glorot_normal(seed=seed)
    
    a_s = tf.keras.layers.Dense(n_hidden_x, activation='tanh', kernel_initializer=dense_init)(X)
    a_s = tf.keras.layers.Dense(1, activation=None, kernel_initializer=dense_init)(a_s)
    
    attn_masker = MaskForAttn()
    a_s = attn_masker.mask_pads(a_s, seq_lens)
    
    a_s = tf.nn.softmax(a_s, axis=1)
    a_s = attn_masker.restore(a_s, seq_lens)
    
    G_s = tf.matmul(a_s, X, transpose_a=True, name='G_s_matmul')
    
    G_s = tf.reshape(G_s, [-1, n_hidden_x])
    
    return G_s

def coattn_layer(X, Q, X_lens, Q_lens, n_hidden_x, seed, name=''):
    dense_init = tf.keras.initializers.glorot_normal(seed=seed)
    
    A = tf.matmul(X, Q, transpose_b=True)
    
    attn_masker = MaskForAttn()

    A = attn_masker.mask_pads(A, X_lens, Q_lens)

    S_s = tf.matmul(tf.nn.softmax(A, axis=2), Q)
    S_s = attn_masker.restore(S_s, X_lens)
    
    
    S_q = tf.matmul(tf.nn.softmax(A, axis=1), X, transpose_a=True)
    S_q = attn_masker.restore(S_q, Q_lens)

    C_s = tf.matmul(tf.nn.softmax(A, axis=2), S_q)
    C_s = attn_masker.restore(C_s, X_lens)

    gru_init = tf.keras.initializers.glorot_uniform(seed=7)
    gru_init2 = tf.keras.initializers.orthogonal(seed=7)
    C_s = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(n_hidden_x, 
                                                                kernel_initializer=gru_init,
                                                                recurrent_initializer=gru_init2,
                                                                return_sequences=True))(C_s)

#     fw_cell = tf.keras.layers.GRUCell(n_hidden_x, 
# kernel_initializer=gru_init,
#                                                                 recurrent_initializer=gru_init2,
#                                         name=name)
#     bw_cell = tf.keras.layers.GRUCell(n_hidden_x, 
# kernel_initializer=gru_init,
#                                                                 recurrent_initializer=gru_init2,
#         name=name)
#     C_s, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
#                                                           C_s, sequence_length=X_lens, dtype=tf.float32)
#     C_s = tf.concat(C_s, axis=-1)

    U_s = tf.concat([C_s, S_s], axis=-1, name='document_context')
    
    U_s = tf.keras.layers.Dense(2*n_hidden_x, activation='tanh', kernel_initializer=dense_init)(U_s)
    
    return U_s

class MaskForAttn():
    def __init__(self):
        pass

    def mask_pads(self, X, X_lens, Q_lens=None):
        if Q_lens is None:
            masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
            masks_x = tf.expand_dims(masks_x, axis=-1)
            
            paddings = tf.ones_like(X)*(-2**32+1)
            X = tf.where(tf.equal(masks_x, 0), paddings, X)
            return X

        masks_x = tf.cast(tf.sequence_mask(X_lens), tf.float32)
        masks_q = tf.cast(tf.sequence_mask(Q_lens), tf.float32)
        masks_x = tf.expand_dims(masks_x, axis=-1)
        masks_q = tf.expand_dims(masks_q, axis=-1)
        masks = tf.matmul(masks_x, masks_q, transpose_b=True)
        
        paddings = tf.ones_like(X)*(-2**32+1)

        X = tf.where(tf.equal(masks, 0), paddings, X)

        return X

    def restore(self, X, seq_lens):
        masks = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        X = X * masks

        return X
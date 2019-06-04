import tensorflow as tf
import tensorflow_hub as hub
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class ELMoFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    
    def transform(self, X):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        vectors = []
        batch_size = 50
        for idx in tqdm(range(0, len(X), batch_size)):
            batch = X.loc[idx:idx+batch_size-1]
            tokens = batch.tokens.values

            seq_lens = [len(x) for x in tokens]
            max_len = max(seq_lens)
            tokens = [s + ['']*(max_len-len(s)) for s in tokens]

            embeddings = self.elmo(inputs={
                                    "tokens": np.array(tokens),
                                    "sequence_len": seq_lens
                                    },
                                     signature="tokens",
                                     as_dict=True)["elmo"]
            embeddings = sess.run(embeddings)

            for idx, (_, row) in enumerate(batch.iterrows()):
                emb = embeddings[idx][:seq_lens[idx]]
                vectors.append(emb)

        return vectors
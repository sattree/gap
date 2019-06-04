import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

class DataLoader():
    def __init__(self, X, batch_size=10, shuffle=False, seed=None):
        self.X = X.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.N = len(self.X)
        self.start = 0
        self.end = 0
        self.seed = seed

        max_len = max(self.X[8])
        self.X[0] = self.X[0].apply(lambda x: np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1])))))
        max_len = max(self.X[9])
        self.X[1] = self.X[1].apply(lambda x: np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1])))))
        max_len = max(self.X[10])
        self.X[2] = self.X[2].apply(lambda x: np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1])))))
        max_len = max(self.X[11])
        self.X[3] = self.X[3].apply(lambda x: np.vstack((x, np.zeros((max_len-x.shape[0], x.shape[1])))))
        
        if self.shuffle:
            self.X = self.X.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                    
    def __iter__(self):
        return self
    
    def __next__(self):
            
        if self.start >= self.N:
            self.start = 0
            if self.shuffle:
                self.X = self.X.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            raise StopIteration
        
        batch = self.X.loc[self.start:self.start+self.batch_size-1].T.values.tolist()
        
        batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, \
                          seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b = batch
        batch_x = self.truncate(batch_x, seq_lens_x)
        batch_x_p = self.truncate(batch_x_p, seq_lens_p)
        batch_x_a = self.truncate(batch_x_a, seq_lens_a)
        batch_x_b = self.truncate(batch_x_b, seq_lens_b)
        
        self.start += self.batch_size
        return (batch_x, batch_x_p, batch_x_a, batch_x_b, batch_c, batch_p, batch_q, batch_y, \
                          seq_lens_x, seq_lens_p, seq_lens_a, seq_lens_b)
    
    def truncate(self, batch_x, seq_lens_x):
        max_len = max(seq_lens_x)
        x = np.array([x[:max_len, :] for x in batch_x])
        return x
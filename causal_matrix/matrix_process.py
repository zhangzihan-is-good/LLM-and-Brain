import numpy as np
import pandas as pd


B = np.load('infer_causal_matrix.npy')

new = np.zeros(B.shape)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if i == j:
            continue

        if np.isnan(B[i][j]) or B[i][j] == 0:
            continue

        if i < j:
            new[j, i] = 1
        else:
            new[i, j] = 1

new = new[:-1]
new = new[:,:-1]

new = new.astype(int)
print(new.shape)

np.save("infer_causal_matrix.npy", new)

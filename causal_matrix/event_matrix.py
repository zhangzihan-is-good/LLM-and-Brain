import numpy as np

causal = np.load("cause_event_indice.npy")
result = np.load("result_event_indice.npy")
ind = np.load("ind_event_indice.npy")


all_set = sorted(
                causal.tolist() 
                + 
                result.tolist()
                +
                ind.tolist()
                )

print(all_set)

new_matrix = np.zeros((34, 34))
for i in range(34):
    for j in range(34):
        if i >= j and i in all_set:
            new_matrix[i, j] = 1
            new_matrix[j, i] = 1
        elif i <= j and j in all_set:
            new_matrix[i, j] = 1
            new_matrix[j, i] = 1
np.fill_diagonal(new_matrix, 0)

np.save("agan_yg_matrix.npy", new_matrix)


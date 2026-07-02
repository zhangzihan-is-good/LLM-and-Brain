import os
import numpy as np
from tqdm import tqdm


def find_clean_chains(M):
    N = M.shape[0]
    chains = []

    def no_extra_edges(A, B, C, D):
        nodes = {A, B, C, D}
        wrong = []
        for i in nodes:
            for j in range(N):
                if j not in nodes:
                    # i -> j 或 j -> i 都不允许
                    if M[i, j] == 1 or M[j, i] == 1:
                        wrong.append(j)
        unique = []
        [unique.append(x) for x in wrong if x not in unique]
        return unique

    for A in tqdm(range(N)):
        for B in range(N):
            for C in range(N):
                for D in range(N):
                    if A==B or A==C or A==D or B==C or B==D or C==D:
                        continue
                    if len({A, B, C, D}) < 4: 
                        continue
                    # Case 1: A -> B -> C -> D
                    type_list = []
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D noc"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D noc del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D AC"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D AC del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 1 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D AD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 1 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D AD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 0 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D BD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 0 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D BD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 1 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D ACAD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 1 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D ACAD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 0 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D ACBD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 0 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D ACBD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 0 and M[D, A] == 1 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D ADBD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 1 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D ADBD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 1 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C->D ACADBD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[C, A] == 1 and M[D, A] == 1 and M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C->D ACADBD del E"))        
                    # Case 2: A <- B -> C -> D
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D noc"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D noc del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D AC"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D AC del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D AD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D AD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D BD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D BD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D ACAD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 0 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D ACAD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D ACBD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D ACBD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C->D ADBD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, C] == 1 and (M[D, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1)) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C->D ADBD del E"))
                    # Case 3: A <- B -> C <- D
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D noc"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D noc del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D AC"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D AC del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D AD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D AD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D BD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D BD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D ACAD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 0 and M[D, B] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D ACAD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D ADBD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D ADBD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D ACBD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D ACBD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C<-D ACADBD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and (M[B, D] == 1 or M[D, B] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C<-D ACADBD del E"))
                    # Case 4: A -> B <- C <- D
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D noc"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D noc del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D AC"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D AC del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D AD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D AD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D BD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D BD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D ACAD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 0 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D ACAD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D ACBD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 0 and M[D, A] == 0) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D ACBD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D ADBD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 0 and M[C, A] == 0) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D ADBD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C<-D ACADBD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[C, D] == 1 and (M[A, C] == 1 or M[C, A] == 1) and (M[A, D] == 1 or M[D, A] == 1) and M[B, D] == 1 and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C<-D ACADBD del E"))
                    # Case 5: A->B->C B->D
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 0 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D noc"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 0 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D noc del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 0 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D AC"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 0 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D AC del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 1 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D AD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 1 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D AD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 0 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D CD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 0 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D CD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 1 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D ACAD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 1 and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D ACAD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 0 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D ACCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 0 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D ACCD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 1 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D ADCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 0 and M[D, A] == 1 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D ADCD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 1 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C B->D ACADCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[D, B] == 1 and M[C, A] == 1 and M[D, A] == 1 and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C B->D ACADCD del E"))
                    # Case 6: A->B->C D->B
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 0 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B noc"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 0 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B noc del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 0 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B AC"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 0 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B AC del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 0 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B AD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 0 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B AD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 1 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B CD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 1 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B CD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 0 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B ACAD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 0 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B ACAD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 1 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B ADCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 0 and M[C, D] == 1 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B ADCD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 1 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B ACCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 1 and (M[D, A] == 0 and M[A, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B ACCD del E"))
                    if M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 1 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B->C D->B ACADCD"))
                    elif M[B, A] == 1 and M[C, B] == 1 and M[B, D] == 1 and M[C, A] == 1 and M[C, D] == 1 and (M[D, A] == 1 or M[A, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B->C D->B ACADCD del E"))
                    # Case 7: A<-B->C B->D
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D noc"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D noc del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D AC"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D AC del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D AD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D AD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D CD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D CD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D ACAD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D ACAD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D ADCD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D ADCD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D ACCD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D ACCD del E"))
                    if M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A<-B->C B->D ACADCD"))
                    elif M[A, B] == 1 and M[C, B] == 1 and M[D, B] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A<-B->C B->D ACADCD del E"))
                    # Case 8: A->B<-C B<-D
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D noc"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D noc del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D AC"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D AC del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D AD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D AD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D CD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D CD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D ACAD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 0 and M[C, D] == 0) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D ACAD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D ADCD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 0 and M[A, C] == 0) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D ADCD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D ACCD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 0 and M[A, D] == 0) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D ACCD del E"))
                    if M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) == 0:
                        chains.append((A, B, C, D, "A->B<-C B<-D ACADCD"))
                    elif M[B, A] == 1 and M[B, C] == 1 and M[B, D] == 1 and (M[C, A] == 1 or M[A, C] == 1) and (M[D, A] == 1 or M[A, D] == 1) and (M[D, C] == 1 or M[C, D] == 1) and len(no_extra_edges(A, B, C, D)) != 0:
                        chains.append((A, B, C, D, no_extra_edges(A, B, C, D), "A->B<-C B<-D ACADCD del E"))
        
    # types = ["A->B->C", "A->B->C del D", "A<-B->C", "A<-B->C del D", "A->B<-C", "A->B<-C del D"]

    def canonical_chain(chain, tp):
        a, b, c, d = chain[0], chain[1], chain[2], chain[3]
        nodes = [a, b, c, d]
        if "del" in tp:
            del_num = chain[-2]
            del_list = [x for x in del_num if min(nodes) < x < max(nodes)]
            if len(del_list) != 0:
                new_tp = tp
                return (a, b, c, d, del_list, new_tp)
            else:
                new_tp = tp.split(" del ")[0]
                return (a, b, c, d, new_tp)
        else:
            return (a, b, c, d, tp)

    chain1 = []

    for ch in chains:
        new1 = canonical_chain(ch, ch[-1])
        chain1.append(new1)
    unique1 = []
    [unique1.append(x) for x in chain1 if x not in unique1]

    return unique1

def load_braindata(id, type=None):

    brain_story_path = f"story_{id}.npy"
    brain_eb_path = f"eb_{id}.npy"

    brain_story = np.load(brain_story_path)
    brain_eb = np.load(brain_eb_path)

    mask_B_row = ~np.isnan(brain_story).all(axis=1)
    mask_E_row = ~np.isnan(brain_eb).all(axis=1)

    mask_row = mask_B_row & mask_E_row

    brain_story_mask_row = brain_story[mask_row]
    brain_eb_mask_row = brain_eb[mask_row]

    mask_B_col = ~np.isnan(brain_story_mask_row).any(axis=0)
    mask_E_col = ~np.isnan(brain_eb_mask_row).any(axis=0)
    mask_col = mask_B_col & mask_E_col

    brain_story_mask = brain_story_mask_row[:, mask_col]
    brain_eb_mask = brain_eb_mask_row[:, mask_col]

    return brain_story_mask, brain_eb_mask, mask_row

def load_matrix_dict(mask):

    C_noshort = np.load("/home/ytchen/MyExperiment/brain_reasoning/casual_matrix/C_noshrot.npy")
    C_noshort = C_noshort[mask][:, mask]

    matrix_dict = {"C_noshort": C_noshort}

    return matrix_dict

for id in tqdm(range(1,18)):
    _, _, brain_mask = load_braindata(id)
    matrix_dict = load_matrix_dict(brain_mask)

    for name, matrix in tqdm(matrix_dict.items()):
        a = find_clean_chains(matrix)
        allchain = []
        save_num = 4
        for chain in a:
            nodes = [chain[0], chain[1], chain[2], chain[3]]
            if "del" in chain[-1]:
                all_interval = max(nodes) - min(nodes) + 1
                del_interval = len(chain[-2])
                real_interval = all_interval - del_interval
            else:
                real_interval = max(nodes) - min(nodes) + 1
            if real_interval > save_num:
                allchain.append(chain)

        all_result = []
        for c in allchain:
            nodes = [c[0], c[1], c[2], c[3]]
            low = min(nodes)
            high = max(nodes)
            if "del" in c[-1]:
                tp = c[-1].split(" del ")[0]
                result = [x for x in range(low, high+1) if x not in c[-2]]
            else:
                tp = c[-1]
                result = [x for x in range(low, high+1)]
            all_result.append((result, nodes, tp))

        result_dict = {}

        for r in all_result:
            tpp = r[-1]
            result_dict.setdefault(tpp, []).append((r[0], r[1]))

        for a,b in result_dict.items():
            print(a, len(b))

        os.makedirs(f"casual-structure/dict4/subj{id}", exist_ok=True)
        np.save(f"casual-structure/dict4/subj{id}/{name}_{save_num}.npy", result_dict)
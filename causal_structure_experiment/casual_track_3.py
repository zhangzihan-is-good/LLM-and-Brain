import os
import numpy as np
from tqdm import tqdm

def find_clean_chains(M):
    N = M.shape[0]
    chains = []

    def no_extra_edges(A, B, C):
        nodes = {A, B, C}
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

    for A in range(N):
        for B in range(N):
            if A == B: continue
            for C in range(N):
                if len({A, B, C}) < 3: continue

                # Case 1: A -> B -> C no AC
                type_list = []
                if M[B, A] == 1 and M[C, B] == 1 and M[C, A] == 0 and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A->B->C noc"))
                elif M[B, A] == 1 and M[C, B] == 1 and M[C, A] == 0 and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A->B->C noc del D"))
                if M[B, A] == 1 and M[C, B] == 1 and M[C, A] == 1 and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A->B->C AC"))
                elif M[B, A] == 1 and M[C, B] == 1 and M[C, A] == 1 and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A->B->C AC del D"))
                # Case 2: A <- B -> C
                if M[A, B] == 1 and M[C, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A<-B->C noc"))
                elif M[A, B] == 1 and M[C, B] == 1 and (M[A, C] == 0 and M[C, A] == 0) and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A<-B->C noc del D"))
                if M[A, B] == 1 and M[C, B] == 1 and (M[A, C] == 1 or M[C, A] == 1) and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A<-B->C AC"))
                elif M[A, B] == 1 and M[C, B] == 1 and (M[A, C] == 1 or M[C, A] == 1) and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A<-B->C AC del D"))
                # Case 3: A -> B <- C
                if M[B, A] == 1 and M[B, C] == 1 and (M[A, C] == 0 and M[C, A] == 0) and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A->B<-C noc"))
                elif M[B, A] == 1 and M[B, C] == 1 and (M[A, C] == 0 and M[C, A] == 0) and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A->B<-C noc del D"))
                if M[B, A] == 1 and M[B, C] == 1 and (M[A, C] == 1 or M[C, A] == 1) and len(no_extra_edges(A, B, C)) == 0:
                    chains.append((A, B, C, "A->B<-C AC"))
                elif M[B, A] == 1 and M[B, C] == 1 and (M[A, C] == 1 or M[C, A] == 1) and len(no_extra_edges(A, B, C)) != 0:
                    chains.append((A, B, C, no_extra_edges(A, B, C), "A->B<-C AC del D"))
       
    # types = ["A->B->C", "A->B->C del D", "A<-B->C", "A<-B->C del D", "A->B<-C", "A->B<-C del D"]

    def canonical_chain(chain, tp):
        a, b, c = chain[0], chain[1], chain[2]
        nodes = [a, b, c]
        if "del" in tp:
            del_num = chain[-2]
            del_list = [x for x in del_num if min(nodes) < x < max(nodes)]
            if len(del_list) != 0:
                new_tp = tp
                if b == min(nodes) or b==max(nodes):
                # 两端排序，保持中间不变
                    left, right = sorted([a, c])
                    return (left, b, right, del_list, new_tp)
                else:
                    # 不对称情况，保持原样
                    return (a, b, c, del_list, new_tp)
            else:
                new_tp = tp.split(" del ")[0]
            # 中间节点是最小值 => 反向链等价
                if b == min(nodes) or b==max(nodes):
                    # 两端排序，保持中间不变
                    left, right = sorted([a, c])
                    return (left, b, right, new_tp)
                else:
                    # 不对称情况，保持原样
                    return (a, b, c, new_tp)
        else:
            if b == min(nodes) or b==max(nodes):
                # 两端排序，保持中间不变
                left, right = sorted([a, c])
                return (left, b, right, tp)
            else:
                # 不对称情况，保持原样
                return (a, b, c, tp)

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


for id in tqdm(range(1, 18)):
    _, _, brain_mask = load_braindata(id)

    C_noshort = np.load("/home/ytchen/MyExperiment/brain_reasoning/casual_matrix/C_noshrot.npy")
    C_noshort = C_noshort[brain_mask][:, brain_mask]

    matrix_dict = {"C_noshort": C_noshort}

    for name, matrix in tqdm(matrix_dict.items()):
        a = find_clean_chains(matrix)
        allchain = []
        save_num = 3
        for chain in a:
            nodes = [chain[0], chain[1], chain[2]]
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
            nodes = [c[0], c[1], c[2]]
            low = min(nodes)
            high = max(nodes)
            if "del" in c[-1]:
                tp = c[-1].split(" del ")[0]
                result = [x for x in range(low, high+1) if x not in c[-2]]
            else:
                tp = c[-1]
                result = [x for x in range(low, high+1)]
            all_result.append((result, nodes, tp))

        result_dict = {"A->B->C AC":[], "A<-B->C AC":[], "A->B<-C AC":[], "A->B->C noc":[], "A<-B->C noc":[], "A->B<-C noc":[]}

        for r in all_result:
            tpp = r[-1]
            result_dict[tpp].append((r[0], r[1]))

        for a,b in result_dict.items():
            print(a, len(b))

        os.makedirs(f"casual-structure/dict3/subj{id}", exist_ok=True)
        np.save(f"casual-structure/dict3/subj{id}/{name}_{save_num}.npy", result_dict)

import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Path to the pre-trained model')
args = parser.parse_args()

model_name = args.model_name
layer_dict = {"llama3.1-8B": 32, "llama3.1-base-8B": 32, "qwen2.5-7B": 28, "qwen2.5-base-7B": 28, "qwen2.5math": 28, "qwen2.5coder":28}

C_noshort = np.load("C_noshort.npy")

matrix_dict = {"C_noshort":C_noshort}

def fit_lower_triangular_closed(B, E, C, mask):
    B = torch.as_tensor(B, dtype=torch.float64, device=DEVICE)
    E = torch.as_tensor(E, dtype=torch.float64, device=DEVICE)
    C = torch.as_tensor(C, dtype=torch.float64, device=DEVICE)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE)
    
    C = C[mask][:,mask]

    B = B / (torch.norm(B, dim=1, keepdim=True) + 1e-8)
    E = E / (torch.norm(E, dim=1, keepdim=True) + 1e-8)

    R = torch.zeros((B.shape[0], B.shape[0]), dtype=torch.float64, device=DEVICE)
    delta_R = torch.zeros((B.shape[0], B.shape[0]), dtype=torch.float64, device=DEVICE)
    
    for i in range(R.shape[0]):
        col = C[i].bool()
        B_sub = B[col, :]
        delta_i = E[i, :] - B[i, :]
        E_i = E[i, :]
        
        pinv = torch.linalg.pinv(B_sub @ B_sub.T)
        R_i_sub = E_i @ B_sub.T @ pinv
        delta_R_i_sub = delta_i @ B_sub.T @ pinv

        R[i, col] = R_i_sub
        delta_R[i, col] = delta_R_i_sub

    return R, delta_R

def evaluate_llm(llm_story, llm_eb, brain_story, brain_eb, C, mask):
    R = fit_lower_triangular_closed(brain_story, brain_eb, C, mask)
    llm_story = llm_story[mask]
    llm_eb = llm_eb[mask]
    llm_story = llm_story / (np.linalg.norm(llm_story, axis=1, keepdims=True) + 1e-8)
    llm_eb = llm_eb / (np.linalg.norm(llm_eb, axis=1, keepdims=True) + 1e-8)
    predicted = R @ llm_story
    mse = np.mean((predicted - llm_eb) ** 2)
    return mse

def load_braindata(id):
    
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

    return brain_story_mask, brain_eb_mask, mask_row, mask_col

def load_llmdata(layer, model_name, mask):

    llm_story_path = f"story/{layer}/hidden_states.npy"
    llm_eb_path = f"eb/{layer}/hidden_states.npy"

    llm_story = np.load(llm_story_path)
    llm_eb = np.load(llm_eb_path)

    llm_story_mask = llm_story[mask]
    llm_eb_mask = llm_eb[mask]

    return llm_story_mask, llm_eb_mask

id_list = range(1, 18)
for id in tqdm(id_list):
    for layer in range(24, 26):

        brain_story, brain_eb, brain_mask, mask_col = load_braindata(id)
        llm_story, llm_eb = load_llmdata(layer, args.model_name, brain_mask)
        
        for matrix_name, matrix in matrix_dict.items():

            R_brain, delta_brain = fit_lower_triangular_closed(brain_story, brain_eb, matrix, brain_mask)
            R_llm, delta_llm = fit_lower_triangular_closed(llm_story, llm_eb, matrix, brain_mask)

            os.makedirs(f"matrix_result/subj{id}/layer{layer}", exist_ok=True)
            np.save(f"matrix_result/subj{id}/layer{layer}/R_brain.npy", R_brain.cpu().numpy())
            np.save(f"matrix_result/subj{id}/layer{layer}/delta_brain.npy", delta_brain.cpu().numpy())
            np.save(f"matrix_result/subj{id}/layer{layer}/R_llm.npy", R_llm.cpu().numpy())
            np.save(f"matrix_result/subj{id}/layer{layer}/delta_llm.npy", delta_llm.cpu().numpy())



            all_mse = torch.mean((R_brain - R_llm) ** 2)
            delta_mse = torch.mean((delta_brain - delta_llm) ** 2)

            score_mask = np.zeros(brain_story.shape[1])
            delta_score_mask = np.zeros(brain_story.shape[1])

            all_data = np.load(f"story_{id}.npy")

            score_all = np.full(all_data.shape[1], np.nan)
            delta_score_all = np.full(all_data.shape[1], np.nan)

            for j in tqdm(range(brain_story.shape[1])):
                brain_eb_j = np.delete(brain_eb, j, axis=1)
                brain_story_j = np.delete(brain_story, j, axis=1)

                R_brain_j, delta_brain_j = fit_lower_triangular_closed(brain_story_j, brain_eb_j, matrix, brain_mask)
                mse_j = torch.mean((R_brain_j - R_llm) ** 2)
                delta_mse_j = torch.mean((delta_brain_j - delta_llm) ** 2)

                score_mask[j] = mse_j.cpu().numpy()
                delta_score_mask[j] = delta_mse_j.cpu().numpy()

            percent = 20
            top_k = int((percent/100) * brain_story.shape[1])

            score_all[mask_col] = score_mask
            delta_score_all[mask_col] = delta_score_mask

            small10 = np.argsort(np.nan_to_num(score_all, nan=np.inf))[:top_k]
            big10 = np.argsort(-np.nan_to_num(score_all, nan=-np.inf))[:top_k]
            delta_small10 = np.argsort(np.nan_to_num(delta_score_all, nan=np.inf))[:top_k]
            delta_big10 = np.argsort(-np.nan_to_num(delta_score_all, nan=-np.inf))[:top_k]

            small_all = np.zeros(all_data.shape[1])
            big_all = np.zeros(all_data.shape[1])
            delta_small_all = np.zeros(all_data.shape[1])
            delta_big_all = np.zeros(all_data.shape[1])

            small_all[small10] = 1
            big_all[big10] = 1
            delta_small_all[delta_small10] = 1
            delta_big_all[delta_big10] = 1

            mse_result_dict = {"all_mse":all_mse.cpu().numpy(),
                            "delta_mse":delta_mse.cpu().numpy(), 
                            }

            os.makedirs(f"neuron/subj{id}/layer{layer}", exist_ok=True)
            np.save(f"neuron/subj{id}/layer{layer}/mse_dict.npy", mse_result_dict)
            np.save(f"neuron/subj{id}/layer{layer}/small{percent}.npy", small_all)
            np.save(f"neuron/subj{id}/layer{layer}/big{percent}.npy", big_all)
            np.save(f"neuron/subj{id}/layer{layer}/delta_small{percent}.npy", delta_small_all)
            np.save(f"neuron/subj{id}/layer{layer}/delta_big{percent}.npy", delta_big_all)
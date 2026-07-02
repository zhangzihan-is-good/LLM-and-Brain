import numpy as np
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

new_yg_matrix = np.load("new_yg_matrix.npy")
new_yin_matrix = np.load("new_yin_matrix.npy")
new_guo_matrix = np.load("new_guo_matrix.npy")
new_ind_matrix = np.load("new_ind_matrix.npy")
new_other_matrix = np.load("new_other_matrix.npy")

matrix_dict = {"new_yin":new_yin_matrix,
                "new_guo":new_guo_matrix,
                "new_ind":new_ind_matrix,
                "new_other":new_other_matrix,
                "new_yg":new_yg_matrix, 
                }

def react_matrix(B, E, mask=None):

    B = torch.tensor(B, dtype=torch.float64, device=DEVICE)
    E = torch.tensor(E, dtype=torch.float64, device=DEVICE)
    if mask is not None:
        B = B[:, mask == 1]
        E = E[:, mask == 1]

    B_norm = (B - B.mean(axis=1, keepdims=True)) / B.std(axis=1, keepdims=True)
    E_norm = (E - E.mean(axis=1, keepdims=True)) / E.std(axis=1, keepdims=True)
    
    R = (B_norm @ E_norm.t()) / B_norm.shape[1]
    R = torch.clamp(R, -0.999999, 0.999999)

    def fisher_transform(M):
        return 0.5 * torch.log((1 + M) / (1 - M))

    return fisher_transform(R)

def react_mean(story, eb, mmask=None):
    
    matrix = react_matrix(story, eb)
    num = 1
    if mmask is not None:
        mmask = torch.tensor(mmask, dtype=torch.float64, device=DEVICE)
        yg_matrix = matrix * mmask 

        upper = torch.triu(yg_matrix, diagonal=num)
        upper_mask = torch.triu(mmask, diagonal=num)
        upper_mean = (yg_matrix * upper_mask).sum() / upper_mask.sum()

        lower = torch.tril(yg_matrix, diagonal=-num)
        lower_mask = torch.tril(mmask, diagonal=-num)
        lower_mean = (yg_matrix * lower_mask).sum() / lower_mask.sum()

        all_mean = upper_mean - lower_mean

    else: 
        upper = torch.triu(matrix, diagonal=num)
        lower = torch.tril(matrix, diagonal=-num)

        upper_mean = upper.mean()
        lower_mean = lower.mean()

        all_mean = upper_mean - lower_mean

    return matrix, all_mean

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


for matrix_name, matrix in matrix_dict.items():
    result = {}

    for id in tqdm(range(1, 18)):
        result[f'subj{id}'] = {}
        brain_story, brain_eb, brain_mask, mask_col = load_braindata(id)

        matrix_mask = matrix[brain_mask][:, brain_mask]
        brain_matrix, brain_react = react_mean(brain_story, brain_eb, matrix_mask)

        for layer in tqdm(range(0,32)):
            llm_story, llm_eb = load_llmdata(layer, args.model_name, brain_mask)

            llm_matrix, llm_react = react_mean(llm_story, llm_eb, matrix_mask)

            result[f'subj{id}'][f"layer{layer}"] = {"brain_matrix":brain_matrix.cpu().numpy(), 
                                                    "llm_matrix":llm_matrix.cpu().numpy(), 
                                                    "brain_react":brain_react.cpu().numpy(), 
                                                    "llm_react":llm_react.cpu().numpy(),
                                                    "matrix_mask":matrix_mask
                                                    }
    
    os.makedirs(f"result/{args.model_name}", exist_ok=True) 
    np.save(f"result/{args.model_name}/{matrix_name}_react.npy", result)

print("已完成")

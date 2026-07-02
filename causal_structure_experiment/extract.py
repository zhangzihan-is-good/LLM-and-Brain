import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import gudhi as gd
from utils import load_braindata, load_llmdata, load_struct, process_struct_noeb, process_struct, process_struct_nocasual, rips_compute
from tqdm import tqdm
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

STRUCT_LEN=4
print(f"此时因果链长度: {STRUCT_LEN}")

if STRUCT_LEN==3:   
    NEIBOUR=50*2
elif STRUCT_LEN==4:
    NEIBOUR=150*2

def pre_process(sample_trajectories, all_trajectories=None):
    
    sample_lengths = [(name, len(t[0])) for name, traj in sample_trajectories.items() for t in traj]
    
    sample_points = [t[0] for _, traj in sample_trajectories.items() for t in traj]
    sample_stack = np.vstack(sample_points)

    scaler = StandardScaler()
    scaler.fit(sample_stack)
    sample_scaled = scaler.transform(sample_stack)

    pca = PCA(n_components=6, random_state=42)
    pca.fit(sample_scaled)
    sample_pca = pca.transform(sample_scaled)

    reducer = umap.UMAP(
        n_components=3,
        metric='cosine', 
        n_neighbors=NEIBOUR, 
        min_dist=0.05, 
        init='spectral',
        low_memory=True,
        densmap=True
    )
    reducer.fit(sample_pca)
    sample_umap = reducer.transform(sample_pca)

    def split(lengths, umap_vector):
        results = []
        start = 0
        for s in lengths:
            name, length = s
            traj_umap = umap_vector[start:start+length]
            
            rips = gd.AlphaComplex(points=traj_umap)
            st = rips.create_simplex_tree()
            diag = st.persistence()
            pi_vector = rips_compute(st)
            
            results.append({
                'type_name': name,
                'umap_coords': traj_umap,
                'persistence_figure': diag,
                "persistence_vector": pi_vector
            })
            start += length

        return results

    sample_result = split(sample_lengths, sample_umap)

    if all_trajectories is not None:
        all_lengths = [(name, len(t[0])) for name, traj in all_trajectories.items() for t in traj]
        all_points = [t[0] for _, traj in all_trajectories.items() for t in traj]
        all_stack = np.vstack(all_points)

        all_scaled = scaler.transform(all_stack)
        all_pca = pca.transform(all_scaled)
        all_umap = reducer.transform(all_pca)

        all_result = split(all_lengths, all_umap)

        return sample_result, all_result
    
    else:
        return sample_result


def traj_collect_noeb(story, struct_dict):

    all_traj = {}
    sample_traj = {}
    for key, structs in struct_dict.items():
        if "noc" in key:
            key_name = key.split(" noc")[0]
            traj = process_struct_noeb(story, structs, key_name, STRUCT_LEN)
            all_traj[key_name] = traj
            if len(traj) >= 20:
                sample_traj[key_name] = random.sample(traj, 20)
            else:
                sample_traj[key_name] = traj

    return sample_traj, all_traj

def traj_collect(story, eb, struct_dict):

    all_traj = {}
    sample_traj = {}
    for key, structs in struct_dict.items():
        if "noc" in key:
            key_name = key.split(" noc")[0]
            traj = process_struct(story, eb, structs, key_name, STRUCT_LEN)
            all_traj[key_name] = traj
            if len(traj) >= 10:
                sample_traj[key_name] = random.sample(traj, 10)
            else:
                sample_traj[key_name] = traj

    return sample_traj, all_traj

def yg_compare(story, eb, struct_dict):
    sample_yg_traj, _ = traj_collect(story, eb, struct_dict)
    yg_traj_list = []
    all_traj_dict = {}
    for _, traj in sample_yg_traj.items():
        yg_traj_list += traj

    all_traj_dict["yg"] = yg_traj_list

    noyg_traj_list = process_struct_nocasual(story, STRUCT_LEN)
    all_traj_dict["noyg"] = noyg_traj_list

    return all_traj_dict

def yg_compare_noeb(story, struct_dict):
    sample_yg_traj, _ = traj_collect_noeb(story, struct_dict)
    yg_traj_list = []
    all_traj_dict = {}
    for _, traj in sample_yg_traj.items():
        yg_traj_list += traj

    all_traj_dict["yg"] = yg_traj_list

    noyg_traj_list = process_struct_nocasual(story, STRUCT_LEN)
    all_traj_dict["noyg"] = noyg_traj_list

    return all_traj_dict

def yg_compare_all(brain_story, brain_eb, llm_story, llm_eb, struct_dict):
    brain_sample_yg_traj, _ = traj_collect(brain_story, brain_eb, struct_dict)
    llm_sample_yg_traj, _ = traj_collect(llm_story, llm_eb, struct_dict)

    brain_yg_traj_list = []
    llm_yg_traj_list = []
    all_traj_dict = {}

    for _, brain_traj in brain_sample_yg_traj.items():
        brain_yg_traj_list += brain_traj

    for _, llm_traj in llm_sample_yg_traj.items():
        llm_yg_traj_list += llm_traj

    all_traj_dict["brain_yg"] = brain_yg_traj_list
    all_traj_dict["llm_yg"] = llm_yg_traj_list

    brain_noyg_traj_list = process_struct_nocasual(brain_story, STRUCT_LEN)
    llm_noyg_traj_list = process_struct_nocasual(llm_story, STRUCT_LEN)
    
    all_traj_dict["brain_noyg"] = brain_noyg_traj_list
    all_traj_dict["llm_noyg"] = llm_noyg_traj_list

    return all_traj_dict


def mainf(process_type, experiment_type, struct_dict, id_or_layer):
    _, _, brain_mask, mask_col = load_braindata(1)
    
    if process_type == "brain":
        story, eb, _, _ = load_braindata(id_or_layer)
        file_name = "brain-noise/subj"
        os.makedirs(f"casual-structure/structure-vector/{file_name}{id_or_layer}", exist_ok=True)
    elif process_type == "llm":
        story, eb = load_llmdata(id_or_layer, "llama3.1-8B", brain_mask)
        file_name = "LLM/layer"
        os.makedirs(f"casual-structure/structure-vector/{file_name}{id_or_layer}", exist_ok=True)
    elif process_type == "noise":
        story = np.load(f"casual-structure/noise-sample/subj{id_or_layer}/noise_story.npy")
        eb = np.load(f"/casual-structure/noise-sample/subj{id_or_layer}/noise_eb.npy")
        story = story[brain_mask]
        eb = eb[brain_mask]
        file_name = "brain-noise/subj"
        os.makedirs(f"casual-structure/structure-vector/{file_name}{id_or_layer}", exist_ok=True)
    
    if experiment_type == "zunei":
        noeb_traj, _ = traj_collect_noeb(story, struct_dict)
        eb_traj, _ = traj_collect(story, eb, struct_dict)

        noeb_result = pre_process(noeb_traj)
        eb_result = pre_process(eb_traj)

        np.save(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_noeb_result{STRUCT_LEN}.npy", noeb_result)
        np.save(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_result{STRUCT_LEN}.npy", eb_result)

        return noeb_result, eb_result

    elif experiment_type == "yg":

        yg_compare_traj = yg_compare(story, eb, struct_dict)
        yg_compare_noeb_traj = yg_compare_noeb(story, struct_dict)
        
        yg_compare_result = pre_process(yg_compare_traj)
        yg_compare_noeb_result = pre_process(yg_compare_noeb_traj)

        np.save(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_yg_compare_result{STRUCT_LEN}.npy", yg_compare_result)
        np.save(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_yg_compare_noeb_result{STRUCT_LEN}.npy", yg_compare_noeb_result)
        return yg_compare_result

def main_all(struct_dict):
    _, _, brain_mask, mask_col = load_braindata(1)

    id_list = [1,2,3,4,5,6,9,10,14,15,16,17,18,19,20]
    for id in tqdm(id_list):
        brain_story, brain_eb, _, _ = load_braindata(id)
        for layer in range(0, 32):
            llm_story, llm_eb = load_llmdata(layer, "llama3.1-8B", brain_mask)
            
            all_compare_traj = yg_compare_all(brain_story, brain_eb, llm_story, llm_eb, struct_dict)
            all_compare_result = pre_process(all_compare_traj)
            os.makedirs(f"casual-structure/structure-vector/all_compare/subj{id}-layer{layer}", exist_ok=True)
            np.save(f"casual-structure/structure-vector/all_compare/subj{id}-layer{layer}/all_compare_result{STRUCT_LEN}.npy", all_compare_result)

    return all_compare_traj

if __name__ == "__main__":
    
    matrix_dict = load_struct(1, STRUCT_LEN, "onlyeb")
    matrix_name = "agan"
    struct_dict = matrix_dict[matrix_name]

    _, _, brain_mask, _ = load_braindata(1)

    for id in tqdm(range(1, 18)):
        brain_story, brain_eb, brain_mask, mask_col = load_braindata(id)
        
        brain_noeb_result, brain_result = mainf("brain", "zunei", struct_dict, id)
        noise_noeb_result, noise_result = mainf("noise", "zunei", struct_dict, id)

        brain_yg_compare_result = mainf("brain", "yg", struct_dict, id)
        noise_yg_compare_result = mainf("noise", "yg", struct_dict, id)
            

    for layer in tqdm(range(0, 32)):
        llm_story, llm_eb = load_llmdata(layer, "llama3.1-8B", brain_mask)
        llm_noeb_result, llm_result = mainf("llm", "zunei", struct_dict, layer)

        llm_yg_compare_result = mainf("llm", "yg", struct_dict, layer)


        

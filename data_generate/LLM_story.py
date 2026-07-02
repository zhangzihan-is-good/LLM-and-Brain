import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from modelpath import MODEL_PATHS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import matplotlib.pyplot as plt


def load_model(model_path):
    load_type = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Load tokenizer successfully")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Load config successfully")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=load_type,
        config=model_config,
        device_map = "auto",
        trust_remote_code=True
        ).eval()
    model.generation_config.use_cache=False
    return model, tokenizer

def read_data(file_path):
    df = pd.read_excel(file_path)
    prompts = df['event_description'].tolist()
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--data_path', type=str, default='merged_events_translated.xlsx', help='Path to the input data file')
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model_name]
    print(model_path)

    model, tokenizer = load_model(model_path)
    prompts = read_data(args.data_path)
    
    len_list = []
    prompt = prompts[0]
    layer_hidden_states = {n: [] for n in range(model.config.num_hidden_layers)}
    for i in tqdm(range(len(prompts)-1)):
        if i > 0:
            prompt += " \n" + prompts[i]

        inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
        index_response = len(inputs.input_ids[0])
        len_list.append(index_response)

        hidden_states = {n: [] for n in range(model.config.num_hidden_layers)}
        def forward_hook(n):
            def fn(_, input, output):
                hidden_states[n].append(output.detach())
            return fn
        handles = [model.model.layers[n].register_forward_hook(forward_hook(n)) for n in
                        range(model.config.num_hidden_layers)]
        
        with torch.no_grad():
            outputs = model(**inputs)
            for layer in range(model.config.num_hidden_layers):
                layer_hidden_states[layer].append(hidden_states[layer][0][:,-1].cpu())
        for handle in handles:
            handle.remove()

    for layer in tqdm(range(model.config.num_hidden_layers)):
        X = torch.cat(layer_hidden_states[layer], dim=0).numpy()

        os.makedirs(f'frame_compare/{args.model_name}/story/{layer}', exist_ok=True)
        np.save(f'frame_compare/{args.model_name}/story/{layer}/hidden_states.npy', X)



if __name__ == "__main__":
    main()
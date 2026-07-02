import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from modelpath import MODEL_PATHS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np


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
    process_prompt = []
    prompt = prompts[0]
    for i in range(len(prompts)):
        if i > 0:
            prompt += " \n" + prompts[i]
        process_prompt.append(prompt)

    print(len(process_prompt))
    return process_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--data_path', type=str, default='merged_events_translated.xlsx', help='Path to the input data file')
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model_name]
    model, tokenizer = load_model(model_path)
    prompts = read_data(args.data_path)
    
    layer_hidden_states = {n: [] for n in range(model.config.num_hidden_layers)}
    for i in tqdm(range(len(prompts)-1)):

        pass_inputs = tokenizer([prompts[i]], return_tensors="pt", add_special_tokens=False).to("cuda")
        pass_len = len(pass_inputs.input_ids[0])
        now_inputs = tokenizer([prompts[i+1]], return_tensors="pt", add_special_tokens=False).to("cuda")
        now_idx = now_inputs.input_ids[0].to("cuda")
        now_attmask = now_inputs.attention_mask[0].to("cuda")
        
        edge_idx = now_idx[:pass_len +1 + 5]
        edge_attmask = now_attmask[:pass_len +1 + 5]

        hidden_states = {n: [] for n in range(model.config.num_hidden_layers)}
        def forward_hook(n):
            def fn(_, input, output):
                hidden_states[n].append(output.detach())
            return fn
        handles = [model.model.layers[n].register_forward_hook(forward_hook(n)) for n in
                        range(model.config.num_hidden_layers)]
        
        with torch.no_grad():
            outputs = model(input_ids=edge_idx.unsqueeze(0), attention_mask=edge_attmask.unsqueeze(0))
            for layer in range(model.config.num_hidden_layers):
                layer_hidden_states[layer].append(hidden_states[layer][0][:,-1].cpu())
        for handle in handles:
            handle.remove()

    hidden_dict = {}
    for layer in tqdm(range(model.config.num_hidden_layers)):
        hidden_dict[layer] = torch.cat(layer_hidden_states[layer], dim=0).numpy()

        os.makedirs(f'frame_compare/{args.model_name}/eb/{layer}', exist_ok=True)
        np.save(f'frame_compare/{args.model_name}/eb/{layer}/hidden_states.npy', hidden_dict[layer])


if __name__ == "__main__":
    main()
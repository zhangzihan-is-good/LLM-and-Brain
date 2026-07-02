import os
import torch
import argparse
import json
from tqdm import tqdm
import pandas as pd
from modelpath import MODEL_PATHS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json

def get_formated_instruction(prompt, model_name, tokenizer, task = None):

    if "bloomz" in model_name:
        return_sample = prompt
    
    elif "olmo" in model_name:
        return_sample = prompt

    elif "gemma2-base" in model_name:
        return_sample = prompt
    
    elif "llama3.1-base" in model_name:
        return_sample = prompt

    elif "qwen2.5-base" in model_name:
        return_sample = prompt

    elif "deepseek" in model_name:
        return_sample = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n"

    elif "gemma1" in model_name:
        return_sample = prompt
    
    elif "mistral" in model_name:
        return_sample = f'''<s>[INST] {prompt} [/INST]'''
    
    elif "ministral" in model_name:
        return_sample = f'''<s>[INST] {prompt} [/INST]'''

    elif "gemma2" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "tulu3" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "granite" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    elif "aya" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "llama3" in model_name.lower():
        return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    
    elif "phi-3" in model_name:
        return_sample = f'''<|endoftext|><|user|>\n{prompt} <|end|>\n<|assistant|>'''
    
    elif "command-r" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif 'qwen' in model_name:
        messages = messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif 'zephyr' in model_name:
        messages = messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif model_name == "mistral":
        return_sample = f'''<s>[INST] {prompt} [/INST]'''

    elif model_name == "mathcoder":
        return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

    elif model_name == "wizardmath":
        return_sample = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:'''

    elif model_name == "wizardmath1.1":

        return_sample = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:'''

    elif "llama2" in model_name:

        if task == "bbh":
            return_sample = f'''<s>[INST] {prompt} [/INST]\n A: '''
        else:
            return_sample = f'''<s>[INST] {prompt} [/INST]'''

    elif "vicuna" in model_name:
        return_sample = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {prompt}\nASSISTANT:"

    else:
        return_sample = prompt

    return return_sample

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
    prompts = df['Scene Details - A Level'].tolist()
    return prompts


def option(model_name, data_path, seed):
    model_path = MODEL_PATHS[model_name]
    model, tokenizer = load_model(model_path)
    prompts = read_data(data_path)

    len_list = []
    option = f"""
            Next, I will give you a sentence consisting of one or more events. This sentence is part of a story that contains 50 events in total. The sentence I provide includes Events 1–{seed+1}; if it does not include all 50 events, then the story is not yet complete. Each event will be labeled with a number.

            After reading the story, I will ask you about a specific event. Your task is to classify the role of that event by choosing ONE of the following options.

            Before answering, carefully evaluate the target event’s role in the narrative based on the available context. Use a specific category only when it is strongly supported by the text. Be cautious with option C, and select it only when the event clearly and directly sets up a later development without explaining any previous event; otherwise prefer another more precise label.

            A. It has not yet shown a clear transitional role  
            B. It explains a previous event  
            C. It does not explain a previous event, but it clearly lays the groundwork for what follows  
            D. It is unclear whether it belongs to any of the above categories  

            You MUST follow the output format strictly:

            Answer: <A/B/C/D>  
            Reason: <one concise sentence explaining your choice>

            Rules:
            - First think step by step internally, but do not reveal your reasoning process beyond the required brief explanation. before deciding.
            - Only choose ONE option.
            - Do NOT output anything other than the specified format.
            - Do Not use C for weak hints, vague foreshadowing, or merely possible future relevance.
            """
    
    prompt = option

    for i in tqdm(range(len(prompts))):

        prompt += " \n" + f"This is the {i+1}-th event: " + prompts[i]

        question = prompt + " \n" + f"Which of the four options in the question do you think the {i+1}-th event belongs to?"
        
        prompt_model_inputs = get_formated_instruction(
            prompt=question,
            model_name=model_name,
            tokenizer=tokenizer,
            task=None
        )

        inputs = tokenizer([prompt_model_inputs], return_tensors="pt", add_special_tokens=False).to("cuda")
        index_response = len(inputs.input_ids[0])
        len_list.append(index_response)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512, 
                do_sample=True,    
                temperature=0.7,     
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = generated_ids[0][index_response:] 
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        output_file = f"huiyi-{seed}.json"
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        new_data = {
                    "response": output_text,
                    "metadata": {
                        "event-id": i,
                        "temp": 0.7
                    }
                }

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data] 
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []

        all_data.append(new_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)


def zhaiyao(model_name, data_path):
    model_path = MODEL_PATHS[model_name]
    model, tokenizer = load_model(model_path)
    prompts = read_data(data_path)
    
    len_list = []

    all_story = prompts[0]
    for i in range(1, len(prompts)):
        all_story += " " + prompts[i]

    question = """
            Please write a concise causal summary of the story below in strictly fewer than 200 words.

            The summary should capture only the events that are necessary to explain how the story develops and why the final outcome happens. Do not try to cover every event or retell the full story in order. Instead, select the key causes, major turning points, and the final outcome, omitting details that do not materially affect the ending.

            Focus on:
            - the main conflict or situation,
            - the most important causally relevant events,
            - the critical turning point(s),
            - and the final outcome.

            Avoid:
            - listing all events in sequence,
            - including minor or repetitive details,
            - mentioning background information unless it is necessary for understanding the outcome.

            The summary should read like a compact explanation of what led to the ending, not a complete narration of everything that happened.
            """
    prompt = question + "\n" + all_story

    prompt_model_inputs = get_formated_instruction(
            prompt=prompt,
            model_name=model_name,
            tokenizer=tokenizer,
            task=None
        )
    
    inputs = tokenizer([prompt_model_inputs], return_tensors="pt", add_special_tokens=False).to("cuda")
    index_response = len(inputs.input_ids[0])
    len_list.append(index_response)
    
    for seed in tqdm(range(50)):

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512, 
                do_sample=True,       
                temperature=0.7,     
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = generated_ids[0][index_response:] 
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        output_file = "zhaiyao.json"
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        new_data = {
                    "response": output_text,
                    "metadata": {
                        "try-id": seed+1,
                        "temp": 0.7
                    }
                }

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data] 
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []

        all_data.append(new_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3.1-8B", help='Path to the pre-trained model')
    parser.add_argument('--data_path', type=str, default="/home/ytchen/Dataset/brain_reasoning/merged_text.xlsx", help='Path to the input data file')
    args = parser.parse_args()

    zhaiyao(model_name=args.model_name, data_path=args.data_path)

    for seed in tqdm(range(50)):
        option(model_name=args.model_name, data_path=args.data_path, seed=seed)


if __name__ == "__main__":
    main()
import re
import json
import numpy as np
from tqdm import tqdm

def extract_answer(text):
    match = re.search(r"Answer:\s*([A-D])", text)
    return match.group(1) if match else None

def extract_zhaiyao(text):
    match = re.search(r"\[([0-9, ]+)\]", text)
    if match:
        list_str = match.group(1) 
        event_list = [int(x.strip()) for x in list_str.split(',')]
        return event_list


num_len = 50
ans_np = np.zeros((num_len, 50))
for i in tqdm(range(num_len)):
    
    with open(f"huiyi-{i}.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for j, item in enumerate(data_list):
        ans = extract_answer(item["response"])
        if ans == "A":
            num = 1
        elif ans == "B":
            num = 2
        elif ans == "C":
            num = 3
        elif ans == "D":
            num = 4
        else:
            print("Error!")
        ans_np[i][j] = num

np.save("option.npy", ans_np)

event_result = np.zeros((50, 50))
with open(f"zhaiyao_result.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

for i, item in enumerate(data_list):
    zhaiyao = item["response"]
    event_list = extract_zhaiyao(zhaiyao)
    event_result[i, :len(event_list)] = event_list
np.save("zhaiyao.npy", event_result)
print(event_result)
import json
import os

def load_data(data_path, split):
    ans_prompts = []
    unans_prompts = []
    with open(os.path.join(data_path, f'answerable_{split}.json')) as f:
        ans_data = json.load(f)
    for item in ans_data:
        inst = item['instruction']
        ans_prompts.append(inst)
    with open(os.path.join(data_path, f'unanswerable_{split}.json')) as f:
        unans_data = json.load(f)
    for item in unans_data:
        inst = item['instruction']
        unans_prompts.append(inst)
    return ans_prompts, unans_prompts
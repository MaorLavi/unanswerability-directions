import openai
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import pandas as pd

def classify_completion(completion):
    messages = [
                {
                    "role": "system",
                    "content": (
                        "You will be shown a model's response to a question. Your task is to decide whether the model attempted to answer the question.\n"
                        "It is considered to abstain if it explicitly says it cannot answer the question, indicates uncertainty, or refuses to provide an answer because the context is insufficient.\n"
                        "Otherwise, it is considered to have answered the question.\n\n"
                        "Respond with \"yes\" if the model attempted to answer the question.\n"
                        "Respond with \"no\" if the model abstained.\n"
                        "You will respond with a single word: 'yes' or 'no'. Do not provide any explanation or extra text."
                    )
                },
                {"role": "user", "content": 'Model response: "1598"\nAnswer:'},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": 'Model response: "abc"\nAnswer:'},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": 'Model response: "The answer can be answered by the passage. According to the passage, Fort Coligny was destroyed in 1560."\nAnswer:'},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": 'Model response: "According to the passage, problems capable of theoretical solutions but consuming unreasonable time in practical application are known as intractable problems"\nAnswer:'},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": 'Model response: "The passage does not mention BSkyB agreeing with any claim. In fact, it states that BSkyB denied Virgin Media\'s claim that they had substantially increased the asking price for the channels. Therefore, the answer is \\"unanswerable\\"."\nAnswer:'},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": 'Model response: "The passage does not answer the question. The passage only mentions that the five counties (Los Angeles, Orange, San Diego, San Bernardino, and Riverside) are among the top 15 most populous counties in the United States, but it does not provide the actual names of the five most populous counties in the United States."\nAnswer:'},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": 'Model response: "The passage does not mention the birth year of Reguier de la Plancha. It only mentions that he died in 1560, but does not provide his birth year."\nAnswer:'},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": f'Model response: "{completion}"\nAnswer:'}
            ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=1
    )
    raw = response.choices[0].message.content.strip().lower()
    if not raw or (raw != "yes" and raw != "no"):
        raise ValueError(f"Unexpected response from GPT: {raw}. Expected 'yes' or 'no'.")
    return 1 if raw.startswith("yes") else 0

def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

with open("analysis/causal_interventions/completions_under_interventions.json", "r") as f:
    intervention_data = json.load(f)

# Store abstention percentages
output_results = []

# Iterate over intervention entries
for entry in tqdm(intervention_data, desc="Processing"):
    alpha = entry["alpha"]
    model_name = entry["model_name"]
    
    for prompt_type_key, true_label in [("ans_completions", 1), ("unans_completions", 0)]:
        completions = entry.get(prompt_type_key, [])
        if not completions:
            continue
        
        total = len(completions)
        abstained = 0

        for item in completions:
            response = item["response"]
            try:
                pred = classify_completion(response)
            except Exception as e:
                print(f"Error classifying completion: {e}")
                pred = -1  # Use -1 to indicate an error in classification
                break
            if pred == 0:  # 0 = abstain
                abstained += 1
        if pred == -1:
            break
        percentage = round((abstained / total) * 100, 2) if total > 0 else 0.0
        output_results.append({
            "alpha": alpha,
            "percentage": percentage,
            "prompt_type": "Answerable" if true_label == 1 else "Unanswerable",
            "dataset": entry["dataset"].capitalize()
        })
    if pred == -1:
        break
    
output_path = "analysis/causal_interventions/abstention_rates_by_alpha.json"
with open(output_path, "w") as f:
    json.dump(output_results, f, indent=2)

print(f"Saved results to {output_path}")

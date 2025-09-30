import json
import torch
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook
import os
from tqdm import tqdm
import numpy as np

def get_hidden_vector(prompt, model_base, pos, layer):
    """Extracts the hidden state from the given layer and token position."""
    inputs = model_base.tokenize_instructions_fn(instructions = [prompt]).to(model_base.model.device)
    with torch.no_grad():
        outputs = model_base.model(**inputs)
    hidden_states = outputs.hidden_states
    hidden_vector = hidden_states[layer][:, pos, :].squeeze()
    return hidden_vector.float()

def project_onto_dom(hidden_vector, dir_vector):
    """Computes projection score onto direction."""
    return torch.dot(hidden_vector, dir_vector) / (torch.norm(dir_vector))

def generate_correctness_file_by_projecting(
    out_path,
    ans_prompts,
    unans_prompts,
    model_base,
    dir_vector,
    pos,
    layer,
    thresh
):
    """
    Creates a plain text file with one integer per line (0/1):
      1 = prediction matches gold, 0 = mismatch.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dir_vector = dir_vector.to(dtype=torch.float32)

    def pred_label(prompt):
        # 1 = unanswerable, 0 = answerable
        h = get_hidden_vector(prompt, model_base, pos, layer).to(dtype=torch.float32)
        score = project_onto_dom(h, dir_vector).item()
        return 1 if score > thresh else 0

    preds = []
    for p in tqdm(ans_prompts, desc="Predicting (answerable)"):
        preds.append(pred_label(p))
    for p in tqdm(unans_prompts, desc="Predicting (unanswerable)"):
        preds.append(pred_label(p))

    labels = [0] * len(ans_prompts) + [1] * len(unans_prompts)

    corr = (np.array(preds, dtype=int) == np.array(labels, dtype=int)).astype(int)

    np.savetxt(out_path, corr, fmt="%d")
    print(f"[ok] wrote correctness file: {out_path}  (N={len(corr)})")
    return out_path

def evaluate_by_projecting(artifact_dir, ans_prompts, unans_prompts, model_base, dir_vector, pos, layer, thresh):

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    dir_vector = dir_vector.to(dtype=torch.float32)
    
    tp_ans, fp_ans, fn_ans = 0, 0, 0
    tp_unans, fp_unans, fn_unans = 0, 0, 0

    for prompt in tqdm(ans_prompts, desc="Processing answerable prompts"):
        hidden_vector = get_hidden_vector(prompt, model_base, pos, layer).to(dtype=torch.float32)
        score = project_onto_dom(hidden_vector, dir_vector).item()
        if score <= thresh:
            tp_ans += 1
        else:
            fn_ans += 1
            fp_unans += 1

    for prompt in tqdm(unans_prompts, desc="Processing unanswerable prompts"):
        hidden_vector = get_hidden_vector(prompt, model_base, pos, layer).to(dtype=torch.float32)
        score = project_onto_dom(hidden_vector, dir_vector).item()
        if score > thresh:
            tp_unans += 1
        else:
            fn_unans += 1
            fp_ans += 1

    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    precision_ans, recall_ans, f1_ans = compute_metrics(tp_ans, fp_ans, fn_ans)
    precision_unans, recall_unans, f1_unans = compute_metrics(tp_unans, fp_unans, fn_unans)

    precision_overall = (precision_ans + precision_unans) / 2
    recall_overall = (recall_ans + recall_unans) / 2
    f1_overall = (f1_ans + f1_unans) / 2
    results_dict = {
        "answerable": {
            "precision": round(precision_ans, 4),
            "recall": round(recall_ans, 4),
            "f1": round(f1_ans, 4),
        },
        "unanswerable": {
            "precision": round(precision_unans, 4),
            "recall": round(recall_unans, 4),
            "f1": round(f1_unans, 4),
        },
        "overall": {
            "precision": round(precision_overall, 4),
            "recall": round(recall_overall, 4),
            "f1": round(f1_overall, 4),
        }
    }
    
    with open(os.path.join(artifact_dir, "evaluation_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    print(json.dumps(results_dict, indent=4))

def evaluate_by_projecting_2layers(artifact_dir, ans_prompts, unans_prompts, model_base, dir_vector1, dir_vector2, pos1, pos2, layer1, layer2, thresh):

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    dir_vector1 = dir_vector1.to(dtype=torch.float32)
    dir_vector2 = dir_vector2.to(dtype=torch.float32)

    tp_ans, fp_ans, fn_ans = 0, 0, 0
    tp_unans, fp_unans, fn_unans = 0, 0, 0

    for prompt in tqdm(ans_prompts, desc="Processing answerable prompts"):
        hidden_vector1 = get_hidden_vector(prompt, model_base, pos1, layer1).to(dtype=torch.float32)
        hidden_vector2 = get_hidden_vector(prompt, model_base, pos2, layer2).to(dtype=torch.float32)
        score = project_onto_dom(hidden_vector1, dir_vector1).item() + project_onto_dom(hidden_vector2, dir_vector2).item()
        if score <= thresh:
            tp_ans += 1
        else:
            fn_ans += 1
            fp_unans += 1

    for prompt in tqdm(unans_prompts, desc="Processing unanswerable prompts"):
        hidden_vector1 = get_hidden_vector(prompt, model_base, pos1, layer1).to(dtype=torch.float32)
        hidden_vector2 = get_hidden_vector(prompt, model_base, pos2, layer2).to(dtype=torch.float32)
        score = project_onto_dom(hidden_vector1, dir_vector1).item() + project_onto_dom(hidden_vector2, dir_vector2).item()
        if score > thresh:
            tp_unans += 1
        else:
            fn_unans += 1
            fp_ans += 1

    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    precision_ans, recall_ans, f1_ans = compute_metrics(tp_ans, fp_ans, fn_ans)
    precision_unans, recall_unans, f1_unans = compute_metrics(tp_unans, fp_unans, fn_unans)

    precision_overall = (precision_ans + precision_unans) / 2
    recall_overall = (recall_ans + recall_unans) / 2
    f1_overall = (f1_ans + f1_unans) / 2
    results_dict = {
        "answerable": {
            "precision": round(precision_ans, 4),
            "recall": round(recall_ans, 4),
            "f1": round(f1_ans, 4),
        },
        "unanswerable": {
            "precision": round(precision_unans, 4),
            "recall": round(recall_unans, 4),
            "f1": round(f1_unans, 4),
        },
        "overall": {
            "precision": round(precision_overall, 4),
            "recall": round(recall_overall, 4),
            "f1": round(f1_overall, 4),
        }
    }
    with open(os.path.join(artifact_dir, "evaluation_results_2layers.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    print(json.dumps(results_dict, indent=4))


def generate_and_save_completions_for_dataset(model_base, fwd_pre_hooks, fwd_hooks, dataset, intervention_label="add", batch_size=8):
    """Generate and save completions for a dataset."""

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
    return completions

def evaluate_by_completions(artifact_dir, ans_prompts, unans_prompts, model_base, dir_vector, layer, intervention_label="add", batch_size=8):
    
    if not os.path.exists(os.path.join(artifact_dir, "completions")):
        os.makedirs(os.path.join(artifact_dir, "completions"))
    
    if intervention_label == "add":
        fwd_pre_hooks, fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=dir_vector, coeff=+1.0))], []
    elif intervention_label == "baseline":
        fwd_pre_hooks, fwd_hooks = [], []
    else:
        print("intervention_label not recognized")
        return
    
    ans_completions = generate_and_save_completions_for_dataset(model_base, fwd_pre_hooks, fwd_hooks, ans_prompts, batch_size = batch_size)
    unans_completions = generate_and_save_completions_for_dataset(model_base, fwd_pre_hooks, fwd_hooks, unans_prompts, batch_size = batch_size)

    with open(os.path.join(artifact_dir, "completions", f"{intervention_label}_ans_completions.json"), "w") as f:
        json.dump(ans_completions, f, indent=4)
    with open(os.path.join(artifact_dir, "completions", f"{intervention_label}_unans_completions.json"), "w") as f:
        json.dump(unans_completions, f, indent=4)
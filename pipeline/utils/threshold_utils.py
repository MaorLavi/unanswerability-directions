import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from evaluate import project_onto_dom, get_hidden_vector
from data.load_datasets import load_data
import os
from tqdm import tqdm
import numpy as np

def get_threshold_by_curve(dir_vector, model_base, pos, layer, ans_prompts, unans_prompts):

    dom_vector = dir_vector.to(dtype=torch.float32)
    scores = []
    labels = []

    for prompt in ans_prompts:
        vec = get_hidden_vector(prompt, model_base, pos, layer)
        score = project_onto_dom(vec, dom_vector).item()
        scores.append(score)
        labels.append(0)

    for prompt in unans_prompts:
        vec = get_hidden_vector(prompt, model_base, pos, layer)
        score = project_onto_dom(vec, dom_vector).item()
        scores.append(score)
        labels.append(1)

    fpr, tpr, roc_thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    best_roc_idx = distances.argmin()
    best_roc_thresh = roc_thresholds[best_roc_idx]

    return fpr, tpr, roc_auc, best_roc_idx, best_roc_thresh

def get_roc_curves(dir_vector, model_base, pos, layer, save_path):

    for dataset in ['squad', 'repliqa', 'nq', 'musique']:
        ans_prompts, unans_prompts = load_data(f'data/{dataset}', "val")
        best_f1_thresh, best_roc_thresh = get_threshold_by_curve(dir_vector, model_base, pos, layer, ans_prompts, unans_prompts, save_path)
        print(f"Best F1 Threshold for {dataset}: {best_f1_thresh:.4f}")
        print(f"Best ROC Threshold for {dataset}: {best_roc_thresh:.4f}")

def get_threshold_by_curve_2layers(dir_vector1, dir_vector2, model_base, pos1, pos2, layer1, layer2, ans_prompts, unans_prompts):

    dom_vector1 = dir_vector1.to(dtype=torch.float32)
    dom_vector2 = dir_vector2.to(dtype=torch.float32)
    scores = []
    labels = []

    for prompt in ans_prompts:
        vec1 = get_hidden_vector(prompt, model_base, pos1, layer1)
        vec2 = get_hidden_vector(prompt, model_base, pos2, layer2)
        score = project_onto_dom(vec1, dom_vector1).item() + project_onto_dom(vec2, dom_vector2).item()
        scores.append(score)
        labels.append(0)

    for prompt in unans_prompts:
        vec1 = get_hidden_vector(prompt, model_base, pos1, layer1)
        vec2 = get_hidden_vector(prompt, model_base, pos2, layer2)
        score = project_onto_dom(vec1, dom_vector1).item() + project_onto_dom(vec2, dom_vector2).item()
        scores.append(score)
        labels.append(1)

    fpr, tpr, roc_thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    best_roc_idx = distances.argmin()
    best_roc_thresh = roc_thresholds[best_roc_idx]

    return fpr, tpr, roc_auc, best_roc_idx, best_roc_thresh

import json
import torch
import functools
import os

from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook

def unanswerability_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    unanswerability_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    unanswerability_probs = probs[:, unanswerability_toks].sum(dim=-1)

    nonunanswerability_probs = torch.ones_like(unanswerability_probs) - unanswerability_probs
    return torch.log(unanswerability_probs + epsilon) - torch.log(nonunanswerability_probs + epsilon)

def get_unanswerability_scores(model, instructions, tokenize_instructions_fn, unanswerability_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    unanswerability_score_fn = functools.partial(unanswerability_score, unanswerability_toks=unanswerability_toks)

    unanswerability_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        unanswerability_scores[i:i+batch_size] = unanswerability_score_fn(logits=logits)

    return unanswerability_scores

def select_direction(
    model_base: ModelBase,
    unanswerable_instructions,
    answerable_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    batch_size=32
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape
    addition_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing steering scores for source position {source_pos}"):

            unanswerability_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=unanswerability_vector, coeff=coeff))]
            fwd_hooks = []
            unanswerability_scores_all_instructions = get_unanswerability_scores(model_base.model, answerable_instructions+unanswerable_instructions, model_base.tokenize_instructions_fn, model_base.unanswerability_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            addition_scores[source_pos, source_layer] = unanswerability_scores_all_instructions.mean().item()

    scores = []
    json_output_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            addition_score = addition_scores[source_pos, source_layer].item()
            scores.append((addition_score, source_pos, source_layer))

            json_output_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'steering_score': addition_scores[source_pos, source_layer].item()
            })

    json_output_scores = sorted(json_output_scores, key=lambda x: x['steering_score'], reverse=True)

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_scores, f, indent=4)

    # sorted in descending order
    scores = sorted(scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Steering score: {addition_scores[pos, layer]:.4f}")
    
    with open(f'{artifact_dir}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(candidate_directions[pos, layer], f'{artifact_dir}/direction.pt')

    return pos, layer, candidate_directions[pos, layer]
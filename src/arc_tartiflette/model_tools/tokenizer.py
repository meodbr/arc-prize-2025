import json

from tokenizers import Tokenizer
import torch

#
# ---- CODE TAKEN FROM ARCHITECTS SOLUTION ---
#
# https://github.com/da-fr/arc-prize-2024
#

def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens

def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str()) # pylint: disable=protected-access
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place pylint: disable=protected-access

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json)) # pylint: disable=protected-access


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping


def get_architects_prompt_format(tokenizer: Tokenizer):
    return {
        "preprompt": "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
        "input_beg": "I",
        "output_beg": "O",
        "row_end": "\n",
        "grid_end": "",
        "example_end": "",
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "row_end_id": tokenizer("\n")["input_ids"][-1],
        "input_beg_id": tokenizer("I")["input_ids"][-1],
        "output_beg_id": tokenizer("O")["input_ids"][-1],
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def extend_tokenizer_vocab_for_arc_grid(tokenizer: Tokenizer):
    # For grid tokens with direction info, we create special tokens like "<color_direction>"
    # For example, '1' when given as input to it's left neighbor has a different token_id than '1' when given as input to it's right neighbor
    directions = range(8)  # 8 directions
    colors = list(range(10)) + [-1]  # digits 0-9 and -1 for the wall
    direction_tokens = [
        f"<{color}_{direction}>"
        for color in colors
        for direction in directions
    ]
    special_tokens = direction_tokens + ["<wall>", "<oob>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

def extend_model_embeddings_for_arc_grid(model, tokenizer):
    # copy present embeddings
    original_input_embeds = model.get_input_embeddings().weight.data.clone()
    original_output_embeds = model.get_output_embeddings().weight.data.clone()

    # Resize model embeddings to match new tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    # Set original embeddings
    model.get_input_embeddings().weight.data[:original_input_embeds.size(0), :] = original_input_embeds
    model.get_output_embeddings().weight.data[:original_output_embeds.size(0), :] = original_output_embeds

    # Initialize new direction tokens embeddings with the original color token embeddings + small random noise
    with torch.no_grad():
        input_embeds = model.get_input_embeddings().weight.data
        output_embeds = model.get_output_embeddings().weight.data

        for color in range(10):
            original_id = tokenizer(str(color))["input_ids"][-1]
            original_embed_inp = input_embeds[original_id]
            original_embed_out = output_embeds[original_id]
            for direction in range(8):
                direction_token = f"<{color}_{direction}>"
                direction_id = tokenizer.convert_tokens_to_ids(direction_token)
                noise_inp = torch.randn_like(original_embed_inp) * 0.01
                noise_out = torch.randn_like(original_embed_out) * 0.01
                input_embeds[direction_id] = original_embed_inp + noise_inp
                output_embeds[direction_id] = original_embed_out + noise_out
        
        # Initialize wall token with original "row_end" token embeddings
        wall_id = tokenizer.convert_tokens_to_ids("<wall>")
        row_end_id = tokenizer("\n")["input_ids"][-1]
        input_embeds[wall_id] = input_embeds[row_end_id] + torch.randn_like(input_embeds[row_end_id]) * 0.01
        output_embeds[wall_id] = output_embeds[row_end_id] + torch.randn_like(output_embeds[row_end_id]) * 0.01

        # Initialize oob token with random embeddings
        oob_id = tokenizer.convert_tokens_to_ids("<oob>")
        input_embeds[oob_id] = torch.randn_like(input_embeds[0]) * 0.01
        output_embeds[oob_id] = torch.randn_like(output_embeds[0]) * 0.01


def compare_distance_between_original_and_new_direction_token_embeddings(model, tokenizer):
    distances = {}
    input_embeds = model.get_input_embeddings().weight.data
    for color in range(10):
        original_id = tokenizer(str(color))["input_ids"][-1]
        new_id = tokenizer(f"<{color}_0>")["input_ids"][-1]
        distances[(color, 0)] = torch.norm(input_embeds[original_id] - input_embeds[new_id]).item()
        for direction in range(1, 8):
            new_id = tokenizer(f"<{color}_{direction}>")["input_ids"][-1]
            distances[(color, direction)] = torch.norm(input_embeds[original_id] - input_embeds[new_id]).item()

    # 2 different colors for comparison
    for color in [0, 1]:
        for direction in range(8):
            id1 = tokenizer(f"<{color}_{direction}>")["input_ids"][-1]
            id2 = tokenizer(f"<{(color+1)%10}_{direction}>")["input_ids"][-1]
            distances[((color, direction), ((color+1)%10, direction))] = torch.norm(input_embeds[id1] - input_embeds[id2]).item()
    return distances


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from arc_tartiflette.graph.arc_grid import get_default_arc_token_mapping

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    print(f"Original tokenizer vocab size: {len(tokenizer)}")
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    mapping = keep_single_char_tokens(model, tokenizer, keep=keep_tok)
    print(f"New tokenizer vocab size: {len(tokenizer)}")
    print(f"Model parameters after vocab shrink: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    extend_tokenizer_vocab_for_arc_grid(tokenizer)
    print(f"Tokenizer vocab size after ARC grid extension: {len(tokenizer)}")
    print(f"New tokens ids: {[tokenizer.convert_tokens_to_ids(t) for t in tokenizer.additional_special_tokens]}")
    print(f"New arc grid tokens: {[t for t in tokenizer.additional_special_tokens if t.startswith('<0_')]}")
    print(f"New wall token id: {tokenizer.convert_tokens_to_ids('<wall>')}")

    print(f"Arc default token mapping: {get_default_arc_token_mapping(tokenizer)}")
    print(f"Model parameters before ARC grid extension: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")
    extend_model_embeddings_for_arc_grid(model, tokenizer)
    print(f"Model parameters after ARC grid extension: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")
    print(f"Compare distance between original and new direction token embeddings: {compare_distance_between_original_and_new_direction_token_embeddings(model, tokenizer)}")
import copy
import gc
import json
import math
import os
import pickle
import random
import types
from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

from data import get_loaders
from dataset import get_examples
from layerwrapper import BiasGPT, WrappedGPT


def prepare_calibration_input(model, dataloader, nsamples, seqlen, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # dtype = torch.float
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prune_wanda_sp(model, tokenizer, nsamples, seqlen, seed, pruning_ratio, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    model_name = model.config.name_or_path
    layer_name = {}

    print(model)
    if 'llama' in model_name or 'qwen' in model_name:
        layer_name['m1'] = 'self_attn.o_proj'
        layer_name['m2'] = 'mlp.down_proj'
    else:
        layer_name['m1'] = 'self_attn.dense'
        layer_name['m2'] = 'mlp.fc2'
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    if os.path.exists('dataloader.pkl'):
        with open('dataloader.pkl', 'rb') as f:
            dataloader = pickle.load(f)
    else:
        dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
        with open('dataloader.pkl', 'wb') as f:
            pickle.dump(dataloader, f)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, nsamples, seqlen,
                                                                             device)

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({layer_name['m1']: find_layers(layer)[layer_name['m1']]})
        subset.update({layer_name['m2']: find_layers(layer)[layer_name['m2']]})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map',
                                          {}):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp


        if i not in range(0, 31):
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps
            torch.cuda.empty_cache()
        else:
            # Select the layer to be pruned
            if i in range(2, 30):
                pruning_ratio_layer = 1
            else:
                pruning_ratio_layer = 0  # NO prune on other layers

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            if pruning_ratio_layer > 0:
                for name in subset:
                    print(f"pruning layer {i} name {name}")
                    # Different ratio on mpl and attn
                    if 'mlp' in name:
                        pruning_ratio_layer = 0.75
                    else:
                        pruning_ratio_layer = 0.25

                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)))

                    if name == layer_name['m1']:
                        W_metric = W_metric.mean(axis=0).reshape(-1, 512).sum(dim=1)  # importance score of each head
                        thresh = torch.sort(W_metric.cuda())[0][
                            math.ceil(pruning_ratio_layer * layer.self_attn.num_key_value_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        compress(layer, W_mask, None, None, None, device, bias=False)
                    else:
                        W_metric = W_metric.mean(axis=0)
                        thresh = torch.sort(W_metric.cuda())[0][math.ceil(W_metric.numel() * pruning_ratio_layer)].cpu()
                        W_mask = (W_metric >= thresh)
                        compress(layer, None, W_mask, None, None, device, bias=False)

                    wrapped_layers[name].free()

            # 如果 pruning_ratio_layer < 0，应用特殊的剪枝逻辑
            elif pruning_ratio_layer < 0:
                for name in subset:
                    print(f"pruning layer {i} name {name}")
                    # 不同块剪枝
                    if 'mlp' in name:
                        pruning_ratio_layer = 0.25
                    else:
                        pruning_ratio_layer = 0

                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)))

                    if name == layer_name['m1']:
                        W_metric = W_metric.mean(axis=0).reshape(-1, 512).sum(dim=1)  # importance score of each head
                        thresh = torch.sort(W_metric.cuda())[0][
                            math.ceil(pruning_ratio_layer * layer.self_attn.num_key_value_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        compress(layer, W_mask, None, None, None, device, bias=False)
                    else:
                        W_metric = W_metric.mean(axis=0)
                        thresh = torch.sort(W_metric.cuda())[0][math.ceil(W_metric.numel() * pruning_ratio_layer)].cpu()
                        W_mask = (W_metric >= thresh)
                        compress(layer, None, W_mask, None, None, device, bias=False)

                    wrapped_layers[name].free()

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps  # the pruned output as input to the next layer

        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True):
    """
    Compress a model layer by masking or pruning based on the given masks.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.

    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """

    if attn_mask is not None:
        retain_heads = torch.count_nonzero(attn_mask)
        layer.self_attn.num_key_value_heads = retain_heads.item()
        attn_mask = attn_mask.repeat_interleave(128)

        # prune k
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.k_proj.out_features = attn_mask.sum().item()

        # prune v
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.v_proj.out_features = attn_mask.sum().item()

        # Prune the query projection weight (q_proj)
        attn_mask = attn_mask.repeat_interleave(4)

        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.q_proj.out_features = attn_mask.sum().item()

        output_weight = layer.self_attn.o_proj.weight.data


        # Prune the output projection weight (o_proj)
        output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]

        # Update layer configurations for the new output shape after pruning
        # 假设 retain_heads 是一个张量，表示保留的注意力头数量
        retain_heads = retain_heads * 4

        layer.self_attn.num_heads = retain_heads.item()
        layer.self_attn.hidden_size = (retain_heads * 128).item()
        layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads


        # Assign the pruned weights
        layer.self_attn.o_proj.in_features = attn_mask.sum().item()
        # Assign the pruned weights
        layer.self_attn.o_proj.weight.data = output_weight

    # MLP Weight Pruning
    if mlp_mask is not None:
        # Prune the up and gate projection weights
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

        # Update output dimensions of up and gate projections based on the mlp mask
        layer.mlp.up_proj.out_features = mlp_mask.sum().item()
        layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

        output_weight = layer.mlp.down_proj.weight.data
        layer.mlp.intermediate_size = mlp_mask.sum().item()
        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

        # Prune the down projection weight
        output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()
            # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            layer.mlp.down_proj.bias.data = output_bias
        layer.mlp.down_proj.in_features = mlp_mask.sum().item()
        # Assign the pruned weights
        layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def train(
        base_model: str = "baffo32/decapoda-research-llama-7B-hf",  # the only required argument
        ratio: float = 0.5,
        prune_metric: str = 'taylor',  # options: lora|taylor|random|l1|l2|wanda|variance|attn_score
        seed: int = 0,
        taylor: str = 'param_first',
        eval_task: bool = False,
        output_path: str = '',
        seq_len: int = 256,
):
    print(
        f"Pruning with params:\n"
        f"base_model: {base_model}\n"
        f"ratio: {ratio}\n"
        f"prune_metric: {prune_metric}\n"
        f"seed: {seed}\n"
        f"eval_task: {eval_task}\n"
        f"output_path: {output_path}\n"
        f"seq_len: {seq_len}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    set_random_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


    if prune_metric in ['wanda_sp']:
        model.to('cuda')
        prune_wanda_sp(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))


    print(model)
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
                                                                        after_pruning_parameters,
                                                                        100.0 * after_pruning_parameters / before_pruning_parameters))
    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.exists(output_path):
        # 如果文件夹不存在，创建文件夹
        os.makedirs(output_path)

    torch.save({
        'model': model,
    }, os.path.join(output_path, "model.bin"))

    model.save_pretrained(output_path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    fire.Fire(train)
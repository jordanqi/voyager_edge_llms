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
# from loraprune.dataset import get_examples
from layerwrapper import BiasGPT, WrappedGPT


# from loraprune.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb
# import lm_eval


class TaylorImportance():
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction == 'second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_impwanda_sp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):

        group_imp = []

        idxs.sort()

        if prune_fn in ['attn_linear_out']:
            salience = {}
            for sub_layer in [layer.o_proj, layer.q_proj, layer.k_proj, layer.v_proj]:
                salience[sub_layer] = sub_layer.weight * sub_layer.weight.grad

                if self.taylor in ['param_second']:
                    salience[sub_layer] = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
                elif self.taylor in ['param_mix']:
                    salience[
                        sub_layer] = -salience + 0.5 * sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
        else:
            salience = layer.weight * layer.weight.grad

            if self.taylor in ['param_second']:
                salience = layer.weight * layer.weight.acc_grad * layer.weight
            elif self.taylor in ['param_mix']:
                salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight

        # Linear out_channels
        if prune_fn in ["linear_out"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(1).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(1)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        # Linear in_channels
        elif prune_fn in ["linear_in"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(0).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(0)
            else:
                raise NotImplementedError
            local_norm = local_norm[idxs]
            group_imp.append(local_norm)

        elif prune_fn in ['attn_linear_out']:
            local_norm = 0
            for sub_layer in [layer.o_proj]:  # linear out channel, first dim in linear.weight
                if self.taylor == 'vectorize':
                    local_norm += salience[sub_layer].sum(1).abs()
                elif 'param' in self.taylor:
                    local_norm += salience[sub_layer].abs().sum(1)
                else:
                    raise NotImplementedError

            for sub_layer in [layer.q_proj, layer.k_proj,
                              layer.v_proj]:  # linear in channel, second dim in linear.weight
                if self.taylor == 'vectorize':
                    local_norm += salience[sub_layer].sum(0).abs()
                elif 'param' in self.taylor:
                    local_norm += salience[sub_layer].abs().sum(0)
                else:
                    raise NotImplementedError
            group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        # if self.normalizer is not None:
        # group_imp = self.normalizer(group, group_imp)
        return group_imp


class RandomImportance():
    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):
        return torch.rand(len(idxs))


class MagnitudeImportance():
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):
        group_imp = []
        idxs.sort()
        # Linear out_channels
        if prune_fn in ['linear_out']:
            w = layer.weight.data[idxs].flatten(1)
            local_norm = w.abs().pow(self.p).sum(1)
            group_imp.append(local_norm)
        # Linear in_channels
        elif prune_fn in [
            'linear_in'
        ]:
            w = layer.weight
            local_norm = w.abs().pow(self.p).sum(0)
            local_norm = local_norm[idxs]
            group_imp.append(local_norm)
        # Attention
        elif prune_fn in ['attn_linear_out']:
            local_norm = 0
            for sub_layer in [layer.o_proj]:
                w_out = sub_layer.weight.data[idxs]
                local_norm += w_out.abs().pow(self.p).sum(1)

            for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                w_in = sub_layer.weight.data[:, idxs]
                local_norm += w_in.abs().pow(self.p).sum(0)
            group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        # if self.normalizer is not None:
        # group_imp = self.normalizer(group, group_imp)
        return group_imp


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


def get_mask(imps, layer, prune_fn, target_sparsity, head_dim=1):
    if prune_fn in ["linear_out"]:
        current_channels = layer.out_features
    elif prune_fn in ["linear_in"]:
        current_channels = layer.in_features
    else:
        current_channels = layer.out_features
    n_pruned = current_channels - int(
        current_channels *
        (1 - target_sparsity)
    )
    if n_pruned <= 0:
        return

    if head_dim > 1:
        imps = imps.view(-1, head_dim).sum(1)

    imp_argsort = torch.argsort(imps)

    if head_dim > 1:
        # n_pruned//consecutive_groups
        pruning_groups = imp_argsort[:(n_pruned // head_dim)]
        group_size = head_dim
        pruning_idxs = torch.cat(
            [torch.tensor([j + group_size * i for j in range(group_size)])
             for i in pruning_groups], 0)
    else:
        pruning_idxs = imp_argsort[:n_pruned]
    # print(len(pruning_idxs))
    return pruning_idxs


def apply_mask_idxs(layer, idxs, prune_fn):
    idxs.sort(reverse=True)
    before = layer.weight.numel()
    if prune_fn in ["linear_out"]:
        rows_mask = torch.ones(layer.weight.data.size(0), dtype=torch.bool)
        rows_mask[idxs] = False
        layer.weight.data = layer.weight.data[rows_mask]
        layer.out_features -= len(idxs)
    elif prune_fn in ["linear_in"]:
        cols_mask = torch.ones(layer.weight.data.size(1), dtype=torch.bool)
        cols_mask[idxs] = False
        layer.weight.data = layer.weight.data[:, cols_mask]
        layer.in_features -= len(idxs)
    return 1 - layer.weight.numel() / before


def apply_mask_tensor(layer, mask, prune_fn):
    mask = mask.bool()
    if prune_fn in ["linear_out"]:
        layer.weight.data = layer.weight.data[mask]
        layer.out_features = int(mask.sum())
    elif prune_fn in ["linear_in"]:
        layer.weight.data = layer.weight.data[:, mask]
        layer.in_features = int(mask.sum())


def prune_flap(model, tokenizer, nsamples, seqlen, seed, pruning_ratio, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
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

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], 'WIFV')

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        if i not in range(3, 31):
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps
            torch.cuda.empty_cache()
        else:
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            metrics = {
                'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
                'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
                    subset[name].weight.data.pow(2), dim=0),
                'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
            }

            for name in subset:
                if name == 'self_attn.o_proj':
                    W_metric = metrics["WIFV"](wrapped_layers, subset, name) ** 2
                    # if args.structure == "UL-UM":

                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    attn_metric_list.append(W_metric.cpu())
                    thresh = torch.sort(W_metric.cuda())[0][
                        math.ceil(pruning_ratio * layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric >= thresh)
                    attn_mask.append(W_mask)

                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                else:
                    W_metric = metrics["WIFV"](wrapped_layers, subset, name)

                    mlp_metric_list.append(W_metric.cpu())
                    thresh = torch.sort(W_metric.cuda())[0][math.ceil(W_metric.numel() * pruning_ratio)].cpu()
                    W_mask = (W_metric >= thresh)
                    mlp_mask.append(W_mask)
                    mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))

                wrapped_layers[name].free()
            inps, outs = outs, inps  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

    attn_mask = torch.stack(attn_mask)
    mlp_mask = torch.stack(mlp_mask)

    for idx in range(3, 31):
        compress(model.model.layers[idx], attn_mask[idx - 3], None,
                 attn_baseline_inp_list[idx - 3], None, device,
                 bias=False)
        compress(model.model.layers[idx], None, mlp_mask[idx - 3], None,
                 mlp_baseline_inp_list[idx - 3], device,
                 bias=False)


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
    if 'llama' in model_name or 'Qwen' in model_name:
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
                                          {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
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

        if i not in range(3, 27):
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps
            torch.cuda.empty_cache()
        else:
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1)))

                if name == layer_name['m1']:
                    W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)  # importance score of each head
                    thresh = torch.sort(W_metric.cuda())[0][math.ceil(pruning_ratio * layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric >= thresh)
                    compress(layer, W_mask, None, None, None, device, bias=False)
                else:
                    W_metric = W_metric.mean(axis=0)
                    thresh = torch.sort(W_metric.cuda())[0][math.ceil(W_metric.numel() * pruning_ratio)].cpu()
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
        attn_mask = attn_mask.repeat_interleave(128)

        # Prune the query projection weight (q_proj)
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[torch.where(attn_mask)[0]]

        # layer.self_attn.k_proj and layer.self_attn.v_proj remain unchanged

        # Update output dimensions of q projection based on the attn_mask
        layer.self_attn.q_proj.out_features = attn_mask.sum().item()

        output_weight = layer.self_attn.o_proj.weight.data

        # if bias:
        #     # Add the additional bias to compensate for the loss
        #     output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)

        # Prune the output projection weight (o_proj)
        output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]

        # Update layer configurations for the new output shape after pruning
        layer.self_attn.num_heads = retain_heads
        layer.self_attn.hidden_size = retain_heads * 128
        layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads

        # if bias:
        #     # Re-initialize the Linear layer with new shape and bias
        #     layer.self_attn.o_proj.in_features = attn_mask.sum().item()
        #     layer.self_attn.o_proj.bias.data = output_bias

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

    if prune_metric == 'random':
        imp = RandomImportance()
    elif prune_metric == 'l1':
        imp = MagnitudeImportance(p=1)
    elif prune_metric == 'l2':
        imp = MagnitudeImportance(p=2)
    elif prune_metric == 'taylor':
        imp = TaylorImportance(group_reduction='sum', taylor=taylor)
    elif prune_metric == 'wanda':
        imp = RandomImportance()
    elif prune_metric == 'wanda_sp':
        imp = RandomImportance()
    elif prune_metric == 'variance':
        imp = RandomImportance()
    elif prune_metric == 'attn_score':
        imp = RandomImportance()
    elif prune_metric == 'attn_score_global':
        imp = RandomImportance()
    if prune_metric in ['random', 'l1', 'l2', 'taylor']:
        model.to('cuda')
        example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to('cuda')

        if taylor in ['param_mix', 'param_second']:
            for j in range(10):
                batch_input = example_prompts[j].unsqueeze(0)
                loss = model(batch_input, labels=batch_input).loss
                loss.backward()

                for module_param in model.parameters():
                    if module_param.requires_grad:
                        module_param.grad = module_param.grad * module_param.grad / 10
                        if hasattr(module_param, 'acc_grad'):
                            module_param.acc_grad += module_param.grad
                        else:
                            module_param.acc_grad = copy.deepcopy(module_param.grad)
                model.zero_grad()
                del loss.grad
        loss = model(example_prompts, labels=example_prompts).loss
        loss.backward()

        for z in range(3, 31):
            layer = model.model.layers[z]
            imps = imp(layer.self_attn, "attn_linear_out", [])

            pruning_idxs = get_mask(imps, layer.self_attn.q_proj, "linear_out",
                                    ratio,
                                    layer.self_attn.head_dim)
            apply_mask_idxs(layer.self_attn.q_proj, pruning_idxs.tolist(), "linear_out")
            apply_mask_idxs(layer.self_attn.k_proj, pruning_idxs.tolist(), "linear_out")
            apply_mask_idxs(layer.self_attn.v_proj, pruning_idxs.tolist(), "linear_out")
            apply_mask_idxs(layer.self_attn.o_proj, pruning_idxs.tolist(), "linear_in")

            imps = imp(layer.mlp.gate_proj, "linear_out", [])
            pruning_idxs = get_mask(imps, layer.mlp.gate_proj, "linear_out",
                                    ratio)

            apply_mask_idxs(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out")
            apply_mask_idxs(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
            apply_mask_idxs(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
    # elif prune_metric in ['wanda']:
    #     model.to('cuda')
    #     prune_wanda(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))
    elif prune_metric in ['wanda_sp']:
        model.to('cuda')
        prune_wanda_sp(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))
    elif prune_metric in ['variance']:
        # for i in range(32):
        # model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
        #     torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device="cpu"))
        # model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
        #     torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device="cpu"))
        # torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        # torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        # model.seqlen = 2048
        model.to('cuda')
        prune_flap(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))
    # elif prune_metric in ['attn_score']:
    #     model.to('cuda')
    #     dict = prune_attn_score(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))
    # elif prune_metric in ['attn_score_global']:
    #     model.to('cuda')
    #     dict = prune_attn_score_modelwise(model, tokenizer, 2000, seq_len, seed, ratio, torch.device("cuda:0"))

    print(model)
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
                                                                        after_pruning_parameters,
                                                                        100.0 * after_pruning_parameters / before_pruning_parameters))
    gc.collect()
    torch.cuda.empty_cache()

    # ppl_after, result_after = utils.eval(model, tokenizer)
    # print(
    #     f"Current pruning ratio: {ratio}",
    #     " PPL after pruning: {}".format(ppl_after))

    if not os.path.exists(output_path):
        # 如果文件夹不存在，创建文件夹
        os.makedirs(output_path)

    torch.save({
        'model': model,
    }, os.path.join(output_path, "model.bin"))
    # if eval_task:
    #     with torch.no_grad():
    #         # merge_all(model)
    #         model.eval()
    #         model.half()
    #
    #         wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer, batch_size='auto')
    #         results = lm_eval.simple_evaluate(  # call simple_evaluate
    #             model=wrapped_model,
    #             tasks=["openbookqa", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "boolq"],
    #             # tasks=["openbookqa"],
    #             num_fewshot=0,
    #             log_samples=False,
    #         )
    #         if output_path:
    #             def _handle_non_serializable(o):
    #                 if isinstance(o, np.int64) or isinstance(o, np.int32):
    #                     return int(o)
    #                 elif isinstance(o, set):
    #                     return list(o)
    #                 else:
    #                     return str(o)
    #
    #             path = Path(output_path)
    #             # check if file or 'dir/results.json' exists
    #             if path.is_file():
    #                 raise FileExistsError(f"File already exists at {path}")
    #             output_path_file = path.joinpath("results_after_pruning.json")
    #             if path.suffix in (".json", ".jsonl"):
    #                 output_path_file = path
    #                 path.parent.mkdir(parents=True, exist_ok=True)
    #                 path = path.parent
    #             else:
    #                 path.mkdir(parents=True, exist_ok=True)
    #             dumped = json.dumps(
    #                 results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    #             )
    #             output_path_file.open("w", encoding="utf-8").write(dumped)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    fire.Fire(train)

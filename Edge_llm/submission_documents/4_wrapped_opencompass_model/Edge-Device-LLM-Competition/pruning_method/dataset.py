import os.path

import torch
from datasets import load_dataset
import json
import random
import tqdm

def get_c4(samples, cutoff_len, tokenizer):
    if os.path.exists("data/c4.json"):
        dataset = load_dataset("json", data_files="data/c4.json")
        if len(dataset) == samples:
            print("load c4 from".format("data/c4.json"))
            return dataset

    dataset = load_dataset('allenai/c4',  data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    print("Sampling {} data from c4".format(samples))
    subdata, history = [], []
    for _ in tqdm.tqdm(range(samples)):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > cutoff_len and i not in history:
                history.append(i)
                break
        subdata.append({"inputs": dataset[i]['text']})
    with open('data/c4.json', 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files="data/c4.json")

# def get_bookcorpus(samples, cutoff_len, tokenizer):
#     if os.path.exists("data/bookcorpus.json"):
#         dataset = load_dataset("json", data_files="data/bookcorpus.json")
#         if len(dataset) == samples:
#             print("load bookcorpus from".format("data/bookcorpus.json"))
#             return dataset
#
#     dataset = load_dataset('bookcorpus', split='train')
#     print("Sampling {} data from bookcorpus".format(samples))
#     #dataset = "".join(dataset['text'])
#     subdata, history = [], []
#     for _ in tqdm.tqdm(range(samples)):
#         stop = False
#         while not stop:
#             i = random.randint(0, len(dataset) - 2)
#             if i in history:
#                 continue
#             history.append(i)
#             current_text = dataset[i]['text']
#             sh = []
#             for j in range(i+1, len(dataset) - 1):
#                 sh.append(j)
#                 if j in history:
#                     break
#                 current_text += dataset[j]['text']
#                 trainenc = tokenizer(current_text, return_tensors='pt')
#                 if trainenc.input_ids.shape[1] > cutoff_len:
#                     stop = True
#                     history.extend(sh)
#                     break
#         subdata.append({"inputs": current_text})
#     with open('data/bookcorpus.json', 'w') as f:
#         f.writelines(json.dumps(subdata))
#     return load_dataset("json", data_files="data/bookcorpus.json")

get_dataset = {'c4': get_c4,}
               # 'bookcorpus': get_bookcorpus}


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
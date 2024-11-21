import os
import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import random
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate teacher logits from C4 dataset.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the teacher model.")
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save teacher logits.")
parser.add_argument("--nsamples", type=int, default=5, help="Number of samples to process.")
parser.add_argument("--seqlen", type=int, default=512, help="Maximum sequence length.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
args = parser.parse_args()

# Define function to load and preprocess samples from C4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)  # Set random seed for reproducibility
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    train_samples = []
    for example in traindata:
        text = example['text']
        tokens = tokenizer.encode(text)  # Tokenize the text
        if len(tokens) >= seqlen:
            train_samples.append(text)
        if len(train_samples) >= nsamples:  # Stop when the desired number of samples is reached
            break
    return train_samples

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load teacher model with specified precision
teacher_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16
).to(device)
teacher_model.eval()  # Set model to evaluation mode
teacher_model.config.use_cache = False  # Disable caching to save memory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load samples from the C4 dataset
train_samples = get_c4(nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
print(f"Loaded {len(train_samples)} samples.")  # Print the number of loaded samples

# Ensure save directory exists
os.makedirs(args.save_dir, exist_ok=True)

# Generate and save teacher logits
print("Generating teacher logits and saving them...")
for idx, sample in enumerate(tqdm(train_samples, desc="Processing samples")):
    tokenized_sample = tokenizer(
        sample,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.seqlen
    )
    input_ids = tokenized_sample['input_ids'].to(device)
    attention_mask = tokenized_sample['attention_mask'].to(device)

    # Forward pass to compute logits
    with torch.no_grad():
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu()

    # Save input_ids and logits
    teacher_logits = {
        'input_ids': input_ids.cpu().squeeze(0),
        'logits': logits.squeeze(0),
    }

    batch_file = os.path.join(args.save_dir, f"teacher_logits_{idx}.pkl")
    with open(batch_file, "wb") as f:
        pickle.dump(teacher_logits, f)
    print(f"Saved {batch_file}")

    # Clear memory
    del input_ids, attention_mask, outputs, logits, teacher_logits, tokenized_sample
    torch.cuda.empty_cache()

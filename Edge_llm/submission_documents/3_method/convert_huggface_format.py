from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prepare and save a model in Hugging Face format.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the input model directory.")
parser.add_argument("--save_dir", type=str, required=True, help="Path to the output directory to save the model.")
args = parser.parse_args()

# Load the model configuration
config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

# Load the model, ignoring size mismatch errors
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True
)

# Check and add the `lm_head` layer if it is missing
if not hasattr(model, 'lm_head'):
    print("Adding missing lm_head layer.")
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False).to(torch.bfloat16)

# Save the model in the specified directory in Hugging Face format
model.save_pretrained(args.save_dir, safe_serialization=True)

print(f"Model saved in Hugging Face format with bfloat16 precision to: {args.save_dir}")

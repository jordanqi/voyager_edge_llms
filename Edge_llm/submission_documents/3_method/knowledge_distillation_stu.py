import os
import torch
import torch.nn.functional as F
import pickle
from transformers import AutoTokenizer, Adafactor
from tqdm import tqdm
import argparse
from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Distill a student model using teacher logits.")
parser.add_argument("--student_model_path", type=str, required=True, help="Path to the student model checkpoint.")
parser.add_argument("--teacher_logits_dir", type=str, required=True, help="Directory containing teacher logits.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the distilled student model.")
args = parser.parse_args()

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Define configuration parameters
config = {
    "distillation": {
        "temperature": 2.0,  # Temperature parameter
        "alpha": 1.0,  # Weight of distillation loss vs original loss
    },
    "tokenizer": {
        "max_length": 1024,  # Sequence length limit
    },
    "training": {
        "num_epochs": 1,  # Number of training epochs
        "batch_size": 2,  # Batch size
        "learning_rate": 1e-5,  # Learning rate
        "accumulation_steps": 4,  # Gradient accumulation steps
    }
}

# Pad logits to align student and teacher logits
def pad_logits(student_logits, teacher_logits):
    max_length = max(student_logits.size(1), teacher_logits.size(1))
    if student_logits.size(1) < max_length:
        padding = torch.zeros(student_logits.size(0), max_length - student_logits.size(1), student_logits.size(2)).to(
            student_logits.device)
        student_logits = torch.cat([student_logits, padding], dim=1)
    if teacher_logits.size(1) < max_length:
        padding = torch.zeros(teacher_logits.size(0), max_length - teacher_logits.size(1), teacher_logits.size(2)).to(
            teacher_logits.device)
        teacher_logits = torch.cat([teacher_logits, padding], dim=1)
    return student_logits, teacher_logits

# Compute the distillation loss, combining KLD and original loss
def distillation_loss(student_logits, teacher_logits, inputs, original_loss):
    student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
    temperature = config["distillation"]["temperature"]
    alpha = config["distillation"]["alpha"]
    max_length = config["tokenizer"]["max_length"]
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    loss_kd = F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),
        F.softmax(teacher_logits_scaled, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2) / max_length
    return alpha * loss_kd + (1 - alpha) * original_loss

# Load student model checkpoint
checkpoint = torch.load(args.student_model_path, map_location=device)
student_model = checkpoint['model'].to(device)  # Extract model
student_model.train()  # Set model to training mode

# Enable gradient checkpointing to reduce memory consumption
student_model.gradient_checkpointing_enable()

# Make all model parameters trainable
for param in student_model.parameters():
    param.requires_grad = True

# Set optimizer
optimizer = Adafactor(
    student_model.parameters(),
    lr=config["training"]["learning_rate"],
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
)

# Training settings
num_epochs = config["training"]["num_epochs"]
accumulation_steps = config["training"]["accumulation_steps"]

# List all teacher logits files
teacher_logits_files = sorted(
    [f for f in os.listdir(args.teacher_logits_dir) if f.startswith("teacher_logits_")]
)

# Training loop
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}...")
    optimizer.zero_grad()

    # Show progress bar
    with tqdm(total=len(teacher_logits_files), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for idx, logits_file in enumerate(teacher_logits_files):
            # Load teacher logits
            with open(os.path.join(args.teacher_logits_dir, logits_file), "rb") as f:
                teacher_data = pickle.load(f)

            # Extract inputs and teacher logits
            inputs = teacher_data['input_ids'].unsqueeze(0).to(device)
            teacher_logits = teacher_data['logits'].unsqueeze(0).to(device)
            labels = inputs.clone()

            # Forward pass
            outputs = student_model(inputs, labels=labels)
            student_logits = outputs.logits
            original_loss = outputs.loss

            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_logits, inputs, original_loss) / accumulation_steps

            # Backward pass
            loss.backward()

            # Update parameters after accumulation
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(teacher_logits_files):
                optimizer.step()
                optimizer.zero_grad()

            # Clear memory
            torch.cuda.empty_cache()
            del inputs, labels, teacher_logits, outputs, student_logits, original_loss, loss

            # Update progress bar
            pbar.update(1)

    torch.cuda.empty_cache()

# Save the distilled student model
os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if it doesn't exist
torch.save({'model': student_model}, os.path.join(args.output_dir, "model.bin"))  # Save as model.bin
student_model.save_pretrained(args.output_dir)  # Save in Hugging Face format
print(f"Model saved to {args.output_dir}")

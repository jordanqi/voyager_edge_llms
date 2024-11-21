# 1. Environment Setup

Follow the commands below to create the environment and install the necessary dependencies:


```bash
conda create --name opencompass python=3.10 
conda activate opencompass
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install Faiss-gpu
cd opencompass && pip install -e .
cd opencompass/human-eval && pip install -e .
pip install -r requirements.txt
```

# 2. Dataset Download

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
```

### Unzip the dataset
```bash 
unzip OpenCompassData-core-20240207.zip
```

# 3. Model Download
From the 1_Saved_Model.txt file, download the model.bin files for the three models specified. Place these models in their respective folders according to the setup requirements.

# 4. Dataset Evaluation
The wrapped models and configuration files are already prepared in the 4_wrapped_opencompass_model directory. To evaluate the dataset, follow these instructions:

Place the model Files and configuration files under the folder(already done this step):
```bash
opencompass/opencompass/models

opencompass/configs
```
Ensure that the actual paths to the model.bin files are correctly updated in the configuration files before running the evaluation.
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --datasets commonsenseqa_gen bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen truthfulqa_gen --hf-num-gpus 1 --hf-type base --models custom_model_config --debug --model-kwargs device_map='auto' trust_remote_code=True
# --models: specify the local model
```

# 5. Throughput and Memory Evaluation
To evaluate inference memory and throughput, use the Evaluate.py script in the method folder. Ensure the paths to the model.bin files are updated to the actual paths.

```bash
#Run Evaluate.py for throughput and memory evaluation
python Evaluate.py --model_path /path/to/actual/model.bin
```

# 6. Methodology
### 6.1. Pruning (Oneshot Pruning)
The oneshot_prune method corresponds to the pruning of three models. The pruning rates for Qwen and LLaMA are pre-configured to prune 50% of the parameters. For Phi2, the pruning ratio can be manually adjusted to represent the pruning rate for practical applications.

```bash
# rune oneshot_prune to prune the model
python oneshot_prune_llama.py --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --prune_metric wanda_sp --seq_len 256 --output_path your_output_path
```


### 6.2. Knowledge Distillation
Teacher Logits Generation
Use knowledge_distillation_teacher.py to generate the logits of the teacher model via forward passes.

```bash
# Generate teacher logits
python knowledge_distillation_teacher.py --model_path /path/to/teacher_model --output_dir /path/to/output/logits
```

Student Model Distillation
Use knowledge_distillation_stu.py to load the generated logits and perform student model distillation training.

```bash
# Perform student distillation
python knowledge_distillation_stu.py --student_model_path /path/to/student_model --teacher_logits_dir /path/to/logits --output_dir /path/to/output/student_model
```

### 6.3. Format Conversion
The convert_huggface_format.py script converts saved models in safetensors format to the Hugging Face-compatible safetensors format.

```bash
## Convert model format
python convert_huggface_format.py --model_dir /path/to/safetensor --save_dir /path/to/hugging

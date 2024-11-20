# Edge-Device-LLM-Competition

**Environment setup**

```bash
  conda create --name opencompass python=3.10 
  conda activate opencompass
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install Faiss-gpu
  cd opencompass && pip install -e .
  cd opencompass/human-eval && pip install -e .
  pip install -r requirements.txt
```

**Model Pruning with `oneshot_prune_llama.py**

use the specified `oenshot_prun_.py` script to prune a specified model using various input parameters.
  ```bash
  python oenshot_prun_.py --base_model <BASE_MODEL> --ratio <RATIO> --prune_metric wanda_sp --seq_len <SEQ_LEN> --output_path <OUTPUT_PATH>
  ```

**Model Configuration**

For each model used in OpenCompass, the corresponding model_opencompass.py and config files are located in their respective directories

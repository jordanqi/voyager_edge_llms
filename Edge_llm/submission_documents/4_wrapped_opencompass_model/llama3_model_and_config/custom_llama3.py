from opencompass.models import CustomLlama3

models = [
    dict(
        abbr='llama-3-8b-hf',
        type=CustomLlama3,
        path='path to your llama model.bin',
        tokenizer_path='path to your llama3 local_tokenizer',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

from opencompass.models import CustomPhi2
models = [
    dict(
        type=CustomPhi2,
        abbr='phi-2-hf',
        path='path to the model',
        tokenizer_path='microsoft/phi-2',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            config_path='microsoft/phi-2',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        min_out_len=3,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

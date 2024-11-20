from opencompass.models import CustomQwen

models = [
    dict(
        type=CustomQwen,
        abbr='qwen-7b-hf',
        path='path to the model',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

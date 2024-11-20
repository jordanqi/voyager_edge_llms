from opencompass.models import CustomQwen

models = [
    dict(
        type=CustomQwen,
        abbr='qwen2-7b-hf',
        path='/home/zqin3/EdgeDeviceLLMCompetition-Starting-Kit/outputs/qwen_final',
        max_out_len=100,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
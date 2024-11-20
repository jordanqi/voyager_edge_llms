from opencompass.models import CustomLlama3

models = [
    dict(
        abbr='llama-3-8b-hf',
        type=CustomLlama3,
        path='/home/zqin3/EdgeDeviceLLMCompetition-Starting-Kit/llama3_0/model.bin',
        tokenizer_path='/home/zqin3/EdgeDeviceLLMCompetition-Starting-Kit/llama3/local_tokenizer',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

from opencompass.models import CustomLlama3

models = [
    dict(
        abbr='llama-3-8b-instruct-hf',
        type=CustomLlama3,
        path='/home/zqin3/EdgeDeviceLLMCompetition-Starting-Kit/outputs/qwen_mlp75_att25_7-20_lora_1epoch_32bs_sl1024_ns128/model.bin',
        tokenizer_path='/home/zqin3/EdgeDeviceLLMCompetition-Starting-Kit/outputs/llama3I_20',
        max_out_len=100,
        max_seq_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    ),
]


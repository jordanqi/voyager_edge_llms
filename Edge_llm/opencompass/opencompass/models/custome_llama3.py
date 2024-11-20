from typing import List, Optional

import torch
from transformers import AutoTokenizer

#from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.registry import MODELS

from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel
from opencompass.models.huggingface_above_v4_33 import _set_model_kwargs_torch_dtype
import dill


@MODELS.register_module()
# # 正常的方法
class CustomLlama3(HuggingFaceBaseModel):
    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM


        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)

        self.model = torch.load(path)
        self.model = self.model["model"]

        self.logger.debug(f'using model_kwargs: {model_kwargs}')

        self.model.eval()
        self.model.generation_config.do_sample = False

# class CustomLlama3(HuggingFaceBaseModel):
#     def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
#         DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
#         model_kwargs = DEFAULT_MODEL_KWARGS
#         model_kwargs.update(kwargs)
#
#         # 使用 dill.load 代替 torch.load
#         with open(path, 'rb') as f:
#             model_data = dill.load(f)
#
#         # 从 dill 加载的数据中获取模型和分词器
#         self.model = model_data['model']
#         self.tokenizer = model_data['tokenizer']
#
#         self.logger.debug(f'using model_kwargs: {model_kwargs}')
#
#         self.model.eval()
#         self.model.generation_config.do_sample = False
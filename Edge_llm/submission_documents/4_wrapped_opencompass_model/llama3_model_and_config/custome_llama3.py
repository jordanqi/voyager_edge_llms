from typing import List, Optional

import torch
from transformers import AutoTokenizer

#from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.registry import MODELS

from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel
from opencompass.models.huggingface_above_v4_33 import _set_model_kwargs_torch_dtype


@MODELS.register_module()
class CustomLlama3(HuggingFaceBaseModel):
    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM


        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        #model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)

        #self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = torch.load(path)
        self.model = self.model["model"]

        self.logger.debug(f'using model_kwargs: {model_kwargs}')

        # try:
        #     self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        # except ValueError:
        #     self.model = AutoModel.from_pretrained(path, **model_kwargs)

        # if peft_path is not None:
        #     from peft import PeftModel
        #     peft_kwargs['is_trainable'] = False
        #     self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False


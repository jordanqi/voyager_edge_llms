from typing import List, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoConfig

from opencompass.registry import MODELS

from opencompass.models.huggingface import HuggingFaceCausalLM

@MODELS.register_module()
class CustomQwen(HuggingFaceCausalLM):
    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         hf_cache_dir=hf_cache_dir,
                         max_seq_len=max_seq_len,
                         tokenizer_path=tokenizer_path,
                         tokenizer_kwargs=tokenizer_kwargs,
                         peft_path=peft_path,
                         tokenizer_only=tokenizer_only,
                         model_kwargs=model_kwargs,
                         generation_kwargs=generation_kwargs,
                         meta_template=meta_template,
                         extract_pred_after_decode=extract_pred_after_decode,
                         batch_padding=batch_padding,
                         pad_token_id=pad_token_id,
                         mode=mode,
                         use_fastchat_template=use_fastchat_template,
                         end_str=end_str)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        # self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = torch.load(path + '/model.bin')
        self.model = self.model["model"]

        # if peft_path is not None:
        #     from peft import PeftModel
        #     self.model = PeftModel.from_pretrained(self.model,
        #                                            peft_path,
        #                                            is_trainable=False)
        self.model.eval()
        # print(model_kwargs["config_path"])
        # self.model.config = AutoConfig.from_pretrained(model_kwargs["config_path"])
        # self.model.generation_config = AutoConfig.from_pretrained(model_kwargs["config_path"])
        self.model.generation_config.do_sample = False

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        min_out_len=min_out_len,
                                        stopping_criteria=stopping_criteria,
                                        **generation_kwargs)
        else:
            return sum(
                (self._single_generate(inputs=[input_],
                                       max_out_len=max_out_len,
                                       min_out_len=min_out_len,
                                       stopping_criteria=stopping_criteria,
                                       **generation_kwargs)
                 for input_ in inputs), [])
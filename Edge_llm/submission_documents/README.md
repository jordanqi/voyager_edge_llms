
- 1_Saved_Model.txt: It contains a shared link for downloading your model checkpoints in the huggingface format (make sure that the saved model can be downloaded via this shared link).

- 2_mlc_model: It contains a shared link for downloading the compiled model (compiled by MLC-MiniCPM) **The compiled model includes** the following files necessary for running on the Android platform: .apk, mlc-chat-config.json, ndarray-cache.json, params_shard_x.bin, tokenizer.json, tokenizer.model, and tokenizer_config.json.

- 3_method: Include the runnable source code of the method as well as a readme for usage explanation.

- 4_wrapped_opencompass_model: The (wrapped) model definition file (.py) and its configuration file which are required by opencompass for evaluating your local model. 

- 5_evaluation_result: .csv files for evaluation of compressed models.

P.S. All models (checkpoints, weights..) named llama3 were actually llama3.1 instruct required by the competition.

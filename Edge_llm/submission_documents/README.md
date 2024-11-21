
- 1.A .txt file: It contains a shared link for downloading your model checkpoints in the huggingface format (make sure that the saved model can be downloaded via this shared link).

- 2.A .txt file: It contains a shared link for downloading the compiled model (compiled by MLC-MiniCPM) (make sure that the compiled model can be downloaded via this shared link). **The compiled model should include** the following files necessary for running on the Android platform: .apk, mlc-chat-config.json, ndarray-cache.json, params_shard_x.bin, tokenizer.json, tokenizer.model, and tokenizer_config.json.

- 3.A folder: Include the runnable source code of your method as well as a readme for usage explanation.

- 4.The (wrapped) model definition file (.py) and its configuration file which are required by opencompass for evaluating your local model. 

- 5.A CSV file: All participating teams are required to evaluate their models locally first and submit the results using a .CSV file. It should contain scores of CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA, Throughput, and GPU memory usage. Please generate .CSV file via Generate_CSV.py

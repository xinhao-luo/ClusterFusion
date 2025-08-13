MODEL_PATH=/mnt/model/llama-2-7b
TOKENIZER_PATH=/mnt/model/llama-2-7b-hf/tokenizer.model

ref:
	torchrun --nproc_per_node 1 text_completion_llama.py \
	--ckpt_dir $(MODEL_PATH) \
	--tokenizer_path $(TOKENIZER_PATH) \
	--max_seq_len 1024 --max_batch_size 1

clusterfusion: 
	USE_CLUSTER_FUSION=true torchrun --nproc_per_node 1 text_completion_llama.py \
	--ckpt_dir $(MODEL_PATH) \
	--tokenizer_path $(TOKENIZER_PATH) \
	--max_seq_len 1024 --max_batch_size 1

.PHONY: ref clusterfusion
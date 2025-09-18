MODEL_PATH=/cephfs/shared/model/llama-2-7b-chat
TOKENIZER_PATH=/cephfs/shared/model/llama-2-7b-hf/tokenizer.model

ref:
	torchrun --nproc_per_node 1 chat.py \
	--ckpt_dir $(MODEL_PATH) \
	--tokenizer_path $(TOKENIZER_PATH) \
	--max_seq_len 1024 --max_batch_size 1 \
	--max_gen_len 1024

clusterfusion: 
	USE_CLUSTER_FUSION=true torchrun --nproc_per_node 1 chat.py \
	--ckpt_dir $(MODEL_PATH) \
	--tokenizer_path $(TOKENIZER_PATH) \
	--max_seq_len 1024 --max_batch_size 1 \
	--max_gen_len 1024

.PHONY: ref clusterfusion
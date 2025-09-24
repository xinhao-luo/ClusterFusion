# chat.py
import fire
import time
from typing import List
from llama import Llama
import torch

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    sys.stdout = old_stdout
    prompts: List[str] = ["I believe the meaning of life is"]
    prompt_tokens_list = [generator.tokenizer.encode(p, bos=True, eos=False) for p in prompts]

    print("prompt:", prompts[0])
    print("response: ", end="", flush=True)

    start_time = time.perf_counter()
    buffer_tokens = []
    token_str_prev = ""
    for token_ids in generator.stream_generate(prompt_tokens_list, max_gen_len, temperature, top_p):
        buffer_tokens.append(token_ids[0])
        token_str = generator.tokenizer.decode(buffer_tokens)
        new_text = token_str[len(token_str_prev):]
        print(new_text, end="", flush=True)
        token_str_prev = token_str


    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = sum(len(generator.tokenizer.encode(p, bos=False, eos=False)) for p in prompts) + max_gen_len
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    print("\n" + "="*50)
    print(f"Total completion time: {total_time:.3f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens/sec: {tokens_per_second:.2f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import sys

    class DummyFile:
        def write(self, x): pass
        def flush(self): pass

    old_stdout = sys.stdout
    sys.stdout = DummyFile()

    fire.Fire(main)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
# Launch server with
#python3 -m sglang.launch_server \
#  --model /mnt/model/llama-2-7b-hf \
#  --max-running-requests 1 \
#  --max-total-tokens 1024 \
#  --attention-backend clusterfusion \
#  --tp 1 \
#  --context-length 4096 \
#  --stream-output \
#  --port 5338

import requests
import json

port = 5338
prompt = "I believe the meaning of life is"
print(f"prompt: {prompt}")
print("Generated text: ", end="")

generated = ""
completion_tokens = 0
latency = None

for line in requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": prompt,
        "sampling_params": {
            "temperature": 0.9,
            "max_new_tokens": 1024,
        },
        "stream": True,
    },
    stream=True,
).iter_lines(decode_unicode=True):
    if line and line.startswith("data: "):
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        obj = json.loads(data)
        new_text = obj.get("text", "")
        meta_info = obj.get("meta_info", {})
        if len(new_text) > len(generated):
            print(f"{new_text[len(generated):]}", end="", flush=True)
            generated = new_text
        completion_tokens = meta_info.get("completion_tokens", completion_tokens)
        latency = meta_info.get("e2e_latency", latency)

print()
print("===========================================")
print(f"Tokens generated: {completion_tokens}")
print(f"Latency: {latency:.2f}s")
if completion_tokens and latency:
    print(f"Throughput: {completion_tokens / latency:.2f} tokens/sec")
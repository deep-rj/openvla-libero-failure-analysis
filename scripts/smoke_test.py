import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

torch.cuda.reset_peak_memory_stats()
print("Loading processor and model...")
t0 = time.time()
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")
print(f"Loaded in {time.time() - t0:.1f}s")

load_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory after load: {load_peak:.2f} GB")

dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the red block?\nOut:"

torch.cuda.reset_peak_memory_stats()
inputs = processor(prompt, dummy_image).to("cuda:0", dtype=torch.bfloat16)

t0 = time.time()
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
inference_time = time.time() - t0

inference_peak = torch.cuda.max_memory_allocated() / 1e9

print(f"\nAction output: {action}")
print(f"Action shape: {getattr(action, 'shape', len(action))}")
print(f"Inference time: {inference_time:.3f}s")
print(f"Peak memory during inference: {inference_peak:.2f} GB")

# Add a second call right after the first, reusing the same inputs
t0 = time.time()
action2 = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

inference_peak = torch.cuda.max_memory_allocated() / 1e9

print(f"Second inference time: {time.time() - t0:.3f}s")
print(f"Peak memory during second inference: {inference_peak:.2f} GB")

print(vla.config.text_config.num_hidden_layers)
print(vla.config.text_config.hidden_size)
print(vla.config.text_config.num_attention_heads)
print(vla.config.text_config.num_key_value_heads)

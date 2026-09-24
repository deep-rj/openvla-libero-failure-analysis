import json

from huggingface_hub import hf_hub_download

CHECKPOINT = "openvla/openvla-7b-finetuned-libero-10"

with open(hf_hub_download(CHECKPOINT, "config.json")) as f:
    d = json.load(f)
# print(list(d.keys()))
print(list(d["text_config"].keys()))

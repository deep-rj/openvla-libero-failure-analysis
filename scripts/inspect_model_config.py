import json

d = json.load(
    open(
        "/workspace/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/config.json"
    )
)
# print(list(d.keys()))
print(list(d["text_config"].keys()))

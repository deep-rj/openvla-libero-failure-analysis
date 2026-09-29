"""OpenVLA policy with per-position logit capture.

Importing this module pulls in torch, TensorFlow and the sibling `openvla` package.
`predict` mirrors the vendored `get_vla_action` + `predict_action`, replacing the final
`generate` call with one that returns logits; `reference_action` is the vendored path.
"""

import re
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import torch
from experiments.robot.openvla_utils import (
    DEVICE,
    crop_and_resize,
    get_processor,
    get_vla,
    get_vla_action,
)
from PIL import Image

from harness import guards
from harness.policy import summarize_logits
from harness.schema import ModelOutput

# Appended to the prompt when absent, as `predict_action` does.
EMPTY_TOKEN_ID = 29871
_HUB_MODULE = re.compile(r"^transformers_modules\.[^.]+\.[^.]+\.([0-9a-f]{40})\.")


@dataclass
class PolicyConfig:
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    unnorm_key: str = "libero_10"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def hub_commit_from_module(module_name: str) -> str | None:
    match = _HUB_MODULE.match(module_name)
    return match.group(1) if match else None


def _prompt(task_label: str) -> str:
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def build_inputs(processor, img, task_label, center_crop, device=DEVICE):
    """Mirrors the preprocessing half of `get_vla_action`."""
    image = Image.fromarray(img).convert("RGB")
    if center_crop:
        tensor = tf.convert_to_tensor(np.array(image))
        orig_dtype = tensor.dtype
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        tensor = crop_and_resize(tensor, 0.9, 1)
        tensor = tf.clip_by_value(tensor, 0, 1)
        tensor = tf.image.convert_image_dtype(tensor, orig_dtype, saturate=True)
        image = Image.fromarray(tensor.numpy()).convert("RGB")
    return processor(_prompt(task_label), image).to(device, dtype=torch.bfloat16)


class OpenVLAPolicy:
    def __init__(self, vla, processor, cfg: PolicyConfig, device=DEVICE):
        self.vla = vla
        self.processor = processor
        self.cfg = cfg
        self.device = device

    @classmethod
    def load(cls, cfg: PolicyConfig) -> "OpenVLAPolicy":
        """Loads through the vendored helpers, then hard-fails if the executed code differs."""
        vla, processor = get_vla(cfg), get_processor(cfg)
        guards.verify_loaded_model(vla, processor, cfg.unnorm_key, vla.config.norm_stats)
        return cls(vla, processor, cfg)

    @property
    def action_stats(self) -> dict[str, np.ndarray]:
        stats = self.vla.get_action_stats(self.cfg.unnorm_key)
        q01 = np.array(stats["q01"])
        return {
            "q01": q01,
            "q99": np.array(stats["q99"]),
            "mask": np.array(stats.get("mask", np.ones_like(q01, dtype=bool))),
        }

    def provenance(self) -> dict:
        module = type(self.vla).__module__
        return {
            "model_class_module": module,
            "hub_code_commit": hub_commit_from_module(module),
            "checkpoint_commit": getattr(self.vla.config, "_commit_hash", None),
        }

    def predict(self, img: np.ndarray, task_label: str) -> ModelOutput:
        inputs = build_inputs(self.processor, img, task_label, self.cfg.center_crop, self.device)
        input_ids = inputs.pop("input_ids")
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
            empty = torch.tensor([[EMPTY_TOKEN_ID]], dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat((input_ids, empty), dim=1)

        action_dim = self.vla.get_action_dim(self.cfg.unnorm_key)
        _sync()
        t0 = time.perf_counter()
        out = self.vla.generate(
            input_ids,
            max_new_tokens=action_dim,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            **inputs,
        )
        _sync()
        latency_ms = (time.perf_counter() - t0) * 1e3

        logits = torch.stack(out.logits)[:, 0, :].float().cpu().numpy()
        chosen = out.sequences[0, -action_dim:].cpu().numpy()
        return summarize_logits(logits, chosen, latency_ms)

    def reference_action(self, img: np.ndarray, task_label: str) -> np.ndarray:
        """The vendored end-to-end path: unnormalized action, gripper not yet post-processed."""
        return get_vla_action(
            self.vla,
            self.processor,
            self.cfg.pretrained_checkpoint,
            {"full_image": img},
            task_label,
            self.cfg.unnorm_key,
            center_crop=self.cfg.center_crop,
        )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

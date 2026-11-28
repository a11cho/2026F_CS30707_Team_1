import gc
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .state import ACTION_COUNT, CANVAS, JOINT_ACTIONS


BASE_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


@dataclass(frozen=True)
class DecisionInput:
    board: np.ndarray
    status: np.ndarray
    mask: np.ndarray
    tokens: list[int] | None


class JointActionHead(nn.Module):
    def __init__(self, features, *, additive_only=False):
        super().__init__()
        self.unary = nn.Linear(features, 3 * ACTION_COUNT)
        self.pairs = nn.Linear(features, 3 * ACTION_COUNT**2) if not additive_only else None
        self.residual = nn.Linear(features, len(JOINT_ACTIONS)) if not additive_only else None
        self.register_buffer("slots", torch.from_numpy(JOINT_ACTIONS.T.copy()), persistent=False)
        pairs = np.stack([ACTION_COUNT * JOINT_ACTIONS[:, i] + JOINT_ACTIONS[:, j]
                          for i, j in combinations(range(3), 2)])
        self.register_buffer("pair_indices", torch.from_numpy(pairs), persistent=False)
        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.003)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        unary = self.unary(features).reshape(-1, 3, ACTION_COUNT)
        scores = unary.gather(2, self.slots.expand(len(features), -1, -1)).sum(dim=1)
        if self.pairs is not None:
            pairs = self.pairs(features).reshape(-1, 3, ACTION_COUNT**2)
            scores = scores + pairs.gather(2, self.pair_indices.expand(len(features), -1, -1)).sum(dim=1)
            scores = scores + self.residual(features)
        return scores


class TeamPolicy(nn.Module):
    def __init__(self, backbone, device, *, base=None, additive_only=False, rules_prompt=False,
                 rules_prompt_version=1, explicit_map_prompt=False):
        super().__init__()
        if backbone not in ("cnn", "bf16", "nf4"):
            raise ValueError(f"unknown backbone {backbone}")
        self.device = torch.device(device)
        self.backbone = backbone
        self.additive_only = additive_only
        self.rules_prompt = rules_prompt
        self.rules_prompt_version = rules_prompt_version
        self.explicit_map_prompt = explicit_map_prompt
        self.spatial = nn.Sequential(nn.Conv2d(9, 24, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.spatial_projection = nn.Sequential(nn.Linear(16 * CANVAS[0] * CANVAS[1] + 19, 128), nn.Tanh())
        hidden_size = 128 if backbone == "cnn" else 384
        self.action_head = JointActionHead(hidden_size, additive_only=additive_only)
        self.value_head = nn.Linear(hidden_size, 1)
        self.to(self.device)
        self.text_encoder = None
        self.tokenizer = None
        if backbone != "cnn":
            self.tokenizer, self.text_encoder = load_qwen(base, backbone, self.device)
            text_size = self.text_encoder.config.hidden_size
            self.text_projection = nn.Sequential(nn.Linear(text_size, 256), nn.Tanh()).to(self.device)

    def prepare(self, views: Sequence):
        boards_and_status = [state.spatial_input() for state, _, _ in views]
        if self.tokenizer is None:
            tokens = [None] * len(views)
        else:
            text = [state.text_input(history, rules_prompt=self.rules_prompt, rules_prompt_version=self.rules_prompt_version,
                                     explicit_map_prompt=self.explicit_map_prompt)
                    for state, history, _ in views]
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [DecisionInput(board, status, mask, ids)
                for (board, status), (_, _, mask), ids in zip(boards_and_status, views, tokens, strict=True)]

    def text_hidden(self, batch):
        try:
            return self._text_hidden(batch)
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or torch.is_grad_enabled() or len(batch) == 1:
                raise
        torch.cuda.empty_cache()
        middle = len(batch) // 2
        return torch.cat((self.text_hidden(batch[:middle]), self.text_hidden(batch[middle:])))

    def _text_hidden(self, batch):
        tokens = self.tokenizer.pad({"input_ids": [sample.tokens for sample in batch]}, return_tensors="pt")
        tokens = tokens.to(self.device)
        hidden = self.text_encoder(**tokens, use_cache=False).last_hidden_state
        last = tokens.attention_mask.sum(dim=-1) - 1
        return hidden[torch.arange(len(batch), device=self.device), last].float()

    def encode(self, batch, *, cached_text=None):
        boards = torch.from_numpy(np.stack([sample.board for sample in batch])).to(self.device)
        status = torch.from_numpy(np.stack([sample.status for sample in batch])).to(self.device)
        spatial = self.spatial_projection(torch.cat((self.spatial(boards), status), dim=-1))
        features = spatial
        if self.text_encoder is not None:
            text = self.text_hidden(batch) if cached_text is None else cached_text.to(self.device)
            features = torch.cat((self.text_projection(text), spatial), dim=-1)
        return features

    def forward(self, batch, *, cached_text=None):
        features = self.encode(batch, cached_text=cached_text)
        return self.action_head(features), self.value_head(features).squeeze(dim=-1)

    def _masked_distribution(self, batch, logits):
        masks = torch.from_numpy(np.stack([sample.mask for sample in batch])).to(self.device)
        return Categorical(logits=logits.masked_fill(~masks, -torch.inf))

    def actor(self, batch, *, cached_text=None):
        logits = self.action_head(self.encode(batch, cached_text=cached_text))
        return self._masked_distribution(batch, logits)

    def distribution(self, batch, *, cached_text=None):
        logits, value = self(batch, cached_text=cached_text)
        return self._masked_distribution(batch, logits), value

    def trainable_weights(self):
        return {name: param.detach().cpu().clone() for name, param in self.named_parameters() if param.requires_grad}

    @contextmanager
    def checkpoint_activations(self):
        if self.text_encoder is None:
            raise ValueError("activation checkpointing requires a Qwen text encoder")
        text = self.text_encoder.get_base_model()
        if text.config.attention_dropout:
            raise ValueError("activation recomputation requires a deterministic text policy")
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The input to the first LoRA block is frozen. Only the last eight blocks need a graph.
        text.disable_input_require_grads()
        layers = text.layers[-8:]
        for layer in layers:
            layer.train()
        try:
            yield
        finally:
            for layer in layers:
                layer.eval()
            text.gradient_checkpointing_disable()

    @torch.no_grad()
    def load_trainable_weights(self, weights):
        parameters = {name: param for name, param in self.named_parameters() if param.requires_grad}
        if set(parameters) != set(weights):
            raise ValueError("checkpoint parameters do not match the policy configuration")
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])

    @contextmanager
    def using_trainable_weights(self, weights):
        previous = self.trainable_weights()
        try:
            self.load_trainable_weights(weights)
            yield
        finally:
            self.load_trainable_weights(previous)


def load_qwen(base, backbone, device):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("the Qwen pilot requires a CUDA device with BF16 support")
    source = base or BASE_ID
    loading = {"local_files_only": False, "revision": BASE_REVISION}
    if base is not None:
        if not Path(base).is_dir():
            raise ValueError("--base must be a directory containing the pinned Qwen snapshot")
        loading = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="right", **loading)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = {}
    if backbone == "nf4":
        quantization["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    multimodal = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa", **loading, **quantization,
    )
    text = multimodal.model.language_model
    del multimodal
    gc.collect()
    if backbone == "nf4":
        text.is_loaded_in_4bit = True
        text = prepare_model_for_kbit_training(text, use_gradient_checkpointing=False)
    layer_count = len(text.layers)
    targets = [f"layers.{index}.mlp.{projection}" for index in range(layer_count - 8, layer_count)
               for projection in ("up_proj", "down_proj")]
    if len(set(targets) & set(dict(text.named_modules()))) != 16:
        raise ValueError("expected the up/down projections of Qwen's last eight text layers")
    return tokenizer, get_peft_model(text, LoraConfig(r=16, lora_alpha=32, target_modules=targets, bias="none"))

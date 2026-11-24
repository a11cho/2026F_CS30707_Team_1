"""A frozen Qwen prefix and three independent adapters for its last eight blocks."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


SLOTS = 3


class SlotLoRA(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.base = original.get_base_layer()
        self.a = nn.Parameter(original.lora_A["default"].weight.detach().repeat(SLOTS, 1, 1))
        self.b = nn.Parameter(original.lora_B["default"].weight.detach().repeat(SLOTS, 1, 1))
        self.scale = original.scaling["default"]

    def forward(self, features):
        base = self.base(features)
        by_slot = features.reshape(SLOTS, -1, features.shape[-1])
        low_rank = torch.bmm(by_slot, self.a.to(features.dtype).transpose(1, 2))
        delta = torch.bmm(low_rank, self.b.to(features.dtype).transpose(1, 2))
        return base + delta.reshape_as(base) * self.scale


class SlotQwen(nn.Module):
    def __init__(self, adapted):
        super().__init__()
        self.text = adapted.get_base_model()
        self.split = len(self.text.layers) - 8
        if self.text.config.attention_dropout:
            raise ValueError("slot adapter checkpointing requires deterministic attention")
        for layer in self.text.layers[self.split:]:
            layer.mlp.up_proj = SlotLoRA(layer.mlp.up_proj)
            layer.mlp.down_proj = SlotLoRA(layer.mlp.down_proj)

    def _layers(self, hidden, attention_mask, layers, *, rematerialize=False):
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

        positions = torch.arange(hidden.shape[1], device=hidden.device)
        text_positions = positions.expand(hidden.shape[0], -1)
        rotary_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
        causal_mask = create_causal_mask(config=self.text.config, inputs_embeds=hidden,
                                         attention_mask=attention_mask, cache_position=positions,
                                         past_key_values=None, position_ids=text_positions)
        linear_mask = self.text._update_linear_attn_mask(attention_mask, positions)
        position_embeddings = self.text.rotary_emb(hidden, rotary_positions)
        for layer in layers:
            mask = linear_mask if layer.layer_type == "linear_attention" else causal_mask
            arguments = dict(position_embeddings=position_embeddings, attention_mask=mask,
                             position_ids=rotary_positions, past_key_values=None, use_cache=False,
                             cache_position=positions)
            if rematerialize:
                hidden = checkpoint(layer, hidden, use_reentrant=False, preserve_rng_state=False, **arguments)
            else:
                hidden = layer(hidden, **arguments)
        return hidden

    @torch.no_grad()
    def prefix(self, input_ids, attention_mask):
        hidden = self.text.embed_tokens(input_ids)
        return self._layers(hidden, attention_mask, self.text.layers[:self.split])

    def forward(self, prefix, attention_mask):
        masks = attention_mask.repeat(SLOTS, 1)
        hidden = self._layers(prefix.repeat(SLOTS, 1, 1), masks, self.text.layers[self.split:],
                              rematerialize=torch.is_grad_enabled())
        rows = torch.arange(len(hidden), device=hidden.device)
        last = masks.sum(dim=1) - 1
        pooled = self.text.norm(hidden[rows, last]).float()
        return pooled.reshape(SLOTS, len(prefix), -1).transpose(0, 1)

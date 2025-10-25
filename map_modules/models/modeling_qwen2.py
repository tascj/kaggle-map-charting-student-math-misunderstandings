# Modified from transformers==4.52.4

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from liger_kernel.transformers import (
    LigerSwiGLUMLP,
    LigerRMSNorm,
    liger_rotary_pos_emb,
    LigerFusedAddRMSNorm,
    LigerFusedLinearCrossEntropyLoss,
)


flex_attention = torch.compile(flex_attention)
Qwen2MLP = LigerSwiGLUMLP
Qwen2RMSNorm = LigerRMSNorm


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states,
        position_ids,
        block_mask,
        position_embeddings,
    ):
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(1, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(1, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(1, q_len, -1, self.head_dim).transpose(1, 2)

        # dropout_rate = 0.0 if not self.training else self.attention_dropout

        cos, sin = position_embeddings
        query_states, key_states = liger_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        if layer_idx == 0:
            self.input_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = LigerFusedAddRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LigerFusedAddRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = Qwen2RMSNorm(
        #     config.hidden_size, eps=config.rms_norm_eps
        # )

    def forward(
        self,
        hidden_states,
        position_ids,
        block_mask,
        position_embeddings,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states, residual = hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
            block_mask,
            position_embeddings,
        )
        # hidden_states = residual + hidden_states
        # # Fully Connected
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids,
        position_ids,
        block_mask,
    ):
        input_ids = input_ids.squeeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids.unsqueeze(0))

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = cp.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    position_ids,
                    block_mask,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    block_mask,
                    position_embeddings,
                )

            hidden_states = layer_outputs

        hidden_states = hidden_states[0] + hidden_states[1]  # hidden_states + residual
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=True)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids,
        position_ids,
        suffix_ids,
        doc_ids,
        last_tokens,
    ):
        def custom_mask(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx]) | (
                suffix_ids[kv_idx] == -1
            )
            same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
            return causal & same_suffix & same_doc

        block_mask = create_block_mask(
            custom_mask,
            B=None,
            H=None,
            Q_LEN=input_ids.size(0),
            KV_LEN=input_ids.size(0),
            BLOCK_SIZE=(128, 128),
        )
        hidden_states = self.model(
            input_ids,
            position_ids,
            block_mask,
        )

        hidden_states = hidden_states[last_tokens]
        logits = self.score(hidden_states)
        logits = logits.float()
        return logits


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.score = nn.Linear(config.hidden_size, 1, bias=True)
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids,
        position_ids,
        suffix_ids,
        doc_ids,
        last_tokens,
        suffix_masks,
        suffix_labels,
    ):
        def custom_mask(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx]) | (
                suffix_ids[kv_idx] == -1
            )
            same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
            return causal & same_suffix & same_doc

        block_mask = create_block_mask(
            custom_mask,
            B=None,
            H=None,
            Q_LEN=input_ids.size(0),
            KV_LEN=input_ids.size(0),
            BLOCK_SIZE=(128, 128),
        )
        hidden_states = self.model(
            input_ids,
            position_ids,
            block_mask,
        )
        logits = self.score(hidden_states[last_tokens])
        sft_loss = self.loss_fn(
            self.lm_head.weight, hidden_states[suffix_masks], suffix_labels
        )
        return logits, sft_loss

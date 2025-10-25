# Modified from transformers==4.52.4


import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from liger_kernel.transformers import (
    LigerRMSNorm,
    liger_rotary_pos_emb,
    LigerFusedLinearCrossEntropyLoss,
)
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from .w8a8_kernels import (
    per_token_quant_int8,
    matmul_kernel_dynamic_quant,
    rms_norm_dynamic_quant,
)

flex_attention = torch.compile(flex_attention)
Qwen3RMSNorm = LigerRMSNorm


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_scale=False):
        super().__init__()
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.int8)
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32)
        )
        if input_scale:
            self.register_buffer(
                "input_scale", torch.empty(in_features, dtype=torch.float32)
            )
        else:
            self.register_buffer("input_scale", None)
        self.eps = 1e-7

    def forward(self, x):
        if self.input_scale is not None:
            x = x / self.input_scale
            x_int8, x_scale = per_token_quant_int8(x, self.eps)
        else:
            x_int8, x_scale = x
        weight_int8, weight_scale = self.weight, self.weight_scale
        out = matmul_kernel_dynamic_quant(
            x_int8,
            weight_int8,
            x_scale,
            weight_scale,
            bias=self.bias,
            output_dtype=torch.bfloat16,
        )
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = W8A8Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = W8A8Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = W8A8Linear(
            self.intermediate_size, self.hidden_size, bias=False, input_scale=True
        )
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(
            LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
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

        self.q_proj = W8A8Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = W8A8Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = W8A8Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = W8A8Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            input_scale=True,
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states,
        position_ids,
        block_mask,
        position_embeddings,
    ):
        hidden_states_int8, hidden_states_scale = hidden_states
        q_len, _ = hidden_states_int8.size()

        query_states = self.q_proj((hidden_states_int8, hidden_states_scale))
        key_states = self.k_proj((hidden_states_int8, hidden_states_scale))
        value_states = self.v_proj((hidden_states_int8, hidden_states_scale))

        query_states = self.q_norm(
            query_states.view(1, q_len, -1, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            key_states.view(1, q_len, -1, self.head_dim)
        ).transpose(1, 2)
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


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        position_ids,
        block_mask,
        position_embeddings,
    ):
        residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states_int8, hidden_states_scale = rms_norm_dynamic_quant(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )

        # Self Attention
        hidden_states = self.self_attn(
            (hidden_states_int8, hidden_states_scale),
            position_ids,
            block_mask,
            position_embeddings,
        )
        # Fully Connected
        # hidden_states = residual + hidden_states
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        hidden_states_int8, hidden_states_scale, residual = rms_norm_dynamic_quant(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            residual=residual,
        )
        hidden_states = self.mlp((hidden_states_int8, hidden_states_scale))
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
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
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
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


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
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

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForSequenceClassification(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
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


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
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

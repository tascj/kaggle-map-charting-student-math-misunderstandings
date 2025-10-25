# ruff: noqa: E402
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True

from pathlib import Path


from detectron2.config import LazyConfig, instantiate

from map_modules.models.w8a8_kernels import per_channel_quant
from map_modules.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument(
        "--opts",
        help="""
Modify config options at the end of the command, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.ZERO_OR_MORE,
    )
    return parser.parse_args()


def setup(args):
    cfg = LazyConfig.load(args.config)
    # default work_dir
    cfg_path = Path(args.config)
    work_dir_root = Path(args.output_root)
    work_dir = str(work_dir_root / cfg_path.relative_to("configs/").with_suffix(""))
    cfg.train.work_dir = work_dir
    # override config
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger("MAP")
    logger.info("Start")

    return cfg


def convert_to_w8a8(linear_layer, scale, set_input_scale=False):
    weight = linear_layer.weight.mul_(scale.view(1, -1))
    weight_int8, weight_scale = per_channel_quant(weight, torch.int8)
    linear_layer.weight.data = weight_int8
    linear_layer.register_buffer("weight_scale", weight_scale)
    if set_input_scale:
        linear_layer.register_buffer("input_scale", scale)


def get_weight_scale(linear_layers):
    weight_scales = torch.stack(
        [linear_layer.weight.abs().max(dim=0)[0] for linear_layer in linear_layers]
    )
    return weight_scales.max(dim=0)[0]


@torch.no_grad()
def main():
    args = parse_args()
    cfg = setup(args)
    logger = get_logger("MAP")
    ALPHA = args.alpha

    work_dir = Path(cfg.train.work_dir)
    cfg.model.name_or_path = work_dir / "checkpoint_epoch_1"
    model = instantiate(cfg.model)
    model.requires_grad_(False)
    act_scales = torch.load(
        work_dir / "act_scales.pth", map_location="cuda", weights_only=True
    )
    for layer_idx, layer in enumerate(model.model.layers):
        logger.info(f"Processing layer {layer_idx}")
        # qkv_proj
        act_scale = act_scales[f"model.layers.{layer_idx}.self_attn.q_proj"]
        weight_scale = get_weight_scale(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        )
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        convert_to_w8a8(layer.self_attn.q_proj, scale)
        convert_to_w8a8(layer.self_attn.k_proj, scale)
        convert_to_w8a8(layer.self_attn.v_proj, scale)
        layer.input_layernorm.weight.div_(scale)

        # o_proj
        act_scale = act_scales[f"model.layers.{layer_idx}.self_attn.o_proj"]
        weight_scale = get_weight_scale([layer.self_attn.o_proj])
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        convert_to_w8a8(layer.self_attn.o_proj, scale, set_input_scale=True)

        # gate_proj, up_proj
        if hasattr(layer.mlp, "gate_up_proj"):
            act_scale = act_scales[f"model.layers.{layer_idx}.mlp.gate_up_proj"]
            gate_up_proj = [layer.mlp.gate_up_proj]
        else:
            act_scale = act_scales[f"model.layers.{layer_idx}.mlp.gate_proj"]
            gate_up_proj = [layer.mlp.gate_proj, layer.mlp.up_proj]
        weight_scale = get_weight_scale(gate_up_proj)
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        for gate_up_proj in gate_up_proj:
            convert_to_w8a8(gate_up_proj, scale)
        layer.post_attention_layernorm.weight.div_(scale)

        # down_proj
        act_scale = act_scales[f"model.layers.{layer_idx}.mlp.down_proj"]
        weight_scale = get_weight_scale([layer.mlp.down_proj])
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        convert_to_w8a8(layer.mlp.down_proj, scale, set_input_scale=True)

    model.save_pretrained(work_dir / "checkpoint_w8a8")


if __name__ == "__main__":
    main()

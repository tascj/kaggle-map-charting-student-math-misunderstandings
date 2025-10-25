# ruff: noqa: E402
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import functools
import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F

from pathlib import Path


from detectron2.config import LazyConfig, instantiate

from map_modules.logging import get_logger
from map_modules.utils import to_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", default=None, type=str)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument(
        "--opts",
        help="""
Modify config options at the end of the command, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.ZERO_OR_MORE,
    )
    parser.add_argument("--out", default=None, type=str)
    return parser.parse_args()


@torch.no_grad()
def do_test(cfg, model):
    logger = get_logger("MAP")
    logger.info("Evaluation start")

    val_loader = instantiate(cfg.dataloader.val)

    model.eval()
    from tqdm import tqdm

    prog_bar = tqdm(val_loader)
    probs = []
    for batch in prog_bar:
        batch = to_gpu(batch)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"],
                batch["position_ids"],
                batch["suffix_ids"],
                batch["doc_ids"],
                batch["last_tokens"],
            )
        logits = logits.float().flatten()
        losses = []
        for _logits, _label in zip(
            logits.split(batch["num_candidates"]), batch["label"]
        ):
            losses.append(F.cross_entropy(_logits, _label))
            probs.append(_logits.float().softmax(dim=-1).data.cpu())
        loss = torch.stack(losses).mean()
        prog_bar.set_description(f"Loss: {loss.item()}")

    result = [prob.numpy() for prob in probs]

    logger.info("Evaluation prediction done")
    if not hasattr(val_loader.dataset, "evaluate"):
        eval_result = {"info": f"Not implemented for {type(val_loader.dataset)}"}
    else:
        eval_result = val_loader.dataset.evaluate(result)
    logger.info("Evaluation end")
    return result, eval_result


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


def clean_up():
    pass


def main():
    args = parse_args()
    cfg = setup(args)
    logger = get_logger("MAP")

    # Force fold == 0 for collect scales
    cfg.dataloader.val.dataset.query = "fold == 0"

    load_from = cfg.train.get("load_from", None)
    if args.load_from is not None:
        load_from = args.load_from
    if load_from is not None:
        logger.info(f"Load checkpoint: {load_from}")
        cfg.model.name_or_path = load_from
    model = instantiate(cfg.model)

    # smooth quant collect scales
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    result, eval_result = do_test(cfg, model)
    logger.info(f"Evaluation result: {eval_result}")

    for h in hooks:
        h.remove()

    torch.save(act_scales, os.path.join(cfg.train.work_dir, "act_scales.pth"))


if __name__ == "__main__":
    main()

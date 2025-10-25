# ruff: noqa: E402
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import time
import shutil
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F

from pathlib import Path


from detectron2.config import LazyConfig, instantiate
from detectron2.solver import LRMultiplier
from detectron2.engine.hooks import LRScheduler
from detectron2.utils.env import seed_all_rng

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


def do_train(cfg, model):
    if hasattr(cfg.optimizer["_target_"], "issue_h2d_transfer"):
        cfg.optimizer.model = model
    else:
        cfg.optimizer.params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    max_epochs = cfg.train.max_epochs
    lr_scheduler = LRMultiplier(
        optimizer,
        multiplier=instantiate(cfg.lr_multiplier),
        max_iter=max_epochs * len(train_loader),
    )
    best_param_group_id = LRScheduler.get_best_param_group_id(optimizer)

    logger = get_logger("MAP")
    total_updates = 0

    for curr_epoch in range(max_epochs):
        model.train()
        for curr_iter, batch in enumerate(train_loader):
            batch = to_gpu(batch)
            batch_loss = 0

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
            loss = torch.stack(losses).mean()
            optimizer.ready_for_optimizer_step = True
            loss.backward()
            batch_loss += loss.detach()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_updates += 1
            lr_scheduler.step()
            if total_updates % cfg.train.log_interval == 0:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                loss_val = batch_loss.item()
                logger.info(
                    f"Epoch [{curr_epoch + 1}/{max_epochs}] Iter [{curr_iter + 1}/{len(train_loader)}]"
                    f" lr: {lr:.4e}, loss: {loss_val:.4f}, max_mem: {max_mem_mb:.0f}M"
                )

        # end of epoch checkpoint
        checkpoint_path = (
            Path(cfg.train.work_dir) / f"checkpoint_epoch_{curr_epoch + 1}"
        )
        logger.info(f"Save checkpoint: {checkpoint_path}")
        model.save_pretrained(checkpoint_path)
        logger.info("Save checkpoint done.")

        # evaluate
        if (curr_epoch + 1) % cfg.train.get("eval_interval", 1) == 0:
            result, eval_result = do_test(cfg, model)
            logger.info(f"Epoch {curr_epoch + 1} evaluation result: {eval_result}")
            torch.save(
                result,
                Path(cfg.train.work_dir) / f"result_epoch_{curr_epoch + 1}.pth",
            )


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

    # dump config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_only:
        shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py")

    # logger
    if args.eval_only or args.no_log_file:
        log_file = None
    else:
        log_file = Path(work_dir) / f"{timestamp}.log"
    logger = get_logger("MAP", log_file=log_file)
    logger.info("Start")

    # seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = cfg.train.get("seed", 0)
    seed_all_rng(seed)
    logger.info(f"Set random seed: {seed}")

    return cfg


def clean_up():
    pass


def main():
    args = parse_args()
    cfg = setup(args)
    logger = get_logger("MAP")

    load_from = cfg.train.get("load_from", None)
    if args.load_from is not None:
        load_from = args.load_from
    if load_from is not None:
        logger.info(f"Load checkpoint: {load_from}")
        cfg.model.name_or_path = load_from
    model = instantiate(cfg.model)

    if args.init_only:
        init_path = Path(cfg.train.work_dir) / "initialized.pth"
        torch.save(model.state_dict(), init_path)
        logger.info(f"Saved initialized model: {init_path}")

    if cfg.train.get("cast_to_bf16", False):
        logger.info("Casting model to BF16")
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

    if args.eval_only:
        result, eval_result = do_test(cfg, model)
        logger.info(f"Evaluation result: {eval_result}")
        if args.out is not None:
            torch.save(result, args.out)
    else:
        do_train(cfg, model)

    clean_up()


if __name__ == "__main__":
    main()

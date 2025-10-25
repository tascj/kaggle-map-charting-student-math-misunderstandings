
import torch
from detectron2.config import LazyCall as L

from fvcore.common.param_scheduler import CompositeParamScheduler, LinearParamScheduler
from transformers import AutoTokenizer

from map_modules.data.dataset_v1 import MAPDataset

from map_modules.optim.offload_adam_gr import OffloadAdam

MODEL_NAME_OR_PATH = "zai-org/GLM-Z1-32B-0414"
TRAIN_QUERY = None
TEST_QUERY = "fold == 4"
SEED = 3768


# model config
def build_model(name_or_path=MODEL_NAME_OR_PATH):
    from map_modules.models.modeling_glm4 import Glm4ForSequenceClassification
    model = Glm4ForSequenceClassification.from_pretrained(
        name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        num_labels=1,
        local_files_only=False,
    )
    model.gradient_checkpointing_enable()
    return model


model = L(build_model)()
optimizer = L(OffloadAdam)(
    lr=1e-5,
    decoupled_weight_decay=True,
    mode="stochastic_rounding",
    verbose=10
)


# data config
def build_dataset(query, training):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, local_files_only=False
    )
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    if training:
        dataset = MAPDataset(
            csv_file="../artifacts/dtrainval.csv",
            tokenizer=tokenizer,
            query=query,
        )
    else:
        dataset = MAPDataset(
            csv_file="../artifacts/dtrainval.csv",
            tokenizer=tokenizer,
            query=query,
        )
    return dataset


def build_data_loader(dataset, batch_size, num_workers, training=True):
    collate_fn = dataset.collate_fn
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=training,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        collate_fn=collate_fn,
    )


dataloader = dict(
    train=L(build_data_loader)(
        dataset=L(build_dataset)(query=TRAIN_QUERY, training=True),
        batch_size=32,
        num_workers=4,
        training=True,
    ),
    val=L(build_data_loader)(
        dataset=L(build_dataset)(query=TEST_QUERY, training=False),
        batch_size=16,
        num_workers=4,
        training=False,
    ),
)

max_epochs = 1
lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(LinearParamScheduler)(start_value=0.001, end_value=1),
        L(LinearParamScheduler)(start_value=1, end_value=0.001),
    ],
    lengths=[0.1, 0.9],
    interval_scaling=["rescaled", "rescaled"],
)

train = dict(
    device="cuda",
    max_epochs=max_epochs,
    log_interval=10,
    eval_interval=1,
    cast_to_bf16=False,
    clip_grad=False,
    seed=SEED,
)

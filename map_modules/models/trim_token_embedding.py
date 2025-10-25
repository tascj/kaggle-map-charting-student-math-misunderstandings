import torch
from tqdm import tqdm


def collect_input_ids(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=32,
        num_workers=8,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    all_input_ids = set()

    for batch in tqdm(dataloader, desc="Getting all input ids", dynamic_ncols=True):
        for data in batch:
            all_input_ids.update(data["input_ids"].tolist())
    return all_input_ids


class RemapTokenEmbedding(torch.nn.Module):
    def __init__(self, id_map, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding((id_map != -1).sum(), embed_dim)
        self.register_buffer("id_map", id_map)

    def forward(self, input_ids):
        input_ids = self.id_map[input_ids]
        return self.embedding(input_ids)


def trim_token_embedding(embedding_module, dataset):
    all_input_ids = collect_input_ids(dataset)
    vocab_size, embed_dim = embedding_module.weight.size()
    all_input_ids = torch.tensor(sorted(all_input_ids))
    id_mapping = torch.full((vocab_size,), -1)
    id_mapping[all_input_ids] = torch.arange(all_input_ids.size(0))
    trimmed_embedding = RemapTokenEmbedding(id_mapping, embed_dim)
    trimmed_embedding.embedding.weight.data.copy_(
        embedding_module.weight.data[all_input_ids]
    )
    return trimmed_embedding

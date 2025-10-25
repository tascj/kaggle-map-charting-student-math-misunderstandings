import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

QID2ANSWER = {
    31777: "\\( 72 \\)",
    31774: "\\( \\frac{1}{12} \\)",
    32835: "\\( 6.2 \\)",
    33471: "\\( 15 \\)",
    91695: "\\( 26 \\)",
    31778: "\\( 6 \\)",
    31772: "\\( \\frac{1}{3} \\)",
    33472: "\\( \\frac{11}{15} \\)",
    109465: "Likely",
    32833: "\\( 3 \\frac{1}{3} \\)",
    104665: "\\( 48 \\) hours",
    32829: "\\( 12 \\)",
    89443: "\\( -3 \\)",
    76870: "\\( 10 \\)",
    33474: "\\( \\frac{1}{3} \\times \\frac{2}{3} \\)",
}
QID2MISCONCEPTIONS = {
    31772: ["Incomplete", "WNB"],
    31774: ["FlipChange", "Mult", "SwapDividend"],
    31777: ["Incomplete", "Irrelevant", "Wrong_Fraction"],
    31778: ["Additive", "Irrelevant", "WNB"],
    32829: ["Adding_terms", "Inverse_operation", "Not_variable"],
    32833: ["Duplication", "Inversion", "Wrong_Operation"],
    32835: [
        "Ignores_zeroes",
        "Longer_is_bigger",
        "Shorter_is_bigger",
        "Whole_numbers_larger",
    ],
    33471: ["Incomplete", "Wrong_fraction"],
    33472: [
        "Adding_across",
        "Denominator-only_change",
        "Incorrect_equivalent_fraction_addition",
    ],
    33474: ["Division", "Subtraction"],
    76870: ["Definition", "Interior", "Unknowable"],
    89443: ["Positive", "Tacking"],
    91695: ["Firstterm", "Wrong_term"],
    104665: ["Base_rate", "Multiplying_by_4"],
    109465: ["Certainty", "Scale"],
}
QID2CHOICES = {
    31772: [
        "\\( \\frac{1}{3} \\)",
        "\\( \\frac{3}{6} \\)",
        "\\( \\frac{3}{8} \\)",
        "\\( \\frac{3}{9} \\)",
    ],
    31774: [
        "\\( 3 \\)",
        "\\( \\frac{1}{12} \\)",
        "\\( \\frac{1}{3} \\)",
        "\\( \\frac{6}{2} \\)",
    ],
    31777: ["\\( 24 \\)", "\\( 48 \\)", "\\( 60 \\)", "\\( 72 \\)"],
    31778: ["\\( 3 \\)", "\\( 4 \\)", "\\( 6 \\)", "\\( 9 \\)"],
    32829: ["\\( 12 \\)", "\\( 22 \\)", "\\( 4 \\)", "\\( 48 \\)"],
    32833: [
        "\\( 3 \\frac{1}{3} \\)",
        "\\( 5 \\frac{2}{3} \\)",
        "\\( \\frac{10}{15} \\)",
        "\\( \\frac{2}{15} \\)",
    ],
    32835: ["\\( 6 \\)", "\\( 6.0001 \\)", "\\( 6.079 \\)", "\\( 6.2 \\)"],
    33471: ["\\( 15 \\)", "\\( 3 \\)", "\\( 8 \\)", "\\( 9 \\)"],
    33472: [
        "\\( \\frac{11}{15} \\)",
        "\\( \\frac{11}{30} \\)",
        "\\( \\frac{3}{15} \\)",
        "\\( \\frac{3}{8} \\)",
    ],
    33474: [
        "\\( \\frac{1}{3} \\times \\frac{2}{3} \\)",
        "\\( \\frac{1}{3}+\\frac{2}{3} \\)",
        "\\( \\frac{2}{3} \\div \\frac{1}{3} \\)",
        "\\( \\frac{2}{3}-\\frac{1}{3} \\)",
    ],
    76870: ["Not enough information", "\\( 10 \\)", "\\( 5 \\)", "\\( 6 \\)"],
    89443: ["\\( -13 \\)", "\\( -3 \\)", "\\( 13 \\)", "\\( 3 \\)"],
    91695: ["\\( 20 \\)", "\\( 22 \\)", "\\( 26 \\)", "\\( 36 \\)"],
    104665: [
        "\\( 192 \\) hours",
        "\\( 48 \\) hours",
        "\\( 64 \\) hours",
        "\\( 768 \\) hours",
    ],
    109465: ["Certain", "Impossible", "Likely", "Unlikely"],
}

DATA_FORMATS = [
    """**Question:** {QuestionText}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
""",
    """**Question:** {QuestionText}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
""",
    """**Question:** {QuestionText}
**Choices:** {MC_Answers}
**Correct Answer:** {Answer}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
""",
]


def compute_map3(probs, labels):
    map3 = 0
    for prob, label in zip(probs, labels):
        top3 = np.argsort(-prob, axis=0)[:3]
        match = top3 == label
        if match[0]:
            map3 += 1.0
        elif match[1]:
            map3 += 1.0 / 2
        elif match[2]:
            map3 += 1.0 / 3

    return map3 / len(labels)


def misconceptions2candidates(misconceptions):
    ret = [
        "False_Correct:NA",
        "False_Neither:NA",
        "True_Correct:NA",
        "True_Neither:NA",
    ]
    for misconception in misconceptions:
        ret.append(f"False_Misconception:{misconception}")
        ret.append(f"True_Misconception:{misconception}")
    return ret


class MAPDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        query=None,
        has_system_role=True,
        train_mode=False,
        train_formats=[2],
        test_format=2,
    ):
        df = pd.read_csv(csv_file).fillna("NA")
        if query is not None:
            df = df.query(query).reset_index(drop=True)
        df["label"] = df["Category"] + ":" + df["Misconception"]
        df["Answer"] = df["QuestionId"].map(QID2ANSWER)
        df["MC_Answers"] = df["QuestionId"].map(QID2CHOICES)
        df["MisconceptionCandidates"] = df["QuestionId"].map(QID2MISCONCEPTIONS)
        df["label_candidates"] = df["MisconceptionCandidates"].map(
            misconceptions2candidates
        )
        # cat_id is the index of the label in the label_candidates
        df["cat_id"] = df.apply(
            lambda x: x["label_candidates"].index(x["label"]), axis=1
        )
        self.df = df
        self.tokenizer = tokenizer
        self.has_system_role = has_system_role
        self.train_mode = train_mode
        self.train_formats = train_formats
        self.test_format = test_format

    def __len__(self):
        # return 100
        return len(self.df)

    def __getitem__(self, idx):
        if self.train_mode:
            ret = []
            for train_format in self.train_formats:
                ret.append(self._getitem(idx, train_format))
            return ret
        return [self._getitem(idx, self.test_format)]

    def _getitem(self, idx, format):
        data = self.df.iloc[idx]
        data_format = DATA_FORMATS[format]

        messages = [
            {"role": "user", "content": data_format.format(**data)},
        ]

        prefix_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prefix = self.tokenizer(
            prefix_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        input_ids = [prefix]
        offset = prefix.size(0)
        position_ids = [torch.arange(offset)]
        suffix_ids = [torch.full_like(prefix, -1)]

        suffix_id = 0
        for candidate in data["label_candidates"]:
            suffix_text = f"{candidate}{self.tokenizer.eos_token}"
            suffix = self.tokenizer(
                suffix_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            input_ids.append(suffix)
            position_ids.append(torch.arange(offset, offset + suffix.size(0)))
            suffix_ids.append(torch.full_like(suffix, suffix_id))
            suffix_id += 1

        last_tokens = torch.tensor([_.size(0) for _ in input_ids]).cumsum(0)[1:] - 1
        input_ids = torch.cat(input_ids)
        position_ids = torch.cat(position_ids)
        suffix_ids = torch.cat(suffix_ids)
        num_candidates = len(data["label_candidates"])
        label = torch.zeros(num_candidates, dtype=torch.float)
        label[data["cat_id"]] = 1.0

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            suffix_ids=suffix_ids,
            last_tokens=last_tokens,
            label=label,
            num_candidates=num_candidates,
        )

    def collate_fn(self, samples, block_size=128):
        _samples = []
        for sample in samples:
            _samples.extend(sample)
        samples = _samples
        input_ids = torch.cat([sample["input_ids"] for sample in samples])
        position_ids = torch.cat([sample["position_ids"] for sample in samples])
        suffix_ids = torch.cat([sample["suffix_ids"] for sample in samples])
        labels = [sample["label"] for sample in samples]
        last_tokens = []
        doc_ids = []
        doc_id = 0
        offset = 0
        for sample in samples:
            last_tokens.append(sample["last_tokens"] + offset)
            offset += sample["input_ids"].numel()
            doc_ids.append(torch.zeros_like(sample["input_ids"]) + doc_id)
            doc_id += 1
        doc_ids = torch.cat(doc_ids)
        last_tokens = torch.cat(last_tokens)

        pad_size = block_size - input_ids.size(0) % block_size
        input_ids = F.pad(input_ids, [0, pad_size], value=0)
        position_ids = F.pad(position_ids, [0, pad_size], value=0)
        suffix_ids = F.pad(suffix_ids, [0, pad_size], value=-2)
        doc_ids = F.pad(doc_ids, [0, pad_size], value=-2)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            suffix_ids=suffix_ids,
            last_tokens=last_tokens,
            doc_ids=doc_ids,
            label=labels,
            num_candidates=[sample["num_candidates"] for sample in samples],
        )

    def evaluate(self, probs):
        labels = self.df["cat_id"].values
        metric_loss = np.mean(
            [-np.log(prob[label]) for prob, label in zip(probs, labels)]
        )
        metric_map3 = compute_map3(probs, labels)
        return {"log_loss": metric_loss, "map@3": metric_map3}


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", local_files_only=False
    )
    ds = MAPDataset(
        csv_file="../artifacts/dtrainval.csv",
        tokenizer=tokenizer,
        query="(fold != 0)",
        train_mode=True,
    )
    dl = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn, shuffle=True)
    d = next(iter(dl))

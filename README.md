# MAP - Charting Student Math Misunderstandings

[Competition](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
[Solution Writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution)

## Requirements

### Hardware

Memory: 144GB or more
GPU: 96GB or more
NVIDIA RTX PRO 6000 Blackwell Workstation Edition was used.

### Software

Check `docker/Dockerfile`
```
docker build -t kaggle-map -f docker/Dockerfile docker/
docker run -it --rm --gpus all --ipc=host \
  -v $(pwd)/..:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  kaggle-map
```

Directory structure should be like this:
```
.
├── data
│   ├── map-charting-student-math-misunderstandings.zip
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── artifacts
│   ├── dtrainval.csv
│   └── dtrainval_qwen3_235b_a22b_thinking_2507_fp8.parquet
└── kaggle-map-charting-student-math-misunderstandings
    ├── LICENSE
    ├── README.md
    ├── configs
    └── ...
```

## Prepare Data

Make 5-fold splits of the data.
```
python scripts/prepare_data.py
```

Generate responses using `Qwen3-235B-A22B-Thinking-2507-FP8`.
This requires A100-SXM4-80GB x4 and takes about 1h45m.
Use the result in `data/dtrainval_qwen3_235b_a22b_thinking_2507_fp8.parquet`.
```
python scripts/prepare_response.py
```

## Training

Check `train.sh` for details.


## Inference

Check `convert.sh` for details.

[Kaggle Notebook](https://www.kaggle.com/code/tascj0/map-submit)

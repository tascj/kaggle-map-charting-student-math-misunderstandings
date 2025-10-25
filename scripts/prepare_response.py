import pickle

import pandas as pd
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from transformers import AutoTokenizer

Q1 = """You are analyzing a student's response to a math question. 
Classify the response using these rules:

1. **Answer correctness** → Is the student's answer correct?  
   - Label as True or False in Category (e.g., True_Correct).  

2. **Explanation quality** → Does the explanation show:  
   - Correct: proper mathematical reasoning,  
   - Misconception: a clear mathematical error or misunderstanding,  
   - Neither: too vague, irrelevant, or unclear.  

3. **Misconception detail** → If Misconception, briefly name it. Otherwise use NA.  

---

**Question:** {QuestionText}  
**Student Answer:** {MC_Answer}  
**Student Explanation:** {StudentExplanation}  

---

Respond in this format only:  
- Category: [True/False]_[Correct/Misconception/Neither]  
- Misconception: [Specific misconception OR NA]  
"""
A1 = """- Category: {Category}
- Misconception: {Misconception}
"""

Q2 = """Give a short justification for why this categorization was chosen."""


def format_prompt(tokenizer, row):
    messages = [
        {"role": "user", "content": Q1.format(**row)},
        {"role": "assistant", "content": A1.format(**row)},
        {"role": "user", "content": Q2},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text


def get_content(tokenizer, output_ids):
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content


if __name__ == "__main__":
    df = pd.read_csv("../artifacts/dtrainval.csv").fillna("NA")
    # df = df.sample(100)

    model_id = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    texts = [format_prompt(tokenizer, row) for _, row in df.iterrows()]

    backend_config = TurbomindEngineConfig(
        tp=4,
        enable_prefix_caching=True,
    )
    pipe = pipeline(model_id, backend_config=backend_config)

    gen_config = GenerationConfig(
        top_p=0.95,
        top_k=20,
        temperature=0.6,
        min_p=0,
        max_new_tokens=4096,
        random_seed=0,
    )
    responses = pipe(
        texts,
        gen_config=gen_config,
        do_preprocess=False,
        use_tqdm=True,
    )

    short_responses = [
        get_content(tokenizer, response.token_ids) for response in responses
    ]
    long_responses = [response.text for response in responses]
    df["short_response"] = short_responses
    df["long_response"] = long_responses
    df.to_parquet(
        "../artifacts/dtrainval_qwen3_235b_a22b_thinking_2507_fp8.parquet",
        index=False,
    )
    with open(
        "../artifacts/raw_responses_qwen3_235b_a22b_thinking_2507_fp8.pkl", "wb"
    ) as f:
        pickle.dump(responses, f)

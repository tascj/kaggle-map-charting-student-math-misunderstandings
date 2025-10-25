from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    Path("../artifacts").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("../data/train.csv")
    columns = [
        "QuestionId",
        "QuestionText",
        "MC_Answer",
        "StudentExplanation",
        "Category",
        "Misconception",
    ]
    df = df.drop_duplicates(subset=columns).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(df, df["Category"]))
    df["fold"] = -1
    for fold, (_, val_index) in enumerate(splits):
        df.loc[val_index, "fold"] = fold
    df.to_csv("../artifacts/dtrainval.csv", index=False)
